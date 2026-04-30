"""
Патч для discord-ext-voice-recv: добавляет DAVE (E2EE/MLS) decryption.

Discord на современных каналах форсит DAVE поверх transport-encryption.
voice_recv 0.5.3a180 НЕ реализует DAVE на приёме — opus получает encrypted
payload и крашится / выдаёт garbage PCM.

Этот патч перехватывает PacketDecoder._decode_packet и расшифровывает
payload через davey.DaveSession.decrypt() ПЕРЕД тем как отдать opus.

Использование:
    from audio.dave_patch import apply_dave_patch
    apply_dave_patch()  # вызвать один раз ДО создания Bot/voice connect
"""
import atexit
from typing import Dict, Any

_applied = False
_stats: Dict[str, int] = {
    "opus_ok": 0,
    "opus_err": 0,
    "dave_decrypted": 0,
    "dave_skip_no_session": 0,
    "dave_decrypt_fail": 0,
}


def get_stats() -> Dict[str, int]:
    return dict(_stats)


def apply_dave_patch(verbose: bool = True) -> None:
    """Применяет DAVE-decryption patch к discord-ext-voice-recv. Идемпотентно."""
    global _applied
    if _applied:
        return

    import davey
    from discord.ext.voice_recv import router as _router_mod
    from discord.ext.voice_recv import opus as _opus_mod
    from discord.opus import OpusError

    _orig_decode = _opus_mod.PacketDecoder._decode_packet

    def _decode_with_dave(self, packet):
        try:
            vc = self.sink.voice_client
            state = getattr(vc, "_connection", None)
            dave = getattr(state, "dave_session", None) if state else None

            if dave is not None and dave.ready and dave.protocol_version > 0:
                try:
                    sender_id = self._cached_id or vc._get_id_from_ssrc(self.ssrc)
                    if sender_id and packet:
                        plain = dave.decrypt(int(sender_id),
                                             davey.MediaType.audio,
                                             bytes(packet.decrypted_data))
                        packet.decrypted_data = plain
                        _stats["dave_decrypted"] += 1
                except Exception as e:
                    _stats["dave_decrypt_fail"] += 1
                    if verbose and _stats["dave_decrypt_fail"] <= 3:
                        print(f"[DAVE] decrypt error ssrc={self.ssrc}: {e!r}", flush=True)
                    raise OpusError(-1)
            elif dave is not None and not dave.ready:
                _stats["dave_skip_no_session"] += 1

            result = _orig_decode(self, packet)
            _stats["opus_ok"] += 1
            return result
        except OpusError:
            _stats["opus_err"] += 1
            raise

    _opus_mod.PacketDecoder._decode_packet = _decode_with_dave

    # Делаем router устойчивым к OpusError — один битый пакет не убивает поток
    def _safe_do_run(self):
        while not self._end_thread.is_set():
            self.waiter.wait()
            with self._lock:
                for decoder in self.waiter.items:
                    try:
                        data = decoder.pop_data()
                    except OpusError:
                        continue
                    except Exception as e:
                        if verbose:
                            print(f"[voice_recv] non-opus error: {e!r}", flush=True)
                        continue
                    if data is not None:
                        try:
                            self.sink.write(data.source, data)
                        except Exception as e:
                            if verbose:
                                print(f"[voice_recv] sink.write error: {e!r}", flush=True)

    _router_mod.PacketRouter._do_run = _safe_do_run

    if verbose:
        atexit.register(lambda: print(f"[DAVE STATS] {_stats}", flush=True))
        print("[DAVE] voice_recv пропатчен с DAVE-decryption", flush=True)

    _applied = True


async def wait_dave_ready(voice_client, timeout: float = 5.0) -> bool:
    """
    Ждёт пока DAVE сессия не будет готова (после connect).
    Возвращает True если готова или DAVE не используется на канале.
    """
    import asyncio
    state = getattr(voice_client, "_connection", None)
    if state is None:
        return False
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        ver = getattr(state, "dave_protocol_version", 0)
        if ver == 0:
            return True  # DAVE не активен на канале
        dave = getattr(state, "dave_session", None)
        if dave is not None and dave.ready:
            return True
        await asyncio.sleep(0.1)
    return False
