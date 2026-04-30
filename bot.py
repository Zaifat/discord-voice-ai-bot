"""
Voice AI Bot — финальный pipeline.

Audio → VAD → STT → Memory → LLM → TTS → Voice output.

Команды:
    !join / !зайди       — войти в твой голосовой канал
    !leave / !выйди      — выйти
    !persona <name>      — сменить персону (default/toxic/sarcastic/calm/hyper/angry)
    !personas            — список персон
    !clear               — очистить краткосрочную историю
    !forget              — очистить долгосрочную память (Chroma)
    !remember <фраза>    — явно сохранить факт в долгосрочную память
    !status              — состояние бота
"""
from __future__ import annotations
import asyncio
import os
import sys
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

import discord
from discord.ext import commands, voice_recv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    # stdout/stderr может быть file pipe без .reconfigure (Start-Process redirect)
    pass

# DAVE patch ДО любых импортов discord
from audio.dave_patch import apply_dave_patch, wait_dave_ready
apply_dave_patch()

# Тяжёлые импорты — для прогрева
from audio.vad import SpeechSegmenter
import audio.stt as stt_mod
import audio.tts as tts_mod
import llm.client as llm_mod
from memory.store import ConversationHistory, LongTermMemory
from personality.engine import get_persona, set_persona, list_personas, PERSONAS

from config import DISCORD_TOKEN, AUTO_JOIN_CHANNEL, DEBUG_DIR, ROOT


# ─── Глобальное состояние ─────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states    = True
intents.guilds          = True
intents.members         = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Per-guild state
class GuildState:
    __slots__ = ("vc", "history", "ltm", "speaking", "audio_q", "worker_task",
                 "user_segmenters", "user_leftovers", "user_frame_count",
                 "last_speaker_id", "interrupt_requested", "mode",
                 "passive_chance", "last_reply_at")
    def __init__(self):
        self.vc: voice_recv.VoiceRecvClient | None = None
        self.history: ConversationHistory = ConversationHistory(max_messages=20)
        self.ltm: LongTermMemory = _shared_ltm
        self.speaking: bool = False
        self.audio_q: asyncio.Queue | None = None
        self.worker_task: asyncio.Task | None = None
        self.user_segmenters: dict[int, SpeechSegmenter] = {}
        self.user_leftovers: dict[int, bytes] = {}
        self.user_frame_count: dict[int, int] = {}
        # uid того, кто только что обращался к боту (для разрешения barge-in)
        self.last_speaker_id: int = 0
        # Set'ится в True когда worker сделал stop() для прерывания озвучки
        self.interrupt_requested: bool = False
        # Режим: "all" (на всё), "name" (только по имени), "listener" (slушает, иногда вставляет)
        self.mode: str = "all"
        # В listener-режиме: вероятность что бот вставит реплику (0..1)
        self.passive_chance: float = 0.15
        # В режиме name: время последнего ответа бота, чтоб разрешать
        # follow-up без обращения по имени в течение N секунд.
        self.last_reply_at: float = 0.0

_shared_ltm = LongTermMemory()
_states: dict[int, GuildState] = {}

def _state(guild_id: int) -> GuildState:
    if guild_id not in _states:
        _states[guild_id] = GuildState()
    return _states[guild_id]


# ─── Sink (поток PCM из Discord в asyncio queue) ──────────────────────────────
class StreamSink(voice_recv.AudioSink):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        super().__init__()
        self.loop = loop
        self.queue = queue

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData):
        if user is None or user.bot:
            return
        try:
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, (user.id, user.display_name, data.pcm)
            )
        except RuntimeError:
            pass

    def cleanup(self):
        pass


def _pcm_stereo48_to_mono16_f32(pcm_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(arr) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(arr) % 2:
        arr = arr[:-1]
    mono48 = ((arr[0::2].astype(np.int32) + arr[1::2].astype(np.int32)) >> 1).astype(np.int16)
    f = mono48.astype(np.float32) / 32768.0
    n = (len(f) // 3) * 3
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return sp_signal.resample_poly(f[:n], 1, 3).astype(np.float32)


# ─── Главный воркер: VAD → STT → LLM → TTS ────────────────────────────────────
async def _audio_worker(guild_id: int):
    st = _state(guild_id)
    interrupt_counter: dict[int, int] = {}
    skipped_while_speaking: dict[int, int] = {}
    while True:
        try:
            uid, name, pcm = await asyncio.wait_for(st.audio_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        # Бот говорит — игнорируем входящий звук полностью.
        # Барж-ин отключён: бот всегда договаривает свою фразу.
        if st.speaking:
            continue

        seg = st.user_segmenters.get(uid)
        if seg is None:
            seg = st.user_segmenters[uid] = SpeechSegmenter(
                threshold=0.4,
                min_speech_ms=200,       # ловит даже короткие "да", "нет"
                min_silence_ms=350,      # быстрее закрывать фразу (было 500)
                pre_pad_ms=150,
                post_pad_ms=150,         # меньше хвоста
            )
            print(f"[BOT] Слышу нового говорящего: {name}", flush=True)

        # Накапливаем PCM, обрабатываем кусками кратными 12 байт (3 stereo frames)
        buf = st.user_leftovers.get(uid, b"") + pcm
        usable = len(buf) - (len(buf) % 12)
        if usable <= 0:
            st.user_leftovers[uid] = buf
            continue
        chunk = buf[:usable]
        st.user_leftovers[uid] = buf[usable:]

        mono16 = _pcm_stereo48_to_mono16_f32(chunk)
        if len(mono16) == 0:
            continue

        completed = seg.feed(mono16)
        for phrase in completed:
            asyncio.create_task(_handle_phrase(guild_id, uid, name, phrase))

        # Диагностика: каждые ~50 chunks (≈1с) выводим состояние VAD
        cnt = st.user_frame_count.get(uid, 0) + 1
        st.user_frame_count[uid] = cnt
        if cnt % 50 == 0:
            in_speech = "[SPK]" if seg.is_in_speech else "[sil]"
            rms_now = float(np.sqrt(np.mean(mono16 ** 2))) if len(mono16) else 0
            print(f"[VAD] {name}: {in_speech} rms={rms_now:.3f}", flush=True)


def _bot_names(guild_id: int) -> list[str]:
    """
    Имена под которыми бот откликается. Включает русские транскрипции
    и частые искажения от Whisper.
    """
    names: list[str] = [
        "бот", "ии", "ai",
        # Whisper варианты для VoiceAI:
        "voice", "voiceai", "войс", "войсай", "voice id", "вой",
    ]
    guild = bot.get_guild(guild_id)
    if guild and guild.me:
        if guild.me.display_name:
            names.append(guild.me.display_name.lower())
        if guild.me.name:
            names.append(guild.me.name.lower())
    if bot.user and bot.user.name:
        names.append(bot.user.name.lower())
    return list({n for n in names if len(n) > 1})


def _should_respond(st: GuildState, text: str, user_name: str) -> bool:
    """Решает, отвечать ли боту в зависимости от текущего mode."""
    if st.mode == "all":
        return True

    if st.mode == "name":
        # Если бот недавно отвечал (≤ 10с) — это продолжение диалога, имя не нужно
        if time.monotonic() - st.last_reply_at < 10.0:
            return True
        # Иначе — нужно явное обращение
        text_lower = text.lower()
        for name in _bot_names(0 if st.vc is None else st.vc.guild.id):
            if name in text_lower:
                return True
        print(f"  ⊘ режим 'name' — обращение не к боту, игнор", flush=True)
        return False

    if st.mode == "listener":
        # Иногда вставляем реплику со случайным шансом
        import random
        if random.random() < st.passive_chance:
            print(f"  ✓ режим 'listener' — встрял ({st.passive_chance*100:.0f}% шанс)", flush=True)
            return True
        print(f"  ⊘ режим 'listener' — пропускаю", flush=True)
        return False

    return True


async def _handle_phrase(guild_id: int, user_id: int, user_name: str, audio: np.ndarray):
    """Полный pipeline для одной фразы."""
    st = _state(guild_id)
    if st.speaking:
        return  # бот говорит — игнор

    dur = len(audio) / 16000
    t_total = time.monotonic()

    # 1. STT
    t0 = time.monotonic()
    text = await stt_mod.transcribe_pcm16(audio, language="ru")
    t_stt = time.monotonic() - t0
    if not text:
        print(f"[{user_name}] {dur:.2f}с → (нет текста)", flush=True)
        return
    print(f"[{user_name}] ({dur:.2f}с, STT {t_stt*1000:.0f}мс) > {text}", flush=True)

    # Запоминаем кто только что говорил — для allowed-barge-in
    st.last_speaker_id = user_id

    # ─── Фильтр режима: should_respond() решает отвечать или просто слушать ──
    if not _should_respond(st, text, user_name):
        # В режимах name/listener просто сохраняем фразу в историю и выходим
        st.history.add_user(user_name, text)
        return

    # 2. Memory: сохраняем в RAM-историю; vector search ОТКЛЮЧЁН в hot-path
    # потому что bge-m3 и gemma3 делят VRAM в Ollama — каждый embed выгружает LLM.
    # Долгосрочная память используется только через !remember/!forget команды.
    st.history.add_user(user_name, text)
    memories = []  # await st.ltm.search(guild_id, text, top_k=2)
    # Сохранять в LTM в фоне можно, не блокирует LLM:
    # asyncio.create_task(st.ltm.add(guild_id, f"{user_name} сказал: {text}", speaker=user_name))
    memory_block = ""
    if memories:
        # Берём только реально близкие, и не текущую же реплику
        relevant = [m for m, _, dist in memories
                    if dist < 0.5 and text not in m]
        if relevant:
            memory_block = "Что ты знаешь:\n" + "\n".join(f"- {m}" for m in relevant)

    # 3. LLM
    persona = get_persona(guild_id)
    speaker_line = f"Сейчас с тобой говорит пользователь по имени {user_name}."
    system = f"{persona.system_prompt}\n\n{speaker_line}"
    if memory_block:
        system = f"{system}\n\n{memory_block}"

    t0 = time.monotonic()
    reply = await llm_mod.generate_reply(st.history.get_messages(), system=system)
    t_llm = time.monotonic() - t0
    if not reply:
        return
    print(f"  → ({t_llm*1000:.0f}мс) {reply}", flush=True)

    st.history.add_assistant(reply)
    st.last_reply_at = time.monotonic()

    # 4. TTS
    t0 = time.monotonic()
    wav = await tts_mod.synthesize(reply, voice=persona.voice,
                                    emotion=persona.emotion, rate=persona.rate)
    t_tts = time.monotonic() - t0
    if not wav:
        return

    # 5. Воспроизведение
    await _speak_wav(guild_id, wav)
    t_total = time.monotonic() - t_total
    print(f"  ⏱ STT {t_stt*1000:.0f} + LLM {t_llm*1000:.0f} + TTS {t_tts*1000:.0f} = {t_total*1000:.0f}мс", flush=True)


# ─── Воспроизведение TTS ──────────────────────────────────────────────────────
_TTS_TEMP_DIR = ROOT / "debug" / "tts_temp"
_TTS_TEMP_DIR.mkdir(exist_ok=True, parents=True)
_tts_counter = 0


async def _speak_wav(guild_id: int, wav_bytes: bytes):
    st = _state(guild_id)
    if not st.vc or not st.vc.is_connected():
        return

    global _tts_counter
    _tts_counter += 1
    path = _TTS_TEMP_DIR / f"speak_{_tts_counter}.wav"
    path.write_bytes(wav_bytes)

    # Дожидаемся пока не доиграет предыдущий
    while st.vc.is_playing():
        await asyncio.sleep(0.05)

    st.speaking = True
    done = asyncio.Event()
    loop = asyncio.get_event_loop()

    def _after(error):
        loop.call_soon_threadsafe(done.set)
        try: path.unlink()
        except Exception: pass

    try:
        # FFmpeg ресемплит до 48kHz для Discord
        src = discord.FFmpegPCMAudio(str(path))
        st.vc.play(src, after=_after)
    except Exception as e:
        print(f"[TTS] play error: {e!r}", flush=True)
        st.speaking = False
        return

    # Ждём завершения проигрывания, но не дольше 30с (анти-залипание)
    interrupted = False
    try:
        await asyncio.wait_for(done.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("[TTS] play() timeout — форсируем stop", flush=True)
        try:
            if st.vc and st.vc.is_playing():
                st.vc.stop()
        except Exception: pass

    interrupted = st.interrupt_requested
    st.interrupt_requested = False  # сбрасываем флаг

    if not interrupted:
        # Обычное завершение — пауза + сброс эха/feedback
        await asyncio.sleep(0.5)
        for seg in st.user_segmenters.values():
            seg.reset()
        st.user_leftovers.clear()
        if st.audio_q is not None:
            while not st.audio_q.empty():
                try: st.audio_q.get_nowait()
                except: break
    else:
        # Барж-ин — даём фразе завершиться, не реsetting segmenters
        print("[TTS] прервано пользователем — VAD продолжает слушать", flush=True)
        await asyncio.sleep(0.1)

    st.speaking = False
    print(f"[TTS] play() done, speaking={st.speaking}", flush=True)


# ─── Команды ──────────────────────────────────────────────────────────────────
@bot.command(name="join", aliases=["зайди", "заходи"])
async def join(ctx: commands.Context):
    if not ctx.author.voice:
        await ctx.send("Зайди в голосовой канал и зови."); return
    await _join_channel(ctx.author.voice.channel)
    await ctx.send(f"Зашёл в **{ctx.author.voice.channel.name}**. Говори.")


async def _join_channel(channel: discord.VoiceChannel):
    st = _state(channel.guild.id)
    if st.vc and st.vc.is_connected():
        await st.vc.move_to(channel)
    else:
        st.vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    # Дожидаемся DAVE до 10с, но даже если не дождались — listen() всё равно
    # запускаем: patch decrypt'ит каждый пакет по мере появления session.ready.
    if not await wait_dave_ready(st.vc, timeout=10):
        print("[BOT] DAVE сессия ещё не готова — слушаю в любом случае", flush=True)

    loop = asyncio.get_event_loop()
    st.audio_q = asyncio.Queue(maxsize=4000)
    if st.worker_task:
        st.worker_task.cancel()
    st.worker_task = asyncio.create_task(_audio_worker(channel.guild.id))

    sink = StreamSink(loop, st.audio_q)
    st.vc.listen(sink)
    print(f"[BOT] Слушаю в {channel.name}", flush=True)


@bot.command(name="leave", aliases=["выйди"])
async def leave(ctx: commands.Context):
    st = _state(ctx.guild.id)
    if st.vc and st.vc.is_connected():
        try: st.vc.stop_listening()
        except Exception: pass
        if st.worker_task:
            st.worker_task.cancel()
            st.worker_task = None
        await st.vc.disconnect()
        st.vc = None
        st.user_segmenters.clear()
        st.user_leftovers.clear()
        await ctx.send("Вышел.")
    else:
        await ctx.send("Я не в канале.")


@bot.command(name="persona")
async def persona_cmd(ctx: commands.Context, name: str | None = None):
    if not name:
        cur = get_persona(ctx.guild.id)
        await ctx.send(f"Текущая: **{cur.name}** ({cur.description})\n"
                       f"Доступные: {', '.join(list_personas())}")
        return
    try:
        p = set_persona(ctx.guild.id, name.lower())
        # Очищаем историю при смене персоны (новый характер)
        _state(ctx.guild.id).history.clear()
        await ctx.send(f"Персона: **{p.name}** — {p.description} (голос: {p.voice})")
    except ValueError as e:
        await ctx.send(str(e))


@bot.command(name="personas")
async def personas_cmd(ctx: commands.Context):
    lines = []
    for name, p in PERSONAS.items():
        lines.append(f"**{name}** — {p.description} _(голос: {p.voice}, эмоция: {p.emotion})_")
    await ctx.send("\n".join(lines))


@bot.command(name="clear", aliases=["очисти"])
async def clear_cmd(ctx: commands.Context):
    _state(ctx.guild.id).history.clear()
    await ctx.send("История очищена.")


@bot.command(name="forget")
async def forget_cmd(ctx: commands.Context):
    _shared_ltm.clear(ctx.guild.id)
    await ctx.send("Долгосрочная память забыта.")


@bot.command(name="remember")
async def remember_cmd(ctx: commands.Context, *, fact: str):
    await _shared_ltm.add(ctx.guild.id, fact, speaker=ctx.author.display_name)
    await ctx.send(f"Запомнил: _{fact}_")


@bot.command(name="status", aliases=["статус"])
async def status_cmd(ctx: commands.Context):
    st = _state(ctx.guild.id)
    cur = get_persona(ctx.guild.id)
    ltm_stats = _shared_ltm.stats(ctx.guild.id)
    in_voice = "да" if (st.vc and st.vc.is_connected()) else "нет"
    speaking = "да" if st.speaking else "нет"
    mode_desc = {
        "all": "all — отвечает на все",
        "name": f"name — отвечает только если упомянуть имя ({', '.join(_bot_names(ctx.guild.id))})",
        "listener": f"listener — слушает, иногда встревает ({st.passive_chance*100:.0f}% шанс)",
    }.get(st.mode, st.mode)
    await ctx.send(
        f"В голосовом: **{in_voice}** | "
        f"Говорит: **{speaking}** | "
        f"Режим: **{mode_desc}**\n"
        f"Персона: **{cur.name}** | "
        f"История: {len(st.history)} реплик | "
        f"Долгосрочная память: {ltm_stats['total_memories']} фактов"
    )


@bot.command(name="mode")
async def mode_cmd(ctx: commands.Context, name: str | None = None,
                    chance: float | None = None):
    """!mode all | name | listener [chance]"""
    st = _state(ctx.guild.id)
    if not name:
        await ctx.send(
            f"Текущий режим: **{st.mode}**\n"
            f"Доступные:\n"
            f"  `!mode all` — отвечать на все реплики (текущий по умолчанию)\n"
            f"  `!mode name` — только когда обращаются по имени ({', '.join(_bot_names(ctx.guild.id))})\n"
            f"  `!mode listener [0.0..1.0]` — слушать, иногда вставлять реплику (по умолчанию 15%)"
        )
        return
    name = name.lower()
    if name not in ("all", "name", "listener"):
        await ctx.send("Режимы: `all`, `name`, `listener`")
        return
    st.mode = name
    if name == "listener" and chance is not None:
        st.passive_chance = max(0.0, min(1.0, chance))
    await ctx.send(f"Режим: **{st.mode}**" +
                   (f" (chance={st.passive_chance*100:.0f}%)" if st.mode == "listener" else ""))


# ─── Прогрев и автозапуск ─────────────────────────────────────────────────────
async def _warmup_all():
    """
    Прогрев тяжёлых моделей.
    Важно: LLM прогревается ПОСЛЕДНИМ и embeddings НЕ прогреваются —
    Ollama держит только одну модель в VRAM, embed выгружает gemma.
    """
    print("[BOT] Прогрев Whisper...", flush=True)
    await stt_mod.warmup()
    print("[BOT] Прогрев TTS...", flush=True)
    await tts_mod.warmup()
    print("[BOT] Прогрев LLM (последним — чтобы остался в VRAM)...", flush=True)
    await llm_mod.warmup()
    print("[BOT] Все модели прогреты — готов к разговору", flush=True)


async def _llm_keepalive_loop():
    """
    Каждые 60с делаем пустой ping LLM чтобы Ollama не выгружала модель из VRAM.
    Без этого первый запрос после простоя занимает 5-10с (cold load).
    """
    while True:
        try:
            await asyncio.sleep(20)
            t0 = time.monotonic()
            await llm_mod.keep_loaded()  # preload без генерации (тише в логе)
            dt = time.monotonic() - t0
            if dt > 1.5:
                print(f"[LLM] keepalive {dt:.1f}с (модель перезагрузилась)", flush=True)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[LLM] keepalive error: {e!r}", flush=True)


@bot.event
async def on_ready():
    print(f"[BOT] {bot.user} | Серверов: {len(bot.guilds)}", flush=True)
    await _warmup_all()
    # Запускаем keepalive чтобы LLM не выгружалась
    asyncio.create_task(_llm_keepalive_loop())

    if AUTO_JOIN_CHANNEL:
        for guild in bot.guilds:
            ch = guild.get_channel(AUTO_JOIN_CHANNEL)
            if ch and isinstance(ch, discord.VoiceChannel):
                print(f"[BOT] Автоподключение к {ch.name}", flush=True)
                await _join_channel(ch)
                break


@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return
    st = _state(member.guild.id)
    if st.vc and st.vc.is_connected():
        if not [m for m in st.vc.channel.members if not m.bot]:
            print("[BOT] Канал пуст — выхожу", flush=True)
            try: st.vc.stop_listening()
            except Exception: pass
            if st.worker_task:
                st.worker_task.cancel()
                st.worker_task = None
            await st.vc.disconnect()
            st.vc = None


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("DISCORD_TOKEN не задан в .env"); sys.exit(1)
    bot.run(DISCORD_TOKEN)
