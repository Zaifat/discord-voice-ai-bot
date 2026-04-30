"""
TTS через Yandex SpeechKit — натуральные русские neural-голоса.

Голоса (v3 neural):
  alena    — женский (neutral / good)
  filipp   — мужской (neutral)
  jane     — женский (good / friendly / evil)
  omazh    — женский (neutral / evil)
  zahar    — мужской (neutral / good)
  ermil    — мужской (neutral / good)
  + masha, dasha, julia, lera, marina, madirus
"""
from __future__ import annotations
import asyncio
import io
import os
import re
import time
import wave
from pathlib import Path
from typing import Optional

import aiohttp


from dotenv import load_dotenv
from config import ROOT
load_dotenv(ROOT / ".env")

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "").strip()
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "").strip()
YANDEX_TTS_URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

DEFAULT_VOICE = "zahar"         # мужской, выразительный (как в yandex_angry)
AVAILABLE_VOICES = ["alena", "filipp", "jane", "omazh", "zahar", "ermil",
                    "masha", "dasha", "julia", "lera", "marina", "madirus"]

# Yandex поддерживает emotion на некоторых голосах
EMOTION_PRESETS = {
    "neutral":   {"voice": "jane",   "emotion": "neutral", "speed": "1.0"},
    "happy":     {"voice": "jane",   "emotion": "good",    "speed": "1.05"},
    "sad":       {"voice": "omazh",  "emotion": "neutral", "speed": "0.9"},
    "angry":     {"voice": "zahar",  "emotion": "neutral", "speed": "1.05"},
    "sarcastic": {"voice": "ermil",  "emotion": "neutral", "speed": "0.95"},
    "toxic":     {"voice": "zahar",  "emotion": "neutral", "speed": "1.0"},
    "calm":      {"voice": "alena",  "emotion": "neutral", "speed": "0.95"},
}

_session: aiohttp.ClientSession | None = None


def _digits_to_words(text: str) -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text
    def _conv(m):
        n = int(m.group(0))
        try:    return num2words(n, lang="ru")
        except: return m.group(0)
    return re.sub(r"\b\d{1,9}\b", _conv, text)


def _clean_for_tts(text: str) -> str:
    t = re.sub(r"[\U0001F000-\U0001FFFF\U00002600-\U000027BF]", "", text)
    t = re.sub(r"[*_`#~<>\[\]()]+", " ", t)
    t = _digits_to_words(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    return _session


async def warmup() -> None:
    print(f"[TTS] Yandex SpeechKit...", flush=True)
    if not YANDEX_API_KEY:
        print(f"[TTS] ⚠ YANDEX_API_KEY не задан в .env", flush=True)
        return
    t0 = time.monotonic()
    try:
        await synthesize("ок", emotion="neutral")
        print(f"[TTS] Yandex готов ({time.monotonic()-t0:.1f}с) | voice={DEFAULT_VOICE}", flush=True)
    except Exception as e:
        print(f"[TTS] warmup error: {e!r}", flush=True)


async def synthesize(text: str, voice: str = DEFAULT_VOICE,
                     emotion: str | None = None, rate: float = 1.0) -> Optional[bytes]:
    """
    Возвращает WAV-байты (48kHz mono int16). None при ошибке.
    """
    if not text or not text.strip() or not YANDEX_API_KEY:
        return None
    text = _clean_for_tts(text)
    if not text:
        return None

    if emotion and emotion in EMOTION_PRESETS:
        p = EMOTION_PRESETS[emotion]
        voice = p["voice"]
        emo_str = p["emotion"]
        speed_str = p["speed"]
    else:
        if voice not in AVAILABLE_VOICES:
            voice = DEFAULT_VOICE
        emo_str = "neutral"
        speed_str = f"{max(0.1, min(3.0, rate)):.2f}"

    data = {
        "text": text,
        "voice": voice,
        "emotion": emo_str,
        "speed": speed_str,
        "lang": "ru-RU",
        "format": "lpcm",
        "sampleRateHertz": "48000",
    }
    if YANDEX_FOLDER_ID:
        data["folderId"] = YANDEX_FOLDER_ID
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}"}

    try:
        sess = await _get_session()
        async with sess.post(YANDEX_TTS_URL, data=data, headers=headers,
                              timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                err = await r.text()
                print(f"[TTS] Yandex HTTP {r.status}: {err[:200]}", flush=True)
                return None
            pcm = await r.read()

        if not pcm:
            return None

        # Yandex отдаёт raw LPCM int16 mono без WAV-заголовка — оборачиваем
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
            wf.writeframes(pcm)
        return buf.getvalue()
    except Exception as e:
        print(f"[TTS] Yandex error: {e!r}", flush=True)
        return None


async def synthesize_to_file(text: str, out_path: Path,
                             voice: str = DEFAULT_VOICE,
                             emotion: str | None = None,
                             rate: float = 1.0) -> bool:
    data = await synthesize(text, voice=voice, emotion=emotion, rate=rate)
    if data is None:
        return False
    out_path.write_bytes(data)
    return True
