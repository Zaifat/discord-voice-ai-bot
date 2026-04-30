"""
Whisper STT через faster-whisper (CTranslate2 backend).
Модель large-v3 в int8 на CPU — баланс качества/скорости.

Использование:
    from audio.stt import transcribe_pcm16, warmup
    await warmup()  # один раз при старте, чтобы прогреть модель
    text = await transcribe_pcm16(samples_16k_mono_float32, language="ru")
"""
from __future__ import annotations
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import MODELS_DIR


# ─── Windows: добавляем pip-CUDA DLL'ки в DLL search path ─────────────────────
# CTranslate2 на CUDA требует cublas64_12.dll, cudnn*.dll. Без полного
# CUDA Toolkit берём их из nvidia-cublas-cu12 / nvidia-cudnn-cu12 (pip).
if sys.platform == "win32":
    try:
        import nvidia
        _nv_root = Path(nvidia.__path__[0])
        _added = []
        for _sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin"):
            _p = _nv_root / _sub
            if _p.exists():
                os.add_dll_directory(str(_p))
                _added.append(str(_p))
        # CTranslate2 (C++) грузит DLL через LoadLibrary, который смотрит PATH —
        # поэтому ещё и в PATH прописываем
        if _added:
            os.environ["PATH"] = os.pathsep.join(_added) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# large-v3-turbo (2024) — distil-стиль модель, в 6x быстрее large-v3,
# но multilingual (включая русский). На GPU (~1.5GB VRAM в float16) RTF ~0.05.
_MODEL_NAME = "large-v3-turbo"
_COMPUTE_TYPE = "float16"
_DEVICE = "cuda"

_model = None  # lazy-loaded

# Типичные whisper-галлюцинации на тишине / плохом аудио
_HALLUCINATIONS = {
    "субтитры", "субтитры подогнал", "субтитры сделал dimatorzok",
    "продолжение следует", "продолжение следует...",
    "спасибо за просмотр", "спасибо за внимание",
    "до встречи в следующих видео", "до встречи в следующих сериях",
    "thanks for watching", "thank you for watching",
    "ставьте лайки", "подписывайтесь на канал",
    "[музыка]", "[music]", "[silence]", "[тишина]",
    "music", "музыка",
    "переведено и озвучено", "редактор",
    "...",
}

_CREDIT_KEYWORDS = (
    "субтитр", "dimatorzok", "перевёл", "перевод выполн",
    "озвучил", "озвучка", "корректор", "редактор",
)


def _load_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel
    print(f"[STT] Загружаю Whisper {_MODEL_NAME} ({_COMPUTE_TYPE} на {_DEVICE})...", flush=True)
    t0 = time.monotonic()
    # Кэшируем модели в models/whisper, а не в HF cache
    cache_dir = MODELS_DIR / "whisper"
    cache_dir.mkdir(exist_ok=True)
    _model = WhisperModel(
        _MODEL_NAME,
        device=_DEVICE,
        compute_type=_COMPUTE_TYPE,
        download_root=str(cache_dir),
        cpu_threads=4,
        num_workers=1,
    )
    print(f"[STT] Whisper готов ({time.monotonic()-t0:.1f}с)", flush=True)
    return _model


async def warmup() -> None:
    """Прогревает модель — первый transcribe иначе занимает 5-15с."""
    def _do():
        m = _load_model()
        # Прогон 1 секунды тишины — JIT-компиляция CTranslate2
        silence = np.zeros(16000, dtype=np.float32)
        list(m.transcribe(silence, language="ru", beam_size=1)[0])
    await asyncio.to_thread(_do)


def _post_filter(text: str) -> Optional[str]:
    """Отсев галлюцинаций и titres-style мусора."""
    if not text:
        return None
    t = text.strip()
    if len(t) < 2:
        return None
    tl = t.lower().strip(".,!?…«»\"' ")

    # Точные галлюцинации
    if tl in _HALLUCINATIONS:
        return None

    # Кредиты/титры
    if any(k in tl for k in _CREDIT_KEYWORDS):
        return None

    # Зацикленные повторы слов (например: "ворота ворота ворота")
    words = t.split()
    if len(words) >= 4:
        unique = len(set(w.strip(".,!?") for w in words))
        if unique <= 2:
            return None

    # Whisper любит повторять одну и ту же подстроку — обрежем до первого повтора.
    # Делим на предложения по точкам/восклицаниям и смотрим повторы.
    import re
    sentences = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
    if len(sentences) >= 2:
        seen = set()
        kept = []
        for s in sentences:
            key = s.lower().strip()
            if key in seen:
                # Повторение — обрезаем здесь
                break
            seen.add(key)
            kept.append(s)
        if len(kept) < len(sentences):
            t = ". ".join(kept) + "."

    # Если осталась всего одна короткая фраза, повторённая 3+ раз ("Алло, Алло, Алло")
    if "," in t:
        parts = [p.strip() for p in t.split(",") if p.strip()]
        if len(parts) >= 3:
            unique_parts = len(set(p.lower() for p in parts))
            if unique_parts == 1:
                t = parts[0]

    return t.strip() if t.strip() else None


async def transcribe_pcm16(samples: np.ndarray, language: str = "ru",
                           initial_prompt: str | None = None) -> Optional[str]:
    """
    samples: float32 mono @ 16kHz, диапазон [-1, 1].
    Возвращает текст или None если речи не распознано / отфильтровано.
    """
    assert samples.dtype == np.float32, f"need float32, got {samples.dtype}"
    if len(samples) < 16000 * 0.3:  # < 300мс
        return None

    def _do() -> Optional[str]:
        m = _load_model()
        segments, info = m.transcribe(
            samples,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=initial_prompt or "Разговор в Discord на русском языке. В чате есть бот VoiceAI.",
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,    # отсев повторяющихся пр. фраз
            log_prob_threshold=-1.0,            # отсев низко-вероятностных
            hallucination_silence_threshold=2.0,  # игнор подозрительной "тишины-речи"
        )
        parts = [s.text for s in segments]
        return "".join(parts).strip() if parts else None

    text = await asyncio.to_thread(_do)
    return _post_filter(text) if text else None
