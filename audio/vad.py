"""
Silero VAD v5 через onnxruntime — без зависимости от PyTorch.

Принимает 16kHz mono float32 PCM по 32мс кадрам (512 семплов).
Возвращает probability речи в этом кадре (0..1).

Над модели: streaming-aware (хранит hidden state между кадрами).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import onnxruntime as ort

from config import MODELS_DIR


_SR = 16000
_FRAME_SAMPLES = 512   # 32мс при 16kHz
_CONTEXT_SAMPLES = 64  # Silero v5 требует 64 семпла context от предыдущего чанка

# Глобальная сессия — переиспользуется
_session: ort.InferenceSession | None = None


def _load_session() -> ort.InferenceSession:
    global _session
    if _session is not None:
        return _session
    model_path = MODELS_DIR / "silero_vad.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Silero VAD модель не найдена: {model_path}")
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    _session = ort.InferenceSession(str(model_path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
    return _session


class SileroVAD:
    """
    Streaming VAD: подавай 32мс кадры подряд через .probability(frame).
    Silero v5 требует context-padding: к каждому кадру слева подклеиваются
    последние 64 семпла предыдущего кадра (для нулевого — нули).
    """

    def __init__(self):
        self.session = _load_session()
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)
        self._sr = np.array(_SR, dtype=np.int64)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Один 32мс кадр (512 float32 @ 16kHz) → probability речи."""
        assert frame.dtype == np.float32 and frame.shape == (_FRAME_SAMPLES,), \
            f"frame must be float32 [{_FRAME_SAMPLES}], got {frame.dtype} {frame.shape}"
        # Склеиваем context (64) + frame (512) = 576 семплов
        x = np.concatenate([self._context[0], frame])[None, :]
        out, new_state = self.session.run(
            None,
            {"input": x.astype(np.float32), "state": self._state, "sr": self._sr},
        )
        self._state = new_state
        # Сохраняем последние 64 семпла для следующего вызова
        self._context = x[..., -_CONTEXT_SAMPLES:]
        return float(out[0, 0])


# ─── Stream segmenter: VAD-based phrase boundaries ────────────────────────────
class SpeechSegmenter:
    """
    Принимает float32 16kHz моно сэмплы любым размером,
    режет на 32мс кадры, прогоняет через VAD,
    собирает фрагменты речи между паузами.

    Параметры:
      threshold:        вероятность ≥ — речь
      min_speech_ms:    игнорировать речь короче этого (фильтр щелчков)
      min_silence_ms:   тишина дольше этого = конец фразы
      pre_pad_ms:       сколько добавить ДО начала речи (запас на атаку)
      post_pad_ms:      сколько добавить ПОСЛЕ конца речи (хвост)
    """

    def __init__(self,
                 threshold: float = 0.5,
                 min_speech_ms: int = 250,
                 min_silence_ms: int = 600,
                 pre_pad_ms: int = 200,
                 post_pad_ms: int = 200,
                 max_phrase_ms: int = 12000):
        self.vad = SileroVAD()
        self.threshold = threshold
        self.min_speech_frames  = max(1, min_speech_ms  // 32)
        self.min_silence_frames = max(1, min_silence_ms // 32)
        self.pre_pad_frames  = max(0, pre_pad_ms  // 32)
        self.post_pad_frames = max(0, post_pad_ms // 32)
        self.max_phrase_frames = max(10, max_phrase_ms // 32)

        self._frames: list[np.ndarray] = []     # все кадры, ring-style
        self._probs:  list[float]      = []     # параллельно с _frames
        self._partial: np.ndarray = np.zeros(0, dtype=np.float32)

        self._in_speech: bool = False
        self._speech_start_frame: int = 0
        self._silence_streak: int = 0
        self._speech_streak: int = 0

    def reset(self) -> None:
        self.vad.reset()
        self._frames.clear()
        self._probs.clear()
        self._partial = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._silence_streak = 0
        self._speech_streak = 0

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """
        Подаёт новые сэмплы. Возвращает список ЗАВЕРШЁННЫХ фраз
        (каждая — float32 16kHz моно). Незавершённую фразу не возвращает.
        """
        assert samples.dtype == np.float32, "feed expects float32"
        # Конкатенируем с partial buffer
        buf = np.concatenate([self._partial, samples]) if len(self._partial) else samples
        n_frames = len(buf) // _FRAME_SAMPLES
        leftover_start = n_frames * _FRAME_SAMPLES
        self._partial = buf[leftover_start:].copy()

        completed: list[np.ndarray] = []
        for i in range(n_frames):
            frame = buf[i * _FRAME_SAMPLES:(i + 1) * _FRAME_SAMPLES]
            p = self.vad.probability(frame)
            self._frames.append(frame)
            self._probs.append(p)
            idx = len(self._frames) - 1

            is_speech = p >= self.threshold
            if is_speech:
                self._silence_streak = 0
                self._speech_streak += 1
                if not self._in_speech and self._speech_streak >= self.min_speech_frames:
                    # Начало фразы — отступаем на pre_pad назад
                    self._in_speech = True
                    self._speech_start_frame = max(0, idx - self._speech_streak + 1 - self.pre_pad_frames)
            else:
                self._speech_streak = 0
                if self._in_speech:
                    self._silence_streak += 1

            # Hard timeout: если фраза тянется дольше max_phrase_frames — закрываем принудительно.
            # (Silero иногда "залипает" в speech state на тихом фоне.)
            phrase_len = idx - self._speech_start_frame + 1
            force_close = (self._in_speech and phrase_len >= self.max_phrase_frames)

            if self._in_speech and (self._silence_streak >= self.min_silence_frames or force_close):
                if force_close and self._silence_streak < self.min_silence_frames:
                    print(f"[VAD] Forced phrase close at {phrase_len*32}мс (Silero застрял)", flush=True)
                end_frame = min(len(self._frames), idx + 1)
                seg_frames = self._frames[self._speech_start_frame:end_frame]
                completed.append(np.concatenate(seg_frames))
                self._frames = self._frames[end_frame:]
                self._probs  = self._probs[end_frame:]
                self._in_speech = False
                self._silence_streak = 0
                self._speech_start_frame = 0
                # При forced close — сбрасываем internal state Silero, чтобы он не "запомнил" речь
                if force_close:
                    self.vad.reset()

        # Если речи нет уже долго — обрезаем буфер чтоб не копился
        if not self._in_speech and len(self._frames) > 50:
            keep = self.pre_pad_frames + 5
            self._frames = self._frames[-keep:]
            self._probs  = self._probs[-keep:]
            self._speech_start_frame = max(0, self._speech_start_frame - (len(self._frames) - keep))

        return completed

    @property
    def is_in_speech(self) -> bool:
        return self._in_speech

    def flush(self) -> np.ndarray | None:
        """Принудительно завершает текущий segment (например при выходе из канала)."""
        if not self._in_speech or self._speech_start_frame >= len(self._frames):
            return None
        seg = np.concatenate(self._frames[self._speech_start_frame:])
        self.reset()
        return seg
