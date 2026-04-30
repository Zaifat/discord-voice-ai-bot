"""Глобальный конфиг проекта."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# ─── Discord ──────────────────────────────────────────────────────────────────
DISCORD_TOKEN     = os.getenv("DISCORD_TOKEN", "").strip()
AUTO_JOIN_CHANNEL = int(os.getenv("AUTO_JOIN_CHANNEL", "0") or 0)
TEXT_CHANNEL      = int(os.getenv("TEXT_CHANNEL", "0") or 0)

# ─── Audio (Discord raw output) ───────────────────────────────────────────────
SAMPLE_RATE = 48000   # Opus → PCM 48kHz
CHANNELS    = 2       # стерео
SAMPLE_WIDTH = 2      # int16

# ─── Папки ────────────────────────────────────────────────────────────────────
DEBUG_DIR  = ROOT / "debug"
MODELS_DIR = ROOT / "models"
DEBUG_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ─── Stage 1: capture diagnostic ──────────────────────────────────────────────
CAPTURE_SECONDS = 8   # сколько секунд писать в первом тесте
