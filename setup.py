"""
Установочный скрипт для voice-ai-bot.

  python setup.py

Делает:
  1) Проверяет Python ≥ 3.10
  2) Устанавливает зависимости из requirements.txt + voice_recv с git
  3) Скачивает Silero VAD ONNX модель в models/ (если её нет)
  4) Создаёт .env из .env.example (если .env не существует)
  5) Печатает следующие шаги
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).parent
PYTHON = sys.executable
SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


def step(msg: str) -> None:
    print(f"\n\033[1;36m▶ {msg}\033[0m", flush=True)


def err(msg: str) -> None:
    print(f"\033[1;31m✗ {msg}\033[0m", flush=True)


def ok(msg: str) -> None:
    print(f"\033[1;32m✓ {msg}\033[0m", flush=True)


def run(cmd: list[str], check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd) if not check else subprocess.check_call(cmd)


def main() -> int:
    # 1) Python version
    step("Проверка Python")
    if sys.version_info < (3, 10):
        err(f"Нужен Python 3.10+, сейчас {sys.version}")
        return 1
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # 2) pip install
    step("Установка зависимостей (requirements.txt)")
    try:
        run([PYTHON, "-m", "pip", "install", "--upgrade", "pip"])
        run([PYTHON, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    except subprocess.CalledProcessError as e:
        err(f"pip install упал: {e}")
        return 1

    step("Установка discord-ext-voice-recv с git (для DAVE-поддержки)")
    try:
        run([
            PYTHON, "-m", "pip", "install", "--upgrade", "--force-reinstall",
            "git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git",
        ])
    except subprocess.CalledProcessError as e:
        err(f"voice_recv не поставился: {e}")
        return 1
    ok("Зависимости установлены")

    # 3) Silero VAD model
    step("Скачивание Silero VAD ONNX модели")
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    vad_path = models_dir / "silero_vad.onnx"
    if vad_path.exists():
        ok(f"Уже есть: {vad_path} ({vad_path.stat().st_size // 1024} KB)")
    else:
        try:
            urllib.request.urlretrieve(SILERO_VAD_URL, vad_path)
            ok(f"Скачано: {vad_path}")
        except Exception as e:
            err(f"Не удалось скачать VAD: {e}")
            return 1

    # 4) .env
    step("Настройка .env")
    env_path = ROOT / ".env"
    env_example = ROOT / ".env.example"
    if env_path.exists():
        ok(f".env уже существует — не трогаю")
    else:
        if env_example.exists():
            shutil.copy(env_example, env_path)
            ok(f".env создан из .env.example — заполни ключи!")
        else:
            err("Нет .env.example")
            return 1

    # 5) Done
    print()
    ok("УСТАНОВКА ЗАВЕРШЕНА")
    print()
    print("Дальше:")
    print("  1) Открой .env и заполни DISCORD_TOKEN, ANTHROPIC_API_KEY, YANDEX_API_KEY")
    print("  2) (Опционально) AUTO_JOIN_CHANNEL — ID голосового канала для авто-захода")
    print("  3) Запусти бот:")
    print("       Windows:  run_bot.bat   (или python bot.py)")
    print("       Linux:    python bot.py")
    print()
    print("Подробнее в README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
