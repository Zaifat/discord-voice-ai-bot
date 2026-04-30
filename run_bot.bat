@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d %~dp0
echo === Voice AI Bot — full pipeline ===
python -u bot.py
pause
