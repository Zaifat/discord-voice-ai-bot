"""
LLM клиент через Anthropic API (Claude Sonnet 4.5).

Модель умнее локальных Qwen/Gemma в любом разрезе, отличный русский,
без safety-блокировок на разговорных темах.
"""
from __future__ import annotations
import asyncio
import os
import time
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from config import ROOT
# override=True: значения из .env перебивают систему (у Windows может быть
# глобальная ANTHROPIC_API_KEY, тут хотим именно проектный ключ)
load_dotenv(ROOT / ".env", override=True)

import anthropic


LLM_MODEL = "claude-haiku-4-5"  # быстрая, ~500-800мс на короткий ответ

DEFAULT_OPTIONS = {
    "max_tokens": 100,        # коротко для голоса
    "temperature": 0.8,
}

_client: Optional[anthropic.AsyncAnthropic] = None


def _get_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        key = _get_api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY не задан в .env")
        _client = anthropic.AsyncAnthropic(
            api_key=key,
            timeout=15.0,         # не больше 15с на запрос
            max_retries=1,
        )
    return _client


async def warmup() -> None:
    print(f"[LLM] Прогрев {LLM_MODEL}...", flush=True)
    if not _get_api_key():
        print(f"[LLM] ⚠ ANTHROPIC_API_KEY не задан в .env", flush=True)
        return
    t0 = time.monotonic()
    try:
        cl = _get_client()
        resp = await cl.messages.create(
            model=LLM_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": "Скажи: ок"}],
        )
        text = resp.content[0].text if resp.content else ""
        print(f"[LLM] Готов ({time.monotonic()-t0:.1f}с): {text[:50]!r}", flush=True)
    except Exception as e:
        print(f"[LLM] Ошибка прогрева: {e!r}", flush=True)


async def keep_loaded() -> None:
    """Anthropic API всегда теплый, но оставим для совместимости интерфейса."""
    pass


def _split_system(messages: list[dict], system: str | None) -> tuple[str | None, list[dict]]:
    """
    Anthropic API ожидает messages[role/content], system отдельно.
    Также первое сообщение должно быть user. Перекладываем формат.
    """
    out = []
    for m in messages:
        role = m.get("role")
        if role in ("user", "assistant"):
            out.append({"role": role, "content": m.get("content", "")})
    return system, out


async def generate_reply(messages: list[dict], system: str | None = None,
                         options: dict | None = None) -> str:
    sys_prompt, msgs = _split_system(messages, system)
    if not msgs:
        return ""
    cl = _get_client()
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    resp = await cl.messages.create(
        model=LLM_MODEL,
        system=sys_prompt or "",
        messages=msgs,
        **opts,
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return text.strip()


async def generate_reply_stream(messages: list[dict], system: str | None = None,
                                options: dict | None = None) -> AsyncGenerator[str, None]:
    sys_prompt, msgs = _split_system(messages, system)
    if not msgs:
        return
    cl = _get_client()
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    async with cl.messages.stream(
        model=LLM_MODEL,
        system=sys_prompt or "",
        messages=msgs,
        **opts,
    ) as stream:
        async for tok in stream.text_stream:
            yield tok
