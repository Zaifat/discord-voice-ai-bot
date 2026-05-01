"""
Двухуровневая память:

1. ConversationHistory — короткая RAM-память последних N реплик per guild.
   Используется для context window LLM.
   При создании может загружать последние N реплик из JSONL-лога — диалог
   продолжается после перезапуска бота.

2. LongTermMemory — Chroma vector DB + bge-m3 embeddings через Ollama.
   Постоянная (на диске), для семантического поиска.

3. PersistentChatLog — append-only JSONL файл всех реплик. Голый журнал
   для отладки, ручного просмотра, или восстановления истории.
"""
from __future__ import annotations
import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import ollama

from config import ROOT


# ─── Embeddings via Ollama ────────────────────────────────────────────────────
EMBED_MODEL = "bge-m3"
_embed_client: ollama.AsyncClient | None = None


def _get_embed_client() -> ollama.AsyncClient:
    global _embed_client
    if _embed_client is None:
        _embed_client = ollama.AsyncClient()
    return _embed_client


async def embed(text: str) -> list[float]:
    """Получить embedding текста через Ollama (bge-m3, 1024-dim)."""
    cl = _get_embed_client()
    resp = await cl.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    return await asyncio.gather(*(embed(t) for t in texts))


# ─── Persistent JSONL chat log ────────────────────────────────────────────────
class PersistentChatLog:
    """
    Append-only JSONL файл всех реплик per guild — chat_logs/<guild_id>.jsonl
    Каждая строка: {"ts": float, "role": "user|assistant", "name": "...", "text": "..."}
    """
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (ROOT / "chat_logs")
        self.base_dir.mkdir(exist_ok=True)

    def _path(self, guild_id: int) -> Path:
        return self.base_dir / f"{guild_id}.jsonl"

    def append(self, guild_id: int, role: str, text: str, name: str = "") -> None:
        rec = {"ts": time.time(), "role": role, "name": name, "text": text}
        try:
            with self._path(guild_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ChatLog] write error: {e!r}", flush=True)

    def read_last(self, guild_id: int, n: int) -> list[dict]:
        """Прочитать последние n записей."""
        p = self._path(guild_id)
        if not p.exists():
            return []
        try:
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            tail = lines[-n:] if n > 0 else lines
            return [json.loads(l) for l in tail if l.strip()]
        except Exception as e:
            print(f"[ChatLog] read error: {e!r}", flush=True)
            return []

    def count(self, guild_id: int) -> int:
        p = self._path(guild_id)
        if not p.exists():
            return 0
        try:
            with p.open("rb") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def clear(self, guild_id: int) -> None:
        p = self._path(guild_id)
        if p.exists():
            p.unlink()


# Singleton — общий лог между всеми ConversationHistory
_shared_chatlog: PersistentChatLog | None = None


def _get_chatlog() -> PersistentChatLog:
    global _shared_chatlog
    if _shared_chatlog is None:
        _shared_chatlog = PersistentChatLog()
    return _shared_chatlog


# ─── Краткосрочная память: deque + persistent log ─────────────────────────────
class ConversationHistory:
    """
    Хранит последние N реплик одного guild в RAM (для context window LLM).
    Параллельно дублирует ВСЕ реплики в persistent JSONL-лог.

    При создании с guild_id ≠ 0 — загружает последние max_messages записей
    из JSONL чтобы продолжить диалог после перезапуска.
    """
    def __init__(self, max_messages: int = 30, guild_id: int = 0):
        self._messages: deque[dict] = deque(maxlen=max_messages)
        self._guild_id = guild_id
        self._last_speaker = ""

        if guild_id:
            # Загружаем "хвост" истории с диска
            self._restore_from_log()

    def _restore_from_log(self) -> None:
        log = _get_chatlog()
        recs = log.read_last(self._guild_id, self._messages.maxlen or 30)
        for rec in recs:
            role = rec.get("role")
            text = rec.get("text", "")
            if role in ("user", "assistant") and text:
                self._messages.append({"role": role, "content": text})
        if recs:
            print(f"[History] Восстановлено {len(recs)} реплик из chat_logs/{self._guild_id}.jsonl",
                  flush=True)
            self._last_speaker = next(
                (r.get("name", "") for r in reversed(recs) if r.get("role") == "user"), ""
            )

    def add_user(self, speaker_name: str, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._last_speaker = speaker_name
        if self._guild_id:
            _get_chatlog().append(self._guild_id, "user", text, name=speaker_name)

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        if self._guild_id:
            _get_chatlog().append(self._guild_id, "assistant", text, name="bot")

    @property
    def last_speaker(self) -> str:
        return self._last_speaker

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        """Очищает RAM-историю. На JSONL не влияет."""
        self._messages.clear()

    def clear_persistent(self) -> None:
        """Также удаляет JSONL-лог (полная очистка)."""
        self.clear()
        if self._guild_id:
            _get_chatlog().clear(self._guild_id)

    def __len__(self) -> int:
        return len(self._messages)


# ─── Долгосрочная память: Chroma + bge-m3 ─────────────────────────────────────
class LongTermMemory:
    """
    Vector store воспоминаний (per guild). При каждой новой реплике делаем
    top-k поиск релевантных воспоминаний → добавляются в system prompt.

    Полностью persistent: данные хранятся на диске в memory_db/.
    """
    def __init__(self, persist_dir: Path | None = None):
        import chromadb
        self.persist_dir = persist_dir or (ROOT / "memory_db")
        self.persist_dir.mkdir(exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collections: dict[str, "chromadb.Collection"] = {}

    def _get_coll(self, guild_id: int):
        key = f"guild_{guild_id}"
        if key not in self._collections:
            self._collections[key] = self._client.get_or_create_collection(
                name=key,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[key]

    async def add(self, guild_id: int, text: str,
                  speaker: str | None = None,
                  metadata: dict | None = None) -> None:
        if not text or not text.strip():
            return
        coll = self._get_coll(guild_id)
        try:
            emb = await embed(text)
        except Exception as e:
            print(f"[LTM] embed error при add: {e!r}", flush=True)
            return
        ts = time.time()
        meta = {"speaker": speaker or "", "ts": ts}
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        doc_id = f"{int(ts*1000)}_{hash(text) & 0xffff}"
        await asyncio.to_thread(
            coll.add,
            ids=[doc_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[meta],
        )

    async def search(self, guild_id: int, query: str, top_k: int = 3,
                     max_distance: float = 0.6) -> list[tuple[str, dict, float]]:
        """Top-k релевантных воспоминаний. Фильтрует distance > max_distance."""
        if not query or not query.strip():
            return []
        coll = self._get_coll(guild_id)
        if coll.count() == 0:
            return []
        try:
            emb = await embed(query)
        except Exception as e:
            print(f"[LTM] embed error при search: {e!r}", flush=True)
            return []
        result = await asyncio.to_thread(
            coll.query,
            query_embeddings=[emb],
            n_results=min(top_k, coll.count()),
        )
        out = []
        for doc, meta, dist in zip(result["documents"][0],
                                    result["metadatas"][0],
                                    result["distances"][0]):
            if dist <= max_distance:
                out.append((doc, meta, dist))
        return out

    def stats(self, guild_id: int) -> dict:
        coll = self._get_coll(guild_id)
        return {"total_memories": coll.count()}

    def clear(self, guild_id: int) -> None:
        key = f"guild_{guild_id}"
        try:
            self._client.delete_collection(key)
        except Exception:
            pass
        self._collections.pop(key, None)
