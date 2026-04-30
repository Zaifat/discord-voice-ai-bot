"""
Двухуровневая память:
1. ConversationHistory — короткая RAM-история последних N реплик per guild
   (передаётся LLM в каждый запрос как messages list)
2. LongTermMemory — Chroma vector DB + bge-m3 embeddings через Ollama
   (поиск релевантных воспоминаний при каждой новой реплике)
"""
from __future__ import annotations
import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Iterable

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
    """Embeddings для нескольких текстов параллельно."""
    return await asyncio.gather(*(embed(t) for t in texts))


# ─── Краткосрочная память: deque ──────────────────────────────────────────────
class ConversationHistory:
    """
    Хранит последние N реплик одного guild.
    Используется для ContextWindow LLM.
    """
    def __init__(self, max_messages: int = 20):
        self._messages: deque[dict] = deque(maxlen=max_messages)

    def add_user(self, speaker_name: str, text: str) -> None:
        # Чистый content без "Имя:" префикса — иначе LLM думает что его зовут так же.
        # Текущий говорящий передаётся отдельно через system prompt в bot.py.
        self._messages.append({
            "role": "user",
            "content": text,
        })
        self._last_speaker = speaker_name

    @property
    def last_speaker(self) -> str:
        return getattr(self, "_last_speaker", "")

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# ─── Долгосрочная память: Chroma + bge-m3 ─────────────────────────────────────
class LongTermMemory:
    """
    Vector store воспоминаний (per guild). При каждой новой реплике пользователя
    делаем top-k поиск релевантных воспоминаний и добавляем их в system prompt.
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
            # без default ef — embeddings вычисляем сами через Ollama
            self._collections[key] = self._client.get_or_create_collection(
                name=key,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[key]

    async def add(self, guild_id: int, text: str,
                  speaker: str | None = None,
                  metadata: dict | None = None) -> None:
        """Сохранить новое воспоминание (например, ключевой факт о пользователе)."""
        if not text or not text.strip():
            return
        coll = self._get_coll(guild_id)
        emb = await embed(text)
        ts = time.time()
        meta = {"speaker": speaker or "", "ts": ts}
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        # ID — по timestamp + hash, чтобы не дублировалось
        doc_id = f"{int(ts*1000)}_{hash(text) & 0xffff}"
        await asyncio.to_thread(
            coll.add,
            ids=[doc_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[meta],
        )

    async def search(self, guild_id: int, query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
        """Top-k релевантных воспоминаний. Возвращает [(text, metadata, distance)]."""
        if not query or not query.strip():
            return []
        coll = self._get_coll(guild_id)
        if coll.count() == 0:
            return []
        emb = await embed(query)
        result = await asyncio.to_thread(
            coll.query,
            query_embeddings=[emb],
            n_results=min(top_k, coll.count()),
        )
        out = []
        for doc, meta, dist in zip(result["documents"][0],
                                    result["metadatas"][0],
                                    result["distances"][0]):
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
