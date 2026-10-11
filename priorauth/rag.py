"""
Policy retrieval (RAG). Deterministic hashed n-gram embeddings by default so tests
and CI need no model download or API key; swap `Embedder` for Bedrock Titan in AWS.

Stores:
  * InMemoryVectorStore  -- cosine over numpy (tests / eval)
  * PgVectorStore        -- Postgres + pgvector (HNSW, cosine) for compose/AWS
Retrieval is *code-first*: an exact service-code match outranks any semantic score,
because a coverage lookup must not be fooled by a nearby-sounding policy.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from priorauth.schemas import Policy
from priorauth.synth.policies import POLICIES

DIM = 512
_TOK = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "with", "is", "are", "be", "after", "at", "on"}


class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    dim = DIM

    def embed(self, text: str) -> np.ndarray:
        toks = [t for t in _TOK.findall(text.lower()) if t not in STOP]
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        v = np.zeros(DIM, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % DIM] += 1.0 if (h >> 63) & 1 else -1.0   # signed feature hashing
        n = np.linalg.norm(v)
        return v / n if n else v


class BedrockEmbedder:  # pragma: no cover - requires AWS credentials; not exercised in CI
    dim = 1024

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", region: str = "us-east-1"):
        import boto3
        self._c = boto3.client("bedrock-runtime", region_name=region)
        self._m = model_id

    def embed(self, text: str) -> np.ndarray:
        import json
        r = self._c.invoke_model(modelId=self._m, body=json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True}))
        return np.array(json.loads(r["body"].read())["embedding"], dtype=np.float32)


@dataclass
class Chunk:
    chunk_id: str
    policy_id: str
    text: str


def chunk_policies(policies: list[Policy] = POLICIES) -> list[Chunk]:
    out = []
    for p in policies:
        out.append(Chunk(f"{p.policy_id}#summary", p.policy_id, f"{p.title}. {p.summary}"))
        for c in p.criteria:
            out.append(Chunk(f"{p.policy_id}#{c.id}", p.policy_id, f"{p.title}. {c.text}"))
    return out


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.emb = embedder or HashEmbedder()
        self._chunks: list[Chunk] = []
        self._mat = np.zeros((0, self.emb.dim), dtype=np.float32)

    def upsert(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._mat = np.stack([self.emb.embed(c.text) for c in chunks]) if chunks else self._mat

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.emb.embed(query)
        sims = self._mat @ q
        idx = np.argsort(-sims)[:k]
        return [(self._chunks[i], float(sims[i])) for i in idx]


class PgVectorStore:  # pragma: no cover - needs Postgres+pgvector; see tests/priorauth/test_pgvector.py
    def __init__(self, dsn: str, embedder: Embedder | None = None):
        import sqlalchemy as sa
        self.sa, self.emb = sa, embedder or HashEmbedder()
        self.engine = sa.create_engine(dsn, future=True)
        with self.engine.begin() as c:
            c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            c.execute(sa.text(f"CREATE TABLE IF NOT EXISTS policy_chunks (chunk_id text primary key, policy_id text, "
                              f"text text, embedding vector({self.emb.dim}))"))
            c.execute(sa.text("CREATE INDEX IF NOT EXISTS policy_chunks_hnsw ON policy_chunks "
                              "USING hnsw (embedding vector_cosine_ops)"))

    def upsert(self, chunks: list[Chunk]) -> None:
        sa = self.sa
        with self.engine.begin() as c:
            for ch in chunks:
                c.execute(sa.text("INSERT INTO policy_chunks VALUES (:i,:p,:t,:e) ON CONFLICT (chunk_id) DO UPDATE "
                                  "SET text=:t, embedding=:e"),
                          {"i": ch.chunk_id, "p": ch.policy_id, "t": ch.text, "e": str(self.emb.embed(ch.text).tolist())})

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        sa = self.sa
        with self.engine.connect() as c:
            rows = c.execute(sa.text("SELECT chunk_id, policy_id, text, 1 - (embedding <=> :q) AS sim FROM policy_chunks "
                                     "ORDER BY embedding <=> :q LIMIT :k"), {"q": str(self.emb.embed(query).tolist()), "k": k})
            return [(Chunk(r.chunk_id, r.policy_id, r.text), float(r.sim)) for r in rows]


class PolicyRetriever:
    def __init__(self, store=None, policies: list[Policy] = POLICIES):
        self.policies = {p.policy_id: p for p in policies}
        self.by_code = {c: p for p in policies for c in p.service_codes}
        self.store = store or InMemoryVectorStore()
        self.store.upsert(chunk_policies(policies))

    def retrieve(self, query: str, service_code: str | None = None, k: int = 3) -> list[dict]:
        """Ranked policies: exact code match first (score 1.0, via='code'), then semantic (max chunk sim per policy)."""
        ranked: list[dict] = []
        if service_code and service_code in self.by_code:
            ranked.append({"policy_id": self.by_code[service_code].policy_id, "score": 1.0, "via": "code"})
        best: dict[str, float] = {}
        for ch, s in self.store.search(query, k=max(k * 4, 10)):
            best[ch.policy_id] = max(best.get(ch.policy_id, -1), s)
        for pid, s in sorted(best.items(), key=lambda kv: -kv[1]):
            if not any(r["policy_id"] == pid for r in ranked):
                ranked.append({"policy_id": pid, "score": round(min(s, 0.99), 4), "via": "semantic"})
        return ranked[:k]
