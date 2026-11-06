from __future__ import annotations

from array import array
from dataclasses import dataclass
from hashlib import md5
import sqlite3
from pathlib import Path
from time import time
from typing import Any


def parse_vector_from_fields(vector_from: str) -> list[str]:
    out: list[str] = []
    for p in (vector_from or "").split(","):
        x = p.strip()
        if x:
            out.append(x)
    return out


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalized_vector_text(doc: dict[str, Any], fields: list[str]) -> str:
    vals: list[str] = []
    for f in fields:
        raw = doc.get(f, "")
        vals.append(normalize_text(str(raw or "")))
    return " ".join(vals).strip()


def cache_key_from_text(text: str, model: str, similarity: str) -> str:
    payload = f"{text}\n{model}\n{similarity}"
    return md5(payload.encode("utf-8")).hexdigest()


def cache_key_from_doc(doc: dict[str, Any], fields: list[str], model: str, similarity: str) -> str:
    return cache_key_from_text(normalized_vector_text(doc, fields), model=model, similarity=similarity)


@dataclass
class CacheEntry:
    status: str  # ok | null | missing
    vector: list[float] | None = None


class VectorCache:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                similarity TEXT NOT NULL,
                text_value TEXT NOT NULL,
                status TEXT NOT NULL,
                vector_json TEXT,
                vector_blob BLOB,
                updated_at INTEGER NOT NULL
            )
            """
        )
        cols = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(embeddings_cache)").fetchall()}
        if "vector_blob" not in cols:
            self.conn.execute("ALTER TABLE embeddings_cache ADD COLUMN vector_blob BLOB")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_cache_model ON embeddings_cache(model)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_cache_status ON embeddings_cache(status)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def pack_vector(vector: list[float]) -> bytes:
        arr = array("f", (float(x) for x in vector))
        return arr.tobytes()

    @staticmethod
    def unpack_vector(blob: bytes | bytearray | memoryview | None) -> list[float]:
        if blob:
            arr = array("f")
            arr.frombytes(bytes(blob))
            return [float(x) for x in arr]
        return []

    def get(self, cache_key: str) -> CacheEntry:
        row = self.conn.execute(
            "SELECT status, vector_blob FROM embeddings_cache WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        if not row:
            return CacheEntry(status="missing", vector=None)
        status = str(row[0] or "").strip().lower()
        if status == "ok":
            vec = self.unpack_vector(row[1])
            return CacheEntry(status="ok", vector=vec)
        return CacheEntry(status="null", vector=None)

    def get_status_map(self, *, model: str, similarity: str) -> dict[str, str]:
        rows = self.conn.execute(
            """
            SELECT cache_key, status
            FROM embeddings_cache
            WHERE model=? AND similarity=?
            """,
            (model, similarity),
        ).fetchall()
        return {str(k): str(s or "").strip().lower() for k, s in rows}

    def get_status_many(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        out: dict[str, str] = {}
        chunk_size = 500
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT cache_key, status
                FROM embeddings_cache
                WHERE cache_key IN ({placeholders})
                """,
                tuple(chunk),
            ).fetchall()
            for k, s in rows:
                out[str(k)] = str(s or "").strip().lower()
        return out

    def get_vectors_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        out: dict[str, list[float]] = {}
        chunk_size = 500
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT cache_key, status, vector_blob
                FROM embeddings_cache
                WHERE cache_key IN ({placeholders})
                """,
                tuple(chunk),
            ).fetchall()
            for k, status, vb in rows:
                if str(status or "").strip().lower() != "ok":
                    continue
                out[str(k)] = self.unpack_vector(vb)
        return out

    def upsert_ok(self, *, cache_key: str, model: str, similarity: str, text_value: str, vector: list[float]) -> None:
        self.upsert_ok_many(
            [
                {
                    "cache_key": cache_key,
                    "model": model,
                    "similarity": similarity,
                    "text_value": text_value,
                    "vector": vector,
                }
            ]
        )

    def upsert_ok_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        now = int(time())
        values = [
            (
                str(row["cache_key"]),
                str(row["model"]),
                str(row["similarity"]),
                str(row["text_value"]),
                "ok",
                None,
                sqlite3.Binary(self.pack_vector(row.get("vector", []))),
                now,
            )
            for row in rows
        ]
        self.conn.execute(
            "BEGIN",
        )
        try:
            self.conn.executemany(
                """
                INSERT INTO embeddings_cache(cache_key, model, similarity, text_value, status, vector_json, vector_blob, updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model=excluded.model,
                    similarity=excluded.similarity,
                    text_value=excluded.text_value,
                    status='ok',
                    vector_json=excluded.vector_json,
                    vector_blob=excluded.vector_blob,
                    updated_at=excluded.updated_at
                WHERE embeddings_cache.status!='ok'
                   OR embeddings_cache.vector_json IS NOT excluded.vector_json
                   OR embeddings_cache.vector_blob IS NOT excluded.vector_blob
                   OR embeddings_cache.model IS NOT excluded.model
                   OR embeddings_cache.similarity IS NOT excluded.similarity
                   OR embeddings_cache.text_value IS NOT excluded.text_value
                """,
                values,
            )
        finally:
            self.conn.execute("COMMIT")

    def upsert_null(self, *, cache_key: str, model: str, similarity: str, text_value: str) -> None:
        self.upsert_null_many(
            [
                {
                    "cache_key": cache_key,
                    "model": model,
                    "similarity": similarity,
                    "text_value": text_value,
                }
            ]
        )

    def upsert_null_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        now = int(time())
        values = [
            (
                str(row["cache_key"]),
                str(row["model"]),
                str(row["similarity"]),
                str(row["text_value"]),
                "null",
                None,
                now,
            )
            for row in rows
        ]
        self.conn.execute("BEGIN")
        try:
            self.conn.executemany(
                """
                INSERT INTO embeddings_cache(cache_key, model, similarity, text_value, status, vector_json, updated_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO NOTHING
                """,
                values,
            )
        finally:
            self.conn.execute("COMMIT")

    def commit(self) -> None:
        self.conn.commit()

    def delete_many(self, keys: list[str]) -> int:
        if not keys:
            return 0
        deleted = 0
        chunk_size = 500
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cur = self.conn.execute(
                f"DELETE FROM embeddings_cache WHERE cache_key IN ({placeholders})",
                tuple(chunk),
            )
            deleted += int(cur.rowcount or 0)
        self.conn.commit()
        return deleted
