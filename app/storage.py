import json
import os
import pickle
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import get_settings


@dataclass
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    source: str
    metadata: Dict
    tokens: List[str]
    embedding: np.ndarray
    created_at: float = 0.0


class Storage:
    def __init__(self) -> None:
        settings = get_settings()
        os.makedirs(settings.data_dir, exist_ok=True)
        self.db_path = os.path.join(settings.data_dir, "index.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source TEXT,
                metadata TEXT,
                created_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                text TEXT,
                source TEXT,
                metadata TEXT,
                tokens TEXT,
                embedding BLOB,
                created_at REAL
            )
            """
        )
        self.conn.commit()

    def upsert_document(self, document_id: str, source: str, metadata: Dict) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO documents (id, source, metadata, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, source, json.dumps(metadata or {}), time.time()),
        )
        self.conn.commit()

    def add_chunks(self, document_id: str, chunks: List[Chunk]) -> None:
        cur = self.conn.cursor()
        rows = []
        for chunk in chunks:
            rows.append(
                (
                    chunk.id or str(uuid.uuid4()),
                    document_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.source,
                    json.dumps(chunk.metadata or {}),
                    json.dumps(chunk.tokens or []),
                    sqlite3.Binary(pickle.dumps(chunk.embedding.astype(np.float32))),
                    time.time(),
                )
            )
        cur.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (id, document_id, chunk_index, text, source, metadata, tokens, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def load_chunks(self, limit: Optional[int] = None) -> List[Chunk]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, document_id, chunk_index, text, source, metadata, tokens, embedding, created_at FROM chunks ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
        result: List[Chunk] = []
        for row in rows[:limit] if limit else rows:
            embedding = pickle.loads(row[7])
            result.append(
                Chunk(
                    id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    text=row[3],
                    source=row[4],
                    metadata=json.loads(row[5] or "{}"),
                    tokens=json.loads(row[6] or "[]"),
                    embedding=embedding,
                    created_at=row[8] or 0.0,
                )
            )
        return result

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, document_id, chunk_index, text, source, metadata, tokens, embedding, created_at
            FROM chunks WHERE id = ?
            """,
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return Chunk(
            id=row[0],
            document_id=row[1],
            chunk_index=row[2],
            text=row[3],
            source=row[4],
            metadata=json.loads(row[5] or "{}"),
            tokens=json.loads(row[6] or "[]"),
            embedding=pickle.loads(row[7]),
            created_at=row[8] or 0.0,
        )

    def list_documents(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, source, metadata, created_at FROM documents")
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "source": row[1],
                "metadata": json.loads(row[2] or "{}"),
                "created_at": row[3],
            }
            for row in rows
        ]
