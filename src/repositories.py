"""
Repository Pattern - Data Access Layer
"""
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.orm import Session

import json
from models import DocumentChunk, AchizitieDirecta, AnuntInitiere, ChatSession, ConversationMessage, EMBEDDING_DIM


class DocumentChunkRepository:
    """Repository pentru DocumentChunk (RAG)."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, chunk: DocumentChunk) -> DocumentChunk:
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def add_batch(self, chunks: list[DocumentChunk]) -> int:
        self.session.add_all(chunks)
        self.session.flush()
        return len(chunks)

    def get_by_id(self, chunk_id: int) -> DocumentChunk | None:
        return self.session.query(DocumentChunk).filter_by(id=chunk_id).first()

    def get_by_file(self, file_name: str) -> list[DocumentChunk]:
        return (
            self.session.query(DocumentChunk)
            .filter_by(file_name=file_name)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    def search_similar(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Caută chunks similare folosind cosine similarity.

        Returns:
            Lista de (chunk, score) ordonate descrescător după scor.
        """
        embedding_str = f"[{','.join(map(str, embedding))}]"

        # Folosim format string pentru vector cast (evită conflict cu :param)
        query = text(f"""
            SELECT id, 1 - (embedding <=> '{embedding_str}'::vector) as score
            FROM document_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT :top_k
        """)

        results = self.session.execute(
            query,
            {"top_k": top_k}
        ).fetchall()

        chunks_with_scores = []
        for row in results:
            if row.score >= threshold:
                chunk = self.get_by_id(row.id)
                if chunk:
                    chunks_with_scores.append((chunk, row.score))

        return chunks_with_scores

    def search_similar_filtered(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Vector search with optional metadata filters.
        Filters match on metadata_json jsonb fields (e.g. company_name, doc_type).
        """
        embedding_str = f"[{','.join(map(str, embedding))}]"

        meta_clauses = ""
        params = {"top_k": top_k}
        if metadata_filters:
            for key, value in metadata_filters.items():
                meta_clauses += f" AND LOWER(metadata_json::jsonb->>'{key}') = LOWER(:{key})"
                params[key] = value

        query = text(f"""
            SELECT id, 1 - (embedding <=> '{embedding_str}'::vector) as score
            FROM document_chunks
            WHERE embedding IS NOT NULL{meta_clauses}
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT :top_k
        """)

        results = self.session.execute(query, params).fetchall()

        chunks_with_scores = []
        for row in results:
            if row.score >= threshold:
                chunk = self.get_by_id(row.id)
                if chunk:
                    chunks_with_scores.append((chunk, row.score))

        return chunks_with_scores

    def count(self) -> int:
        return self.session.query(func.count(DocumentChunk.id)).scalar() or 0

    def count_by_file(self) -> dict[str, int]:
        results = (
            self.session.query(
                DocumentChunk.file_name,
                func.count(DocumentChunk.id)
            )
            .group_by(DocumentChunk.file_name)
            .all()
        )
        return {file_name: count for file_name, count in results}

    def delete_by_file(self, file_name: str) -> int:
        count = (
            self.session.query(DocumentChunk)
            .filter_by(file_name=file_name)
            .delete()
        )
        return count

    def delete_all(self) -> int:
        count = self.session.query(DocumentChunk).delete()
        return count


class AchizitieRepository:
    """Repository pentru AchizitieDirecta."""

    def __init__(self, session: Session):
        self.session = session

    def add_batch(self, records: list[dict], progress: bool = False) -> int:
        from tqdm import tqdm
        items = tqdm(records, desc="  achizitii") if progress else records
        for record in items:
            self.session.add(AchizitieDirecta(**record))
        self.session.flush()
        return len(records)

    def count(self) -> int:
        return self.session.query(func.count(AchizitieDirecta.id)).scalar() or 0

    def delete_all(self) -> int:
        return self.session.query(AchizitieDirecta).delete()


class AnuntRepository:
    """Repository pentru AnuntInitiere."""

    def __init__(self, session: Session):
        self.session = session

    def add_batch(self, records: list[dict], progress: bool = False) -> int:
        from tqdm import tqdm
        items = tqdm(records, desc="  anunturi") if progress else records
        for record in items:
            self.session.add(AnuntInitiere(**record))
        self.session.flush()
        return len(records)

    def count(self) -> int:
        return self.session.query(func.count(AnuntInitiere.id)).scalar() or 0

    def delete_all(self) -> int:
        return self.session.query(AnuntInitiere).delete()


class ChatSessionRepository:
    """Repository pentru ChatSession."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, session_id: str) -> ChatSession:
        chat_session = ChatSession(id=session_id)
        self.session.add(chat_session)
        self.session.flush()
        return chat_session

    def get(self, session_id: str) -> ChatSession | None:
        return self.session.query(ChatSession).filter_by(id=session_id).first()

    def get_or_create(self, session_id: str) -> ChatSession:
        chat_session = self.get(session_id)
        if not chat_session:
            chat_session = self.create(session_id)
        return chat_session

    def get_summary(self, session_id: str) -> str:
        chat_session = self.get(session_id)
        if not chat_session or not chat_session.metadata_json:
            return ""
        metadata = json.loads(chat_session.metadata_json)
        return metadata.get("summary", "")

    def save_summary(self, session_id: str, summary: str, message_count: int) -> None:
        chat_session = self.get_or_create(session_id)
        metadata = json.loads(chat_session.metadata_json or "{}")
        metadata["summary"] = summary
        metadata["message_count"] = message_count
        chat_session.metadata_json = json.dumps(metadata)
        self.session.flush()


class ConversationMessageRepository:
    """Repository pentru ConversationMessage."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, session_id: str, role: str, content: str) -> ConversationMessage:
        msg = ConversationMessage(session_id=session_id, role=role, content=content)
        self.session.add(msg)
        self.session.flush()
        return msg

    def get_by_session(self, session_id: str) -> list[ConversationMessage]:
        return (
            self.session.query(ConversationMessage)
            .filter_by(session_id=session_id)
            .order_by(ConversationMessage.created_at)
            .all()
        )

    def get_context(self, session_id: str) -> list[dict]:
        messages = self.get_by_session(session_id)
        return [{"role": m.role, "content": m.content} for m in messages]

    def count_by_session(self, session_id: str) -> int:
        return (
            self.session.query(func.count(ConversationMessage.id))
            .filter_by(session_id=session_id)
            .scalar() or 0
        )

    def delete_by_session(self, session_id: str) -> int:
        return (
            self.session.query(ConversationMessage)
            .filter_by(session_id=session_id)
            .delete()
        )
