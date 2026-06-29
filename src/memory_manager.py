"""
Memory Manager - Logică de sumarizare și selecție peste repositories.

Pattern: SummaryBuffer
- Mesajele raw se salvează în conversation_messages (buffer)
- Când bufferul depășește un prag, mesajele se sumarizează și se șterg
- Rezumatul se salvează în sessions.metadata_json
- La load: summary (istoric comprimat) + mesaje raw recente
"""
import logging
from pathlib import Path

from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from database import transaction
from repositories import ChatSessionRepository, ConversationMessageRepository

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

SUMMARIZE_THRESHOLD = 4  # după câte mesaje se face sumarizarea


class MemoryManager:
    """
    SummaryBuffer persistent peste PostgreSQL.

    - save_message(): salvează un mesaj + verifică dacă trebuie sumarizat
    - load_context(): încarcă summary + mesaje recente ca liste de dict-uri
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

    def ensure_session(self, session_id: str) -> None:
        """Creează sesiunea în DB dacă nu există."""
        with transaction() as db:
            repo = ChatSessionRepository(db)
            repo.get_or_create(session_id)

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Salvează un mesaj și verifică dacă trebuie sumarizat."""
        with transaction() as db:
            msg_repo = ConversationMessageRepository(db)
            msg_repo.add(session_id, role, content)
            count = msg_repo.count_by_session(session_id)

        logger.info(f"[MEMORY] Saved {role} message, total={count}")

        if count >= SUMMARIZE_THRESHOLD:
            self._summarize(session_id)

    def load_context(self, session_id: str) -> dict:
        """
        Încarcă contextul conversației.

        Returns:
            {
                "summary": str,          # rezumatul conversațiilor anterioare
                "recent_messages": [...], # mesaje raw recente (nesumarizate)
            }
        """
        with transaction() as db:
            session_repo = ChatSessionRepository(db)
            msg_repo = ConversationMessageRepository(db)

            summary = session_repo.get_summary(session_id)
            recent_messages = msg_repo.get_context(session_id)

        logger.info(
            f"[MEMORY] Loaded context: summary={'yes' if summary else 'no'}, "
            f"recent_messages={len(recent_messages)}"
        )
        return {
            "summary": summary,
            "recent_messages": recent_messages,
        }

    def _summarize(self, session_id: str) -> None:
        """Sumarizează mesajele, salvează rezumatul, șterge mesajele raw."""
        logger.info(f"[MEMORY] Summarizing session {session_id}")

        with transaction() as db:
            session_repo = ChatSessionRepository(db)
            msg_repo = ConversationMessageRepository(db)

            messages = msg_repo.get_context(session_id)
            existing_summary = session_repo.get_summary(session_id)

        # Generează rezumatul cu LLM
        prompt = self.prompts.render(
            "conversation_summarize",
            messages=messages,
            existing_summary=existing_summary,
        )
        summary = self.llm.generate_sync([{"role": "user", "content": prompt}])

        # Salvează rezumatul și șterge mesajele raw (într-o singură tranzacție)
        with transaction() as db:
            session_repo = ChatSessionRepository(db)
            msg_repo = ConversationMessageRepository(db)

            total_count = msg_repo.count_by_session(session_id)
            session_repo.save_summary(session_id, summary, total_count)
            msg_repo.delete_by_session(session_id)

        logger.info(f"[MEMORY] Summary saved, {len(messages)} messages deleted")
