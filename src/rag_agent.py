"""
RAG Agent - Graf pentru căutare în pgvector

Flow:
    START → refine → search → END

- refine: dacă are feedback, rafinează query-ul (TODO pentru studenți)
- search: caută în pgvector (COMPLET)
"""
import json
import logging
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import (
    RAGAgentState,
    RAGSearchResult,
    SearchResultItem,
    RefinedQuery,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class RAGAgentConfig:
    top_k: int = 5
    default_threshold: float = 0.25


class RAGAgent:
    """
    Agent pentru căutare în pgvector.

    Flow:
        refine → search

    Dacă primește feedback de la Orchestrator, rafinează query-ul
    înainte de a căuta.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: RAGAgentConfig | None = None,
    ):
        self.llm = llm
        self.config = config or RAGAgentConfig()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))
        self.graph = self._build_graph()

    # === NODES ===

    def node_refine(self, state: RAGAgentState) -> dict:
        """
        TODO: Rafinează query-ul dacă avem feedback.

        Dacă state.feedback este None (prima căutare):
            - Returnează refined = RefinedQuery(query=state.query)

        Dacă state.feedback există (orchestratorul a zis că nu poate răspunde):
            1. Construiește found_summary din state.result.results (dacă există)
            2. Renderează prompt "rag_refine" cu:
               - original_query, current_query, found_summary
               - max_score, avg_score, current_threshold
               - feedback (can_answer, missing_info, suggestion)
            3. Apelează LLM
            4. Parsează JSON în RefinedQuery cu model_validate_json()
            5. Return {"refined": RefinedQuery(...)}
        """
        logger.info(f"[REFINE] feedback={state.feedback is not None}")

        # Dacă nu există feedback (prima căutare) — returnează query-ul original
        if not state.feedback:
            return {"refined": RefinedQuery(query=state.query)}

        # Construiește found_summary din rezultatele anterioare
        found_summary = "Nimic găsit."
        if state.result and state.result.results:
            found_summary = "\n".join(
                f"- [{r.file_name}] (score={r.score:.2f}): {r.summary or r.content[:100]}"
                for r in state.result.results
            )

        # Renderează prompt "rag_refine"
        prompt = self.prompts.render(
            "rag_refine",
            original_query=state.query,
            current_query=state.current_query,
            found_summary=found_summary,
            max_score=state.result.max_score if state.result else 0.0,
            avg_score=state.result.avg_score if state.result else 0.0,
            current_threshold=state.current_threshold or self.config.default_threshold,
            can_answer=state.feedback.can_answer,
            missing_info=state.feedback.missing_info,
            suggestion=state.feedback.suggestion,
        )

        # Apelează LLM
        response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        # Parsează JSON direct în Pydantic
        try:
            refined = RefinedQuery.model_validate_json(response)
        except Exception:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            json_str = match.group(1) if match else response
            refined = RefinedQuery.model_validate_json(json_str)

        return {"refined": refined}

    def node_search(self, state: RAGAgentState) -> dict:
        """Caută chunks similare în pgvector. COMPLET - nu modifica."""
        from database import transaction
        from rag_service import RAGService

        query = state.current_query
        threshold = state.current_threshold or self.config.default_threshold

        logger.info(f"[SEARCH] '{query}' (top_k={self.config.top_k}, threshold={threshold})")

        items = []
        with transaction() as db:
            rag = RAGService(db)
            results = rag.search(query, top_k=self.config.top_k, threshold=threshold)

            # Copy data into Pydantic objects while session is still open
            items = [
                SearchResultItem(
                    content=chunk.content,
                    summary=chunk.summary or "",
                    file_name=chunk.file_name,
                    score=score,
                )
                for chunk, score in results
            ]

        # Calculează statistici
        scores = [item.score for item in items]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "result": RAGSearchResult(
                query_used=query,
                results=items,
                max_score=max_score,
                avg_score=avg_score,
            )
        }

    def _extract_metadata(self, query: str) -> dict[str, str]:
        """Folosește LLM-ul să extragă metadata filters din query (doc_type, company_name)."""
        prompt = (
            "Extract metadata from this Romanian query.\n\n"
            f"Query: {query}\n\n"
            "Return exactly 2 lines:\n"
            "Line 1 - document type. Choose ONE of: factura, contract, client, raport, none\n"
            "Line 2 - company name mentioned in the query, or none if no company is mentioned\n\n"
            "Example for 'Care e totalul facturilor TechSoft?':\nfactura\ntechsoft"
        )
        response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        lines = [l.strip().lower() for l in response.strip().split("\n") if l.strip()]

        filters = {}
        if lines and lines[0] in ("factura", "contract", "client", "raport"):
            filters["doc_type"] = lines[0]
        if len(lines) > 1 and lines[1] != "none":
            filters["company_name"] = lines[1]

        logger.info(f"[METADATA] filters={filters}")
        return filters

    def node_search_hybrid(self, state: RAGAgentState) -> dict:
        """Hybrid search: vector similarity filtered by metadata (company_name, doc_type)."""
        from database import transaction
        from rag_service import RAGService

        query = state.current_query
        threshold = state.current_threshold or self.config.default_threshold

        metadata_filters = self._extract_metadata(query)

        logger.info(f"[SEARCH HYBRID] '{query}' (top_k={self.config.top_k}, threshold={threshold}, filters={metadata_filters})")

        items = []
        with transaction() as db:
            rag = RAGService(db)

            # Vector search with metadata filters (company_name, doc_type)
            results = rag.search_filtered(
                query,
                top_k=self.config.top_k,
                threshold=threshold,
                metadata_filters=metadata_filters if metadata_filters else None,
            )

            items = [
                SearchResultItem(
                    content=chunk.content,
                    summary=chunk.summary or "",
                    file_name=chunk.file_name,
                    score=score,
                )
                for chunk, score in results
            ]

        logger.info(f"[SEARCH HYBRID] results={len(items)}, filters={metadata_filters}")

        # Log chunks to debug file
        from debug_log import log_chunks
        log_chunks(query, getattr(state, 'iteration', 0) if hasattr(state, 'iteration') else 0, items)

        scores = [item.score for item in items]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "result": RAGSearchResult(
                query_used=query,
                results=items,
                max_score=max_score,
                avg_score=avg_score,
            )
        }

    # === GRAPH ===

    def _build_graph(self):
        """Construiește graful RAG Agent."""
        graph = StateGraph(RAGAgentState)

        graph.add_node("refine", self.node_refine)
        graph.add_node("search", self.node_search_hybrid)

        graph.add_edge(START, "refine")
        graph.add_edge("refine", "search")
        graph.add_edge("search", END)

        return graph.compile()

    def run(self, query: str, feedback=None) -> RAGAgentState:
        """Execută agentul."""
        initial = RAGAgentState(query=query, feedback=feedback)
        return self.graph.invoke(initial)
