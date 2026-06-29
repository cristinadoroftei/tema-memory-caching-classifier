"""
Orchestrator - Supervizor care coordonează RAG Agent și Analyst Agent

Flow:
    load_memory → check_memory ──┬──→ answer_from_memory → save_memory → END
                                 │
                                 └──→ classify_intent ──┬──→ call_rag → evaluate ──┬──→ answer → save_memory → END
                                      (sklearn)        │        ↑                 │
                                                       │        │   can_answer    │
                                                       │        │   = false       │
                                                       │        └─────────────────┘
                                                       │
                                                       └──→ call_analyst → save_memory → END
                                                             (extract intent)
"""
import logging
import re
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import OrchestratorState, OrchestratorFeedback
from rag_agent import RAGAgent, RAGAgentConfig
from memory_manager import MemoryManager
from intent_classifier import IntentClassifier
from analyst_agent import AnalystAgent

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class OrchestratorConfig:
    max_iterations: int = 3
    min_score: float = 0.25
    tables_config: dict[str, dict] | None = None  # for AnalystAgent (extract intent)


class Orchestrator:
    """
    Supervizor care coordonează RAG Agent.

    Flow:
        1. Apelează RAG Agent pentru căutare
        2. Evaluează: pot răspunde cu aceste chunks?
        3. Dacă DA → generează răspuns
        4. Dacă NU → trimite feedback la RAG Agent, repeat
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        llm: LLMProvider | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # RAG Agent - graf separat
        rag_config = RAGAgentConfig()
        rag_config.default_threshold = self.config.min_score
        self.rag = RAGAgent(self.llm, rag_config)

        # Memory Manager
        self.memory = MemoryManager(self.llm)

        # Intent Classifier (sklearn)
        self.intent_classifier = IntentClassifier()

        # Analyst Agent (for extract intent) — optional, needs tables_config
        self.analyst = None
        if self.config.tables_config:
            self.analyst = AnalystAgent(
                tables_config=self.config.tables_config,
                llm=self.llm,
            )

    # === NODES ===

    def node_load_memory(self, state: OrchestratorState) -> dict:
        """Încarcă istoricul conversației din DB (summary + mesaje recente)."""
        if not state.session_id:
            return {}
        self.memory.ensure_session(state.session_id)
        context = self.memory.load_context(state.session_id)
        return {
            "conversation_history": context["recent_messages"],
            "summary": context["summary"],
        }

    def node_check_memory(self, state: OrchestratorState) -> dict:
        """Verifică dacă întrebarea poate fi răspunsă doar din memorie."""
        if not state.summary and not state.conversation_history:
            logger.info("[CHECK_MEMORY] No memory, skipping to RAG")
            return {"can_answer_from_memory": False}

        prompt = self.prompts.render(
            "memory_check",
            query=state.query,
            summary=state.summary,
            recent_messages=state.conversation_history,
        )
        response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        can_answer = response.strip().upper().startswith("DA")
        logger.info(f"[CHECK_MEMORY] can_answer_from_memory={can_answer}")
        return {"can_answer_from_memory": can_answer}

    def node_answer_from_memory(self, state: OrchestratorState) -> dict:
        """Generează răspuns doar din memoria conversației, fără RAG."""
        logger.info("[ANSWER_FROM_MEMORY]")

        history_parts = []
        if state.summary:
            history_parts.append(f"Rezumat conversație anterioară:\n{state.summary}")
        if state.conversation_history:
            recent = "\n".join(
                f"{m['role']}: {m['content']}" for m in state.conversation_history
            )
            history_parts.append(f"Mesaje recente:\n{recent}")

        messages = [
            {"role": "system", "content": "\n\n".join(history_parts)},
            {"role": "user", "content": state.query},
        ]
        answer = self.llm.generate_sync(messages)
        return {"answer": answer, "status": "success"}

    def node_classify_intent(self, state: OrchestratorState) -> dict:
        """Clasifică intenția query-ului folosind sklearn (fără LLM)."""
        result = self.intent_classifier.predict(state.query)
        logger.info(f"[CLASSIFY_INTENT] {result['label']} ({result['confidence']:.0%})")
        return {
            "intent": result["label"],
            "intent_confidence": result["confidence"],
        }

    def node_call_analyst(self, state: OrchestratorState) -> dict:
        """Apelează Analyst Agent pentru query-uri de tip extract (date numerice, SQL)."""
        logger.info(f"[CALL_ANALYST] {state.query}")
        result = self.analyst.chat(state.query)
        return {"answer": result["answer"], "status": result["status"]}

    def node_call_rag(self, state: OrchestratorState) -> dict:
        """Apelează RAG Agent pentru căutare. COMPLET - nu modifica."""
        logger.info(f"[CALL_RAG] iter {state.iteration + 1}")

        # Apelează RAG Agent cu query și feedback (dacă există)
        rag_result = self.rag.run(
            query=state.query,
            feedback=state.feedback,  # None prima dată
        )

        return {
            "rag_result": rag_result["result"],
            "iteration": state.iteration + 1,
        }

    def node_evaluate(self, state: OrchestratorState) -> dict:
        """
        TODO: Evaluează dacă contextul RAG e suficient.

        1. Construiește context din state.rag_result.results:
           context = "\\n\\n".join(f"[{r.file_name}]\\n{r.content}" for r in results)

        2. Renderează prompt "rag_evaluate" cu:
           - query=state.query
           - context=context
           - max_score=state.rag_result.max_score
           - avg_score=state.rag_result.avg_score

        3. Apelează LLM:
           response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        4. Parsează JSON în Pydantic:
           - Extrage JSON din ```json ... ```
           - feedback = OrchestratorFeedback.model_validate_json(json_str)

        5. Return {"feedback": feedback}
        """
        logger.info(f"[EVALUATE] iter {state.iteration}")

        # 1. Construiește context din state.rag_result.results
        results = state.rag_result.results if state.rag_result else []
        context = "\n\n".join(f"[{r.file_name}]\n{r.content}" for r in results)

        # 2. Renderează prompt "rag_evaluate"
        prompt = self.prompts.render(
            "rag_evaluate",
            query=state.query,
            context=context,
            max_score=state.rag_result.max_score if state.rag_result else 0.0,
            avg_score=state.rag_result.avg_score if state.rag_result else 0.0,
        )

        # 3. Apelează LLM
        response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        # 4. Parsează JSON direct în Pydantic
        try:
            feedback = OrchestratorFeedback.model_validate_json(response)
        except Exception:
            # Fallback: LLM-ul a pus JSON în ```json ... ```
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            json_str = match.group(1) if match else response
            feedback = OrchestratorFeedback.model_validate_json(json_str)

        # Log feedback to debug file
        from debug_log import log_feedback
        log_feedback(state.iteration, feedback)

        return {"feedback": feedback}

    def node_answer(self, state: OrchestratorState) -> dict:
        """
        TODO: Generează răspunsul final.

        1. Construiește context din state.rag_result.results:
           context = "\\n\\n".join(f"[{r.file_name}]\\n{r.content}" for r in results)

        2. Renderează prompt "rag_answer" cu:
           - query=state.query
           - context=context

        3. Apelează LLM:
           answer = self.llm.generate_sync([{"role": "user", "content": prompt}])

        4. Determină status:
           - "success" dacă feedback.can_answer == True
           - "partial" dacă am răspuns dar fără can_answer
           - "failed" dacă nu avem rezultate

        5. Return {"answer": answer, "status": status}
        """
        logger.info("[ANSWER]")

        # 1. Construiește context din state.rag_result.results:
        results = state.rag_result.results if state.rag_result else []
        context = "\n\n".join(f"[{r.file_name}]\n{r.content}" for r in results)

        # 2. Renderează prompt "rag_answer" cu query si context
        prompt = self.prompts.render("rag_answer", query=state.query, context=context)

        # 3. Construiește mesajele pentru LLM cu istoricul conversației
        messages = []

        # Adaugă summary + mesaje recente ca system context
        if state.summary or state.conversation_history:
            history_parts = []
            if state.summary:
                history_parts.append(f"Rezumat conversație anterioară:\n{state.summary}")
            if state.conversation_history:
                recent = "\n".join(
                    f"{m['role']}: {m['content']}" for m in state.conversation_history
                )
                history_parts.append(f"Mesaje recente:\n{recent}")
            messages.append({
                "role": "system",
                "content": "\n\n".join(history_parts),
            })

        messages.append({"role": "user", "content": prompt})

        answer = self.llm.generate_sync(messages)

        # 4. Determină status:
        if state.feedback and state.feedback.can_answer:
            status = "success"
        elif results:
            status = "partial"
        else:
            status = "failed"

        # Log full answer to debug file
        from debug_log import log_answer
        log_answer(state.query, status, answer)

        return {"answer": answer, "status": status}

    def node_save_memory(self, state: OrchestratorState) -> dict:
        """Salvează întrebarea și răspunsul în DB (+ sumarizare dacă e cazul)."""
        if not state.session_id:
            return {}
        self.memory.save_message(state.session_id, "user", state.query)
        self.memory.save_message(state.session_id, "assistant", state.answer)
        return {}

    # === ROUTING ===

    def _memory_or_classify(self, state: OrchestratorState) -> Literal["answer_from_memory", "classify_intent"]:
        """Decide dacă răspundem din memorie sau clasificăm intenția."""
        if state.can_answer_from_memory:
            return "answer_from_memory"
        return "classify_intent"

    def _intent_route(self, state: OrchestratorState) -> Literal["call_rag", "call_analyst"]:
        """Rutează pe baza intenției clasificate: extract → Analyst, altfel → RAG."""
        if state.intent == "extract" and self.analyst:
            return "call_analyst"
        return "call_rag"

    def _should_continue(self, state: OrchestratorState) -> Literal["call_rag", "answer"]:
        """Decide dacă continuăm căutarea sau răspundem."""
        if state.feedback and state.feedback.can_answer:
            return "answer"
        if state.iteration >= self.config.max_iterations:
            logger.info(f"[ROUTING] Max iterations ({self.config.max_iterations}) reached")
            return "answer"
        return "call_rag"

    # === GRAPH ===

    def build_graph(self):
        """Construiește graful Orchestrator."""
        graph = StateGraph(OrchestratorState)

        graph.add_node("load_memory", self.node_load_memory)
        graph.add_node("check_memory", self.node_check_memory)
        graph.add_node("answer_from_memory", self.node_answer_from_memory)
        graph.add_node("classify_intent", self.node_classify_intent)
        graph.add_node("call_rag", self.node_call_rag)
        graph.add_node("call_analyst", self.node_call_analyst)
        graph.add_node("evaluate", self.node_evaluate)
        graph.add_node("answer", self.node_answer)
        graph.add_node("save_memory", self.node_save_memory)

        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "check_memory")
        graph.add_conditional_edges(
            "check_memory",
            self._memory_or_classify,
            {"answer_from_memory": "answer_from_memory", "classify_intent": "classify_intent"}
        )
        graph.add_edge("answer_from_memory", "save_memory")
        graph.add_conditional_edges(
            "classify_intent",
            self._intent_route,
            {"call_rag": "call_rag", "call_analyst": "call_analyst"}
        )
        graph.add_edge("call_analyst", "save_memory")
        graph.add_edge("call_rag", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"call_rag": "call_rag", "answer": "answer"}
        )
        graph.add_edge("answer", "save_memory")
        graph.add_edge("save_memory", END)

        return graph.compile()
