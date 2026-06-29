# Tema Lectia 7-8: Memory, Caching & Intent Classifier

Proiect bazat pe multi-agent system-ul din lectia 5-6 (Orchestrator + RAG + Analyst + NL2SQL), extins cu:

1. **Conversation Memory** — memorie persistenta intre request-uri (L8)
2. **Prompt Caching** — cache pe system prompts cu Anthropic API (L8)
3. **Intent Classifier** — clasificare intent cu scikit-learn, integrat in graf (L7)

---

## Task 1: Conversation Memory

**Problema:** Fiecare request era independent — agentul nu stia ce s-a discutat inainte.

**Solutia:** SummaryBuffer pattern — mesajele raw se salveaza in PostgreSQL, iar cand depasesc un threshold (4 mesaje), se sumarizeaza automat si se sterg cele vechi.

### Ce s-a construit

- **Doua tabele noi**: `sessions` + `conversation_messages` (migratie `004`)
- **MemoryManager** (`src/memory_manager.py`): save_message, load_context, _summarize
- **Memory check routing**: LLM-ul decide daca intrebarea poate fi raspunsa doar din memorie (fara RAG)

### Flow actualizat

```
load_memory → check_memory ──┬──→ answer_from_memory → save_memory → END
                             └──→ ... (RAG/Analyst) → save_memory → END
```

### Fisiere cheie
- `src/memory_manager.py` — logica SummaryBuffer
- `src/models.py` — `ChatSession`, `ConversationMessage`
- `src/repositories.py` — `ChatSessionRepository`, `ConversationMessageRepository`
- `prompts/conversation_summarize.yaml`, `prompts/memory_check.yaml`
- `alembic/versions/004_create_conversation_messages.py`

---

## Task 2: Prompt Caching

**Problema:** System prompt-ul se trimite la fiecare request, consumand tokeni si timp.

**Solutia:** Anthropic `cache_control: {"type": "ephemeral"}` — system prompt-ul se cache-uieste pe server-ul Anthropic dupa primul call.

### Ce s-a construit

- **Modificat `_prepare_messages()`** in `skillab-py/src/skillab/llm/providers/anthropic.py`:
  - System message wrapped cu `cache_control`
  - `self.last_usage` expune token usage (cache_creation/read)
- **Benchmark script** (`scripts/benchmark_caching.py`):
  - Trimite acelasi query de 2 ori cu system prompt mare
  - Call 1: creeaza cache
  - Call 2: citeste din cache (mai rapid, mai ieftin)

### Rezultate benchmark
- Call 1: cache creation (tokeni normali)
- Call 2: cache read (tokeni redusi, latenta mai mica)
- Nota: Anthropic necesita minim ~1024 tokeni in system prompt pentru activare

### Cum se ruleaza
```bash
cd scripts && python benchmark_caching.py
```

---

## Task 3: Intent Classifier cu scikit-learn

**Problema:** Orchestratorul nu stia ce tip de intrebare primeste — totul mergea la RAG Agent.

**Solutia:** Un clasificator ML (TF-IDF + LogisticRegression) care detecteaza intentul fara LLM si ruteaza catre agentul potrivit.

### Cele 3 intente

| Intent | Descriere | Ruteaza catre |
|--------|-----------|---------------|
| `search` | Cauta/gaseste documente | RAG Agent |
| `extract` | Date numerice, totaluri, statistici | Analyst Agent → NL2SQL |
| `summarize` | Rezumat/sinteza informatii | RAG Agent |

### Ce s-a construit

**1. Date de antrenament** (`data/intent_training_data.csv`)
- 90 exemple etichetate (30 per intent), in romana
- Domeniu: achizitii publice, licitatii, furnizori

**2. Script antrenament** (`scripts/train_intent_classifier.py`)
- Pipeline: TF-IDF Vectorizer → LogisticRegression
- TF-IDF converteste text in vectori numerici (cuvintele rare per intent au pondere mare)
- Cross-validation accuracy: **92%**
- Model salvat: `data/intent_classifier.joblib` (36KB)

**3. Clasa IntentClassifier** (`src/intent_classifier.py`)
- Incarca modelul `.joblib` la startup
- `predict(query)` → returneaza label, confidence, all_scores
- Fara apel LLM, fara cost, ~3ms per query

**4. Integrare in Orchestrator** (`src/orchestrator.py`)
- Nod nou `classify_intent` in graf
- Nod nou `call_analyst` pentru intent `extract`
- Routing: `extract` → Analyst Agent, `search`/`summarize` → RAG Agent

**5. Script comparatie** (`scripts/compare_intent_methods.py`)
- Aceleasi query-uri clasificate de sklearn vs LLM
- Masoara: accuracy, latenta, tokeni, cost

### Flow complet (cu toate cele 3 task-uri)

```
load_memory → check_memory ──┬──→ answer_from_memory → save_memory → END
                             │
                             └──→ classify_intent ──┬──→ call_rag → evaluate → answer → save_memory → END
                                  (sklearn, ~3ms)   │     (search & summarize)
                                                    │
                                                    └──→ call_analyst → save_memory → END
                                                          (extract → NL2SQL)
```

### Rezultate comparatie sklearn vs LLM

| Metric | sklearn | LLM |
|--------|---------|-----|
| Accuracy | 100% (10/10) | 100% (10/10) |
| Avg latency | **3.3 ms** | 1,360 ms |
| Cost | **$0 (free)** | ~$0.004 / 10 queries |
| Speedup | **415x** | baseline |

### Cum se ruleaza

```bash
# Antreneaza clasificatorul (optional — modelul e deja salvat)
python scripts/train_intent_classifier.py

# Ruleaza comparatia sklearn vs LLM
python scripts/compare_intent_methods.py
```

---

## Setup

```bash
# Instalare dependinte
pip install -r requirements.txt
pip install scikit-learn joblib
pip install -e skillab-py

# Porneste PostgreSQL + pgvector
docker-compose up -d

# Aplica migratii (inclusiv tabelele de conversation memory)
alembic upgrade head

# Restaureaza date
docker exec -i exercise_orchestrator-postgres-1 pg_restore -U demo -d rag_demo --data-only < data/rag_demo.dump

# Configureaza API key
cp .env.example .env  # editează cu cheia ta
```

## Run

```bash
cd src && python main.py
```

## Structura proiect

```
src/
  main.py                — entry point (chat interactiv cu sesiuni)
  orchestrator.py        — Orchestrator LangGraph (memory + intent + RAG/Analyst)
  intent_classifier.py   — [NOU] IntentClassifier (sklearn)
  memory_manager.py      — [NOU] SummaryBuffer: save, load, summarize
  rag_agent.py           — RAG Agent (refine → search)
  analyst_agent.py       — Analyst Agent (plan → execute → synthesize)
  nl2sql_agent.py        — NL2SQL Agent (generate → validate → execute SQL)
  state.py               — Pydantic state models (+ intent, intent_confidence)
  models.py              — SQLAlchemy models (+ ChatSession, ConversationMessage)
  repositories.py        — Repository pattern (+ ChatSession/Message repos)
scripts/
  train_intent_classifier.py  — [NOU] Antreneaza TF-IDF + LogReg
  compare_intent_methods.py   — [NOU] Comparatie sklearn vs LLM
  benchmark_caching.py        — [NOU] Benchmark prompt caching
data/
  intent_training_data.csv    — [NOU] 90 exemple etichetate
  intent_classifier.joblib    — [NOU] Model antrenat (36KB)
prompts/
  conversation_summarize.yaml — [NOU] Prompt sumarizare conversatie
  memory_check.yaml           — [NOU] Prompt verificare memorie
```
