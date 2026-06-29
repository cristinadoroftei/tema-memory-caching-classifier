# Temă: Multi-Agent System

Două sisteme:
1. **Orchestrator + RAG** - caută în documente
2. **Analyst + NL2SQL** - query-uri SQL

## Arhitectură

### 1. Orchestrator + RAG (Hierarchical Multi-Agent)

```
┌──────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                               │
│                        (Supervizor)                               │
│                                                                   │
│    ┌──────────┐      ┌──────────┐      ┌────────┐                │
│    │ call_rag │ ───► │ evaluate │ ───► │ answer │ ───► END       │
│    └──────────┘      └──────────┘      └────────┘                │
│         ▲                  │                                      │
│         │                  │ can_answer=false                     │
│         │                  │ + feedback                           │
│         └──────────────────┘                                      │
│                                                                   │
│    max 3 iterații                                                 │
└──────────────────────────────────────────────────────────────────┘
         │                   ▲
         │ query +           │ RAGSearchResult
         │ feedback          │
         ▼                   │
┌──────────────────────────────────────────────────────────────────┐
│                         RAG AGENT                                 │
│                         (Worker)                                  │
│                                                                   │
│    ┌────────┐      ┌────────┐                                    │
│    │ refine │ ───► │ search │ ───► END                           │
│    └────────┘      └────────┘                                    │
│                                                                   │
│    refine: dacă are feedback, rafinează query-ul                 │
│    search: caută în pgvector, returnează chunks                  │
└──────────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Orchestrator apelează RAG Agent cu query
2. RAG Agent caută și returnează chunks
3. Orchestrator evaluează: "Pot răspunde?"
4. Dacă NU → trimite feedback, RAG Agent rafinează și caută din nou
5. Dacă DA → generează răspuns final

**Prompturi:** `rag_evaluate.yaml`, `rag_answer.yaml`, `rag_refine.yaml`

### 2. Analyst + NL2SQL (Hierarchical Multi-Agent)

```
┌──────────────────────────────────────────────────────────────────┐
│                      ANALYST AGENT                                │
│                      (Supervizor)                                 │
│                                                                   │
│    ┌───────────┐      ┌──────────────┐      ┌─────────────┐      │
│    │ make_plan │ ───► │ execute_step │ ───► │ synthesize  │──►END │
│    └───────────┘      └──────────────┘      └─────────────┘      │
│                              │  ▲                                 │
│                              └──┘ loop                            │
│                                                                   │
│    Plan: [QueryStep, QueryStep, ToolStep, ...]                    │
│    Slices: {"q1": DataFrame, "q2": DataFrame, "joined": DataFrame}│
└──────────────────────────────────────────────────────────────────┘
           │                              │
           │ QueryStep                    │ ToolStep
           ▼                              ▼
┌──────────────────────────────┐    ┌─────────────────────────┐
│        NL2SQL AGENT          │    │      TOOL REGISTRY      │
│         (Worker)             │    │                         │
│                              │    │  join_data(dfs, keys)   │
│  get_context                 │    │  filter_data(df, cond)  │
│      │                       │    │                         │
│      ▼                       │    └─────────────────────────┘
│  generate_sql                │
│      │                       │
│      ▼                       │
│  validate_sql ───┬───► execute_sql ───┬───► END (success)
│                  │           │        │
│                  │ invalid   │ error  │
│                  ▼           ▼        │
│             handle_error ◄────────────┘
│                  │
│            retry < max?
│             yes │ no
│                 ▼  ▼
│         generate_sql  END (failed)
└──────────────────────────────┘
```

**Flow Analyst:**
1. `make_plan` - LLM generează plan cu QueryStep și ToolStep
2. `execute_step` - execută fiecare pas:
   - QueryStep → apelează NL2SQL Agent → DataFrame în `slices[id]`
   - ToolStep → apelează tool (join/filter) → DataFrame în `slices[id]`
3. `synthesize` - LLM generează răspuns din rezultate

**Flow NL2SQL:**
1. `get_context` - încarcă schema tabelului
2. `generate_sql` - LLM generează SQL
3. `validate_sql` - validează cu sqlparse
4. `execute_sql` - execută în DB → DataFrame
5. `handle_error` - dacă eroare, LLM corectează SQL și retry

**Prompturi:** `analyst_plan.yaml`, `analyst_synthesize.yaml`, `nl2sql_generate.yaml`, `nl2sql_error.yaml`

## Setup

```bash
pip install -r requirements.txt
pip install -e skillab-py

docker-compose up -d
alembic upgrade head

# Restaurează date (694k achiziții, 8k anunțuri, 135 chunks)
docker exec -i exercise_orchestrator-postgres-1 pg_restore -U demo -d rag_demo --data-only < data/rag_demo.dump

# Adaugă metadata (doc_type + company_name) pe document_chunks
cd scripts && python add_doc_type_metadata.py && cd ..

cp .env.example .env  # editează API key
```

## Structură

```
├── alembic/           # Migrații DB
├── data/              # CSV-uri, documente
├── prompts/           # YAML prompts
├── scripts/           # Seed scripts
├── skillab-py/        # LLM, prompts, tools
│   └── src/skillab/tools/
│       ├── implementations.py  # TODO: join_data, filter_data
│       └── params.py           # Pydantic params
└── src/
    ├── database.py        # Connection + transaction
    ├── models.py          # SQLAlchemy models
    ├── repositories.py    # Repository pattern
    ├── rag_service.py     # pgvector search service
    ├── state.py           # Pydantic states
    ├── rag_agent.py       # TODO: node_refine
    ├── orchestrator.py    # TODO: node_evaluate, node_answer
    ├── nl2sql_agent.py    # TODO: node_generate_sql, node_validate_sql, node_execute_sql
    ├── analyst_agent.py   # TODO: node_make_plan, node_synthesize
    └── main.py
```

## De implementat

### 1. RAG Agent (`src/rag_agent.py`)
```python
def node_refine(self, state: RAGAgentState) -> dict:
    """
    Dacă state.feedback există:
    1. Renderează prompt "rag_refine"
    2. Apelează LLM
    3. Parsează JSON în RefinedQuery.model_validate_json()
    4. Return {"refined": refined_query}

    Dacă nu există feedback:
    - Return {"refined": RefinedQuery(query=state.query)}
    """
```

### 2. Orchestrator (`src/orchestrator.py`)
```python
def node_evaluate(self, state: OrchestratorState) -> dict:
    """
    1. Construiește context din state.rag_result.results
    2. Renderează prompt "rag_evaluate"
    3. Apelează LLM
    4. Parsează în OrchestratorFeedback.model_validate_json()
    5. Return {"feedback": feedback}
    """

def node_answer(self, state: OrchestratorState) -> dict:
    """
    1. Construiește context din state.rag_result.results
    2. Renderează prompt "rag_answer"
    3. Apelează LLM
    4. Return {"answer": answer, "status": "success"|"partial"|"failed"}
    """
```

### 3. NL2SQL Agent (`src/nl2sql_agent.py`)
```python
def node_generate_sql(self, state) -> dict:
    # Generează SQL din întrebare

def node_validate_sql(self, state) -> dict:
    # Validează SQL (sqlparse)

def node_execute_sql(self, state) -> dict:
    # Execută SQL, returnează DataFrame
```

### 4. Analyst Agent (`src/analyst_agent.py`)
```python
def node_make_plan(self, state) -> dict:
    # Creează plan cu QueryStep și ToolStep

def node_synthesize(self, state) -> dict:
    # Sintetizează răspuns din state.slices
```

### 5. Tools (`skillab-py/src/skillab/tools/implementations.py`)
```python
@register_tool
def join_data(params: JoinDataParams) -> pd.DataFrame:
    # pd.merge(params.input_dfs[0], params.input_dfs[1], ...)

@register_tool
def filter_data(params: FilterDataParams) -> pd.DataFrame:
    # params.input_dfs[0][mask]
```

## Plan format

LLM generează plan JSON:
```json
[
  {"id": "q1", "action": "query", "table": "achizitii", "sub_question": "..."},
  {"id": "q2", "action": "query", "table": "anunturi", "sub_question": "..."},
  {"id": "joined", "action": "tool", "tool_name": "join_data", "input_steps": ["q1", "q2"], "params": {"left_key": "cui", "right_key": "cui"}},
  {"id": "result", "action": "tool", "tool_name": "filter_data", "input_steps": ["joined"], "params": {"column": "valoare", "operator": ">", "value": "50000"}}
]
```

Rezultate în `state.slices["q1"]`, `state.slices["joined"]`, etc.

## Hints

```python
# Render prompt
prompt = self.prompts.render("rag_evaluate", query=q, context=ctx, ...)

# LLM call
response = self.llm.generate_sync([{"role": "user", "content": prompt}])

# Parse JSON direct în Pydantic (recomandat)
import re
match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
json_str = match.group(1) if match else response
feedback = OrchestratorFeedback.model_validate_json(json_str)

# SQL execution
with transaction() as session:
    result = session.execute(text(sql_query))
    df = pd.DataFrame(result.mappings().all())

# Tool catalog pentru prompt
tools_catalog = ToolWrapper.to_prompt_string()
```

## Run

```bash
cd src && python main.py
```

## Enhancements (beyond TODOs)

### 1. Metadata-based Hybrid Search

RAG Agent-ul folosește un **hybrid search** care combină vector similarity cu filtrare pe metadata.

**Cum funcționează:**

1. LLM-ul extrage metadata din query (method `_extract_metadata` in `rag_agent.py`):
   - `doc_type`: factura / contract / client / raport
   - `company_name`: numele companiei menționate în query (ex: TechSoft, DataPro)

2. Vector search-ul rulează cu WHERE clauses pe `metadata_json` (`search_similar_filtered` in `repositories.py`):
   ```sql
   WHERE embedding IS NOT NULL
     AND LOWER(metadata_json::jsonb->>'company_name') = LOWER(:company_name)
     AND LOWER(metadata_json::jsonb->>'doc_type') = LOWER(:doc_type)
   ORDER BY embedding <=> query_vector
   ```

3. Comparația e **case-insensitive** (LOWER pe ambele părți) — "TechSoft", "techsoft", "TECHSOFT" toate funcționează.

**Metadata pe document_chunks:**

Scriptul `scripts/add_doc_type_metadata.py` adaugă în `metadata_json`:
- `doc_type` — derivat din prefixul file_name (factura_, client_, contract_, raport_)
- `company_name` — derivat din pattern-ul file_name (techsoft, datapro, cloudnet, secureit, webdev)

```
file_name                          | doc_type | company_name
-----------------------------------|----------|-------------
factura_0004_techsoft_202406.docx  | factura  | techsoft
client_datapro_sa.docx             | client   | datapro
raport_Q1_2024.docx                | raport   | (none)
```

**Fișiere modificate:**
- `src/rag_agent.py` — `_extract_metadata()` + `node_search_hybrid()` (original `node_search` nemodificat)
- `src/repositories.py` — `search_similar_filtered()` (original `search_similar` nemodificat)
- `src/rag_service.py` — `search_filtered()`

### 2. Debug Logging

Fiecare run scrie un log detaliat în `logs/run_TIMESTAMP.md` cu:
- **Orchestrator+RAG**: chunks retrieve (cu scores), feedback evaluare, răspuns final
- **Analyst+NL2SQL**: plan generat, SQL generat per step, rezultate execuție (preview primele 10 rânduri), răspuns sintetizat

Fișier: `src/debug_log.py`

### 3. Bugs Fixed

1. **DetachedInstanceError** (`rag_agent.py`): chunk attributes accessed after session close — moved SearchResultItem creation inside `with transaction()` block
2. **dict vs object** (`orchestrator.py`, `analyst_agent.py`): LangGraph `invoke()` returns dict, not Pydantic model — changed `.attr` to `["key"]`
3. **Missing packages**: `pip install anthropic sqlparse`
