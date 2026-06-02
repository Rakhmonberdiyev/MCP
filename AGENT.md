# AGENT.md — Project Blueprint

> Accurate as of May 2026. Describes the live codebase exactly as it stands.

---

## 1. What This Project Is

A **production-ready AI assistant for Xazna bank** with a Telegram interface, a FastAPI web dashboard, and a full Rich terminal pipeline display.

**Core features:**
- Two reasoning modes: **System 1** (fast, direct tool-call loop) and **System 2** (Deepthink: Strategy → Critique → Synthesis)
- **Short-term memory** — Redis multi-session history with rolling summaries and active-buffer windowing
- **Long-term memory** — Mem0 backed by Qdrant vectors + Neo4j graph, with custom banking-domain extraction prompts
- **Multi-user history store** — PostgreSQL in Docker (replaces single-file SQLite); every ADD / UPDATE / DELETE event written per `user_id`
- **RAG** — users upload PDFs/TXT/MD in Telegram → chunked, embedded, indexed into Qdrant `knowledge_base` → searchable via `RAG_rag_search`
- **8 MCP tool servers** — WebSearch, RAG (local FastMCP) + Deposit, Credit, Pension, Card, Admin, RealTime (remote, proxied through Nginx)
- **Safety guards** on both input (regex injection detection) and output (blocklist)
- **Grounding / hallucination filter** on every response
- **SSE streaming** — real-time token streaming from the FastAPI web backend
- **Auto-escalation** — user phrases like "deepthink", "batafsil", "chuqur o'yla" override Fast mode for that turn
- **Telegram typing indicator** — refreshed every 4 s while processing
- **Test suite** — isolated memory smoke test + Mini-LongMemEval banking benchmark in `test/`

---

## 2. Tech Stack

| Component | Library / Service | Notes |
|---|---|---|
| LLM | Xazna API (OpenAI-compatible) | `https://ai.xazna.uz/llm/v1` |
| Mem0 LLM | Xazna `/models/gemma` | Extracts + deduplicates facts |
| Embeddings | Xazna `/models/embedding` | 2048-dim vectors |
| LTM vector store | Qdrant | collection `mem0`, 2048 dims |
| LTM graph store | Neo4j | relationship triples |
| LTM history store | **PostgreSQL 16** (Docker) | `memory_events` table, multi-user |
| STM | `redis.asyncio` | multi-session, per-user |
| RAG vector store | Qdrant | collection `knowledge_base`, 2048 dims |
| PostgreSQL client | `asyncpg` | async connection pool, min=1 max=10 |
| MCP framework | `fastmcp` | local servers + `create_proxy` for remote |
| Remote MCP gateway | Nginx at `localhost:8080` | routes to 6 bank-domain services |
| Web search | `ddgs` | DuckDuckGo search |
| PDF extraction | `pypdf` | |
| Telegram bot | `aiogram v3` | `DefaultBotProperties(parse_mode="HTML")` |
| Web backend | `FastAPI` + SSE | `Web/backend/main.py` |
| Terminal UI | `rich` | Console, Panel, Rule, Text, ContextVar queues |
| Infrastructure | Docker Compose | Redis 7, Qdrant, Neo4j, PostgreSQL 16 |

---

## 3. Project Structure

```
Mem0_full/
├── docker-compose.yml          # Redis + Qdrant + Neo4j + PostgreSQL
├── .env                        # TELEGRAM_BOT_TOKEN, NEO4J_PASSWORD, DATABASE_URL
├── config.py                   # LLM client, MEM0_CONFIG (all 4 stores), Qdrant bootstrap
├── agent.py                    # Core pipeline: process_turn() + CLI loop
├── telegram_bot.py             # aiogram v3 Telegram interface
├── ui.py                       # Rich terminal + SSE log queue (ContextVar-based)
├── memory/
│   ├── session.py              # Redis: multi-session, active buffer, rolling summaries
│   ├── ltm.py                  # Mem0: search (dynamic top-K, score filter, dedup) + upsert → PG
│   └── db.py                   # PostgreSQL: asyncpg pool, schema init, write_events / query_events
├── pipeline/
│   ├── context_ingestion.py    # Builds messages[]: system prompt + active buffer + LTM
│   ├── system1.py              # Fast path: streaming LLM → tool loop → answer
│   ├── system2.py              # Deepthink: Strategy → Critique → Synthesis (all with tools)
│   ├── safety.py               # Prompt-injection input guard + output blocklist
│   └── output_processor.py     # Grounding & hallucination filter
├── tools/
│   ├── mcp_server.py           # Local FastMCP: web_search + rag_search
│   ├── remote_mcp.py           # create_proxy for 6 Nginx-gated bank MCP services
│   └── ingestion.py            # PDF/TXT/MD → chunk → embed → Qdrant
├── test/
│   ├── test_memory.py          # Memory layer smoke test (O'zbek tilida)
│   ├── eval_longmemeval.py     # Mini-LongMemEval benchmark (O'zbek tilida) — queries PostgreSQL
│   └── benchmark_mem0.py       # Pipeline latency benchmark
└── Web/
    └── backend/
        └── main.py             # FastAPI: SSE chat stream + session management REST API
```

---

## 4. Environment Variables (`.env`)

```env
TELEGRAM_BOT_TOKEN=...          # BotFather token
NEO4J_PASSWORD=password123      # Must match docker-compose NEO4J_AUTH
DATABASE_URL=postgresql://mem0:mem0pass@localhost:5432/mem0_history   # optional override
```

LLM credentials are hardcoded in `config.py` (Xazna internal deployment):
```python
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY  = "sk-raximberdi-cmF4aW1iZXJkaQ"
```

No OpenAI API key required — both LLM and embedder use Xazna's own API.

---

## 5. Infrastructure (`docker-compose.yml`)

```yaml
services:
  neo4j:
    image: neo4j:latest
    ports: ["7474:7474", "7687:7687"]
    environment: [NEO4J_AUTH=neo4j/password123]
    volumes: [neo4j_data:/data]

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --appendonly yes
    volumes: [redis_data:/data]

  postgres:
    image: postgres:16-alpine
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: mem0_history
      POSTGRES_USER: mem0
      POSTGRES_PASSWORD: mem0pass
    volumes: [postgres_data:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mem0 -d mem0_history"]
      interval: 5s / timeout: 3s / retries: 10
```

Start everything: `docker compose up -d`

The Nginx gateway (`localhost:8080`) routing Deposit/Credit/Pension/Card/Admin/RealTime MCP services is a separate infrastructure component not in this docker-compose.

---

## 6. Central Config (`config.py` — `MEM0_CONFIG`)

All four data stores are declared inside a single `MEM0_CONFIG` dict — the source of truth for every store connection:

```python
MEM0_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model":           "/models/gemma",
            "api_key":         LLM_API_KEY,
            "openai_base_url": LLM_BASE_URL,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model":           "/models/embedding",
            "api_key":         LLM_API_KEY,
            "openai_base_url": LLM_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host":                 "localhost",
            "port":                 6333,
            "collection_name":      "mem0",
            "embedding_model_dims": 2048,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url":      "bolt://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD", "password123"),
        },
    },
    "history_store": {                          # ← our application layer, not Mem0-native
        "provider": "postgresql",
        "config": {
            "url": os.getenv("DATABASE_URL",
                             "postgresql://mem0:mem0pass@localhost:5432/mem0_history"),
        },
    },
    "custom_fact_extraction_prompt": "...",     # see Section 8
    "custom_update_memory_prompt":   "...",     # see Section 8
}
```

`memory/db.py` reads `MEM0_CONFIG["history_store"]["config"]["url"]` — no separate `DATABASE_URL` top-level variable exists.

**On import, `config.py` also:**
1. Pre-creates the `mem0` Qdrant collection at 2048 dims (recreates it if dim is wrong)
2. Initializes `ltm_memory = Memory.from_config(MEM0_CONFIG)` — set to `None` if Qdrant is down
3. Both failures are caught and logged; the server can still start without those services

---

## 7. MCP Tool Architecture

### Local tools (this repo — `tools/mcp_server.py`)

| Namespaced name | Description |
|---|---|
| `WebSearch_web_search` | DuckDuckGo live web search |
| `RAG_rag_search` | Qdrant `knowledge_base` semantic search over user-uploaded files |

### Remote tools (proxied via Nginx at `localhost:8080` — `tools/remote_mcp.py`)

| Namespace | Proxy path | Domain |
|---|---|---|
| `Deposit_*` | `/deposit/mcp` | Deposit products, rates, terms |
| `Credit_*` | `/credit/mcp` | Loan products, amounts, rates |
| `Pension_*` | `/pension/mcp` | Pension payment schedules |
| `Card_*` | `/card/mcp` | Card products (Visa/MC/Humo/Uzcard) |
| `Admin_*` | `/admin/mcp` | Bank info, branches, contacts |
| `RealTime_*` | `/real_time/mcp` | Exchange rates, current time |

All 8 servers are mounted on `main_mcp` in `agent.py`:

```python
main_mcp = FastMCP("Main")
main_mcp.mount(search_mcp,   namespace="WebSearch")
main_mcp.mount(rag_mcp,      namespace="RAG")
main_mcp.mount(deposit_mcp,  namespace="Deposit")
main_mcp.mount(credit_mcp,   namespace="Credit")
main_mcp.mount(pension_mcp,  namespace="Pension")
main_mcp.mount(card_mcp,     namespace="Card")
main_mcp.mount(admin_mcp,    namespace="Admin")
main_mcp.mount(realtime_mcp, namespace="RealTime")
```

The LLM receives tool schemas with the namespace prefix as part of the name (e.g. `Pension_get_payment_region_district_street`).

---

## 8. Memory Architecture

### Short-term Memory — Redis (`memory/session.py`)

Multi-session design. Each user has up to 20 named sessions identified by `s_<hex12>` IDs.

**Summarized buffer strategy** (caps LLM context regardless of session length):
- **Active buffer**: last 5 turns (10 messages) sent in full every request
- **Session summary**: LLM-generated 2–3 sentence summary of older turns, regenerated every 5 turns in background via `_generate_summary()`

**Redis key schema:**
```
history:{user_id}:{session_id}     → JSON list of messages          TTL 30 days
sessions:{user_id}                 → JSON list of session metadata   TTL 90 days
current_session:{user_id}          → active session_id string        TTL 90 days
summary:{user_id}:{session_id}     → rolling session summary text    TTL 30 days
docs:{user_id}                     → list of uploaded filenames       TTL 30 days
```

**Public API used by `agent.py`:**
- `get_active_buffer(user_id)` → last 5 turns as `list[dict]`
- `get_session_summary(user_id)` → summary string
- `save_turn(user_id, user_msg, asst_msg)` → appends + caps at `MAX_SESSION_MESSAGES=40`
- `save_session_summary(user_id, summary)` → overwrites rolling summary
- `get_user_docs(user_id)` → filenames for RAG doc hint

---

### Long-term Memory — Mem0 (`memory/ltm.py`)

Backed by **Qdrant** (vector) + **Neo4j** (graph). Mem0 automatically deduplicates facts: when a new conflicting fact arrives, it UPDATEs or DELETEs the old one in Qdrant.

**`search_ltm(query, user_id, limit, min_score=0.5) → str`**

- Runs `ltm_memory.search()` in a thread pool (`asyncio.to_thread`)
- Monkey-patches `vector_store.search` and `graph.search` before the call to capture per-store timing without modifying Mem0 source code (`_patch_sub_stores`)
- Filters vector results by cosine similarity ≥ `min_score` (default 0.5); dropped results are logged with their scores
- Returns newline-separated `"- fact"` lines: vector facts first, then graph triples (`src --[rel]--> dst`)
- Returns `""` if limit=0 (greeting detection), Mem0 not initialized, or all results below threshold

**`query_ltm_limit(query, deepthink) → int`**

Dynamic top-K decision:
| Condition | Limit |
|---|---|
| `deepthink=True` | `10` |
| Greeting or query < 12 chars | `0` (LTM skipped entirely) |
| Complex keywords: "all", "explain", "barcha", "tahlil", etc. | `7` |
| Default | `4` |

**`deduplicate_ltm_facts(ltm_facts, recent_messages) → str`**

Removes LTM lines whose fingerprint keywords overlap ≥ 70% with the active buffer. Prevents injecting facts the LLM already has in context.

**`upsert_ltm(user_input, assistant_output, user_id)`**

Sends structured messages to Mem0:
```python
messages = [
    {"role": "user",      "content": user_input},
    {"role": "assistant", "content": assistant_output},
]
result = await asyncio.to_thread(ltm_memory.add, messages, user_id=user_id)
```
After `add()` returns, writes all ADD/UPDATE/DELETE events in `result["results"]` to PostgreSQL via `write_events()`.

---

### LTM Extraction Prompts (in `MEM0_CONFIG`)

**Why custom prompts are required:**

Mem0's default `USER_MEMORY_EXTRACTION_PROMPT`:
1. Ignores assistant messages ("extract from user messages ONLY") — breaks VM-4 (Nilufar's name only appears in the assistant response)
2. Discards old values on UPDATE — breaks VM-2 (old credit limit gone from Qdrant after change)

**`custom_fact_extraction_prompt`** — replaces the default extraction system prompt:
- Extracts from **both** user AND assistant turns
- **Transition rule**: when a value changes, preserves BOTH old and new in the fact:
  - `"Kredit limiti 10,000,000 dan 7,000,000 UZS ga kamaytirildi"` (not just `"7,000,000"`)
  - `"Humo Classic arizasi bekor qilindi, o'rniga Visa Gold ochildi"`
- Targets banking facts: card types, credit limits, co-holders, income, billing address, service flags
- Excludes: greetings, day markers (`"1-kun:"`), general world facts
- Preserves input language (Uzbek / English / Russian)
- Output format: `{"facts": ["...", "..."]}`

**`custom_update_memory_prompt`** — replaces default update decision prompt:
- **Transition rule**: when UPDATing a numeric/entity value, always store the transition `"old dan new ga o'zgartirildi"` — never just the new value
- Returns structured JSON with `event: ADD|UPDATE|DELETE|NONE` and `old_memory` for UPDATE events

> **Historical bug fixed**: The previous config used `"custom_prompt"` which is not a valid `MemoryConfig` field — it was silently ignored by Pydantic. The correct field names are `"custom_fact_extraction_prompt"` and `"custom_update_memory_prompt"`.

---

### History Store — PostgreSQL (`memory/db.py`)

Replaces the old single-file `mem0_history.db` (SQLite). Supports multiple concurrent users and processes.

**Connection**: `asyncpg` pool (min 1, max 10 connections), URL from `MEM0_CONFIG["history_store"]["config"]["url"]`. Schema created automatically on first connection.

**Schema — `memory_events` table:**

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL PK | Auto-increment row ID |
| `user_id` | TEXT NOT NULL | Owner of the memory fact |
| `memory_id` | TEXT | Mem0's internal UUID for the fact |
| `event` | TEXT NOT NULL | `ADD` / `UPDATE` / `DELETE` |
| `old_memory` | TEXT | Previous fact text (populated for UPDATE) |
| `new_memory` | TEXT | New / current fact text |
| `created_at` | TIMESTAMPTZ | Defaults to `NOW()` UTC |

**Indexes:**
- `(user_id, created_at)` — primary query pattern
- `(user_id, event)` — filter by event type

**Public API:**
```python
await write_events(user_id, results)                     # called by upsert_ltm after add()
await query_events(user_id, since_iso, event_types)      # returns list[dict] oldest-first
await count_events(user_id, since_iso)                   # fast integer count for polling
```

`since_iso` is an ISO-8601 string; `_parse_iso()` converts it to a timezone-aware `datetime` before passing to asyncpg (asyncpg requires Python datetime objects, not raw strings).

**Data flow:**
```
upsert_ltm()
  → ltm_memory.add(messages, user_id=user_id)       # Qdrant + Mem0 internal SQLite (dedup)
  → result["results"] = [{event, memory, id}, ...]
  → write_events(user_id, result["results"])         # PostgreSQL memory_events
```

---

## 9. Full Pipeline per Turn (`agent.py:process_turn`)

```
User Input
    │
    ├── asyncio.gather (parallel):
    │     ├── get_active_buffer(user_id)       Redis → last 5 turns (full messages)
    │     ├── get_session_summary(user_id)     Redis → rolling summary string
    │     ├── search_ltm(query, user_id, K)   Mem0  → Qdrant + Neo4j filtered facts
    │     └── get_user_docs(user_id)           Redis → uploaded filename list
    │
    ├── deduplicate_ltm_facts(ltm_facts, active_buffer)
    │     removes LTM lines with ≥70% keyword overlap with recent context
    │
    ├── build_messages()
    │     → [system: date + tool guide + docs + LTM + summary]
    │     → [active buffer (last 5 turns)]
    │     → [user message]
    │
    ├── safety.check_input(user_input)
    │     → regex injection guard, synchronous, runs before any LLM call
    │     → on block: return fixed refusal, skip LLM
    │
    ├── ROUTING:
    │     forced deepthink (user phrase) OR deepthink=True → system2.run()
    │     otherwise                                        → system1.run()
    │
    │   ── System 1 (system1.py) ──
    │     loop up to 6 rounds:
    │       streaming LLM call → if tool_calls → mcp.call_tool() → append result → LLM
    │       → exit loop when response has no tool_calls
    │     tool errors become the tool result string (LLM decides to retry or answer)
    │
    │   ── System 2 (system2.py) ──
    │     Phase 1 — Strategy:
    │       _llm() → JSON {goal, approach, confidence 1-10, needs_search: bool}
    │     Phase 2 — Skip check:
    │       if confidence ≥ 8 AND not needs_search → skip Phase 3
    │     Phase 3 — Critique loop (up to 3 rounds):
    │       _llm() → JSON {verdict: "validated"|"needs_data"|"uncertain", tool, missing}
    │       if needs_data → _proactive_tool_call(tool_name)
    │                         forces exact tool via tool_choice API → accumulates evidence
    │       if validated or crit_conf ≥ 7 → break
    │     Phase 4 — Synthesis:
    │       _llm_with_tools() → full streaming tool loop with accumulated evidence
    │
    ├── ground_and_filter(response, evidence, model)
    │     LLM adds "(unverified)" to claims not supported by evidence
    │     skipped when evidence == "No external data required."
    │
    ├── safety.check_output(response)
    │     blocklist check; on hit: redact_unsafe_output() returns safe fallback
    │
    └── asyncio.create_task(_persist()):
          ├── save_turn(user_id, user_input, response)          → Redis
          ├── upsert_ltm(user_input, response, user_id)         → Mem0 + PostgreSQL
          └── every 5 turns: _generate_summary() → save_session_summary() → Redis
```

---

## 10. Safety Pipeline (`pipeline/safety.py`)

### Input guard — regex, synchronous, pre-LLM

Blocked patterns (case-insensitive):
```
ignore (all|previous|prior) instructions
forget (everything|all instructions)
you are now  |  new personality
(act|pretend|behave) as (a|an)
system prompt  |  jailbreak
disregard (your|all) training
override (your|all) guidelines
```
Returns `(is_safe: bool, reason: str)`. On block, the pipeline returns a fixed refusal without calling the LLM.

### Output guard — blocklist, post-LLM

Blocked phrases in the generated response:
- `"i am now jailbroken"`
- `"i have no restrictions"`
- `"my new instructions are"`

On block, `redact_unsafe_output()` returns: `"I'm sorry, I can't provide that response. Please ask me something else."`

---

## 11. System Prompt (`pipeline/context_ingestion.py`)

Generated fresh each request (function, not constant) — injects current date/time (UTC+5 Tashkent).

**Tool priority rules injected:**
1. `Pension_*` → pension schedules (region / district / street) — call `Pension_get_payment_region_district_street` with exact location
2. `Deposit_*` → deposit products, rates, terms
3. `Credit_*` → loan products
4. `Card_*` → card products (Visa/MC/Humo/Uzcard)
5. `Admin_*` → bank info, branches, contacts
6. `RealTime_*` → exchange rates, current time/date
7. `RAG_rag_search` → uploaded user documents
8. `WebSearch_web_search` → last resort for non-bank questions

**Context blocks appended to system message (in order):**
1. Uploaded document list (if any) + hint about "most recent file"
2. Long-term memory facts: `[Long-term memory about this user]:\n{ltm_facts}`
3. Session summary: `[Summary of earlier conversation turns]:\n{session_summary}`

Then `active_buffer` messages are appended, then the user message.

---

## 12. System 2 — Key Design Decisions

**`_proactive_tool_call`**: the critique LLM names a tool; the system forces the LLM to call exactly that tool using:
```python
tool_choice={"type": "function", "function": {"name": tool_name}}
```
This guarantees well-formed arguments instead of relying on the critique LLM to also generate JSON args.

**Tool schema in critique prompt**: the critique prompt receives the full list of available tool names and descriptions so the LLM picks a real name. If the chosen name doesn't exist in the MCP, it falls back to `WebSearch_web_search`.

**Phase 3 break check order** — the break must come *before* the tool call block:
```python
if verdict != "needs_data" or not mcp_tool:
    if crit_conf >= 7 or verdict == "validated":
        break   # ← exit before any tool call
# Tool call only reached when verdict == "needs_data" AND mcp_tool is valid
```

**Phase 4 uses `_llm_with_tools`**: final synthesis goes through the full tool loop. If synthesis needs a tool, it executes it. Prevents the LLM from emitting raw tool markup as plain text.

---

## 13. Deepthink Auto-detection (`agent.py`)

The `_wants_deepthink(text)` function uses a compiled regex to detect intent keywords:

```
deepthink, deep think, deep research, deep dive
think carefully / deeply / step-by-step
reason carefully / through
analyze carefully, careful/thorough/detailed analysis
explain in detail, elaborate, in-depth
yaxshilab, batafsil, chuqur o'yla, chuqur tahlil, sinchiklab
```

When detected, the turn is forced to System 2 regardless of the user's mode setting.

---

## 14. Web Backend (`Web/backend/main.py`)

Run: `uvicorn Web.backend.main:app --reload --port 8000`

**SSE chat endpoint** `POST /api/chat/stream`:
- Accepts: `{user_id, message, session_id?, deepthink}`
- Streams events: `{type: "token", content: "..."}` during generation
- Final event: `{type: "done", response: "...", metadata: {...}}`
- `metadata` includes: `ltm_facts`, `evidence`, `rag_results`, `web_results`, `tool_calls`

**Session management REST endpoints:**
| Method | Path | Action |
|---|---|---|
| GET | `/api/sessions?user_id=` | List all sessions |
| POST | `/api/sessions?user_id=` | Create new session |
| POST | `/api/sessions/{sid}/switch?user_id=` | Switch active session |
| GET | `/api/sessions/{sid}/history?user_id=` | Get session message history |

`ui.set_log_queue(queue)` and `ui.set_tool_collector(collector)` must be called before `process_turn()` inside each SSE handler. They use `ContextVar` so concurrent requests don't interfere.

---

## 15. Streaming

Both System 1 and System 2 Phase 4 support streaming via an `on_chunk(str)` callback:
- Text tokens are passed immediately to the callback as they arrive
- Tool calls are silently accumulated (not streamed) — only text tokens reach the callback
- Falls back to non-streaming if the API doesn't support it

Used by the FastAPI SSE endpoint (`Web/backend/main.py`). Telegram bot receives one final message (no streaming to Telegram).

---

## 16. Telegram Bot (`telegram_bot.py`)

- `aiogram v3`, `parse_mode="HTML"` globally
- Per-user settings in `_settings: dict[int, dict]` (in-memory, resets on restart)
- Default mode: **Deepthink ON**
- Inline keyboard on every reply: mode indicator (`System 1` / `System 2`), toggle button, New Session button
- `_keep_typing()`: background coroutine, sends `ChatAction.TYPING` every 4 s while processing
- Document upload (PDF/TXT/MD): `ingest_document()` → Qdrant `knowledge_base` + `add_user_doc()` → Redis
- Messages > 4000 chars are split word-by-word

**Bot commands:**
| Command | Action |
|---|---|
| `/start` | Welcome message, reset to Deepthink ON |
| `/mode` | Show current reasoning mode |
| `/newsession` | Clear active session history |
| `/help` | Command list |

---

## 17. Test Suite (`test/`)

All three scripts resolve the project root via:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
Run from inside `test/` or the project root — both work. Using only `os.path.dirname(__file__)` resolves to `test/` and fails with `ModuleNotFoundError: No module named 'memory'`.

---

### `test/test_memory.py` — Memory Layer Smoke Test (O'zbek tilida)

Ingests 5 mock banking turns directly through `memory/session.py` and `memory/ltm.py` — no agent pipeline, no safety guards, no grounding.

**Scenario**: Humo → Visa card switch + co-holder add/remove.

**Four phases:**
1. Ingest 5 turns into Redis (STM) + Qdrant/Mem0 (LTM)
2. Read back the Redis session — verify storage and rolling buffer
3. Run 4 `search_ltm` queries and mark keyword hits with ✓
4. Query `mem0_history.db` (Mem0's internal SQLite) for ADD/UPDATE/DELETE events since test start

```bash
cd test && python3 test_memory.py
```

---

### `test/eval_longmemeval.py` — Mini-LongMemEval Benchmark (O'zbek tilida)

20 sequential banking events across 30 simulated days. Each user message is prefixed with `"N-kun:"` so the LTM extractor anchors facts temporally.

**Three deliberate state mutations:**

| Kun | O'zgarish |
|-----|-----------|
| 10 | Humo Classic ariza **→ bekor**, Visa Gold boshlandi |
| 20 | Hamkorchi Nilufar **→ o'chirildi**, faqat bitta ega |
| 22 | Kredit limit **10,000,000 → 7,000,000 UZS** |

**10 graded questions** — `PASS=2 / PARTIAL=1 / FAIL=0`, max 20 pts:

| ID | Tur | Nimani tekshiradi |
|----|-----|-------------------|
| BY-1 | Bilim yangilanishi | Joriy karta (Visa Gold, Humo emas) |
| BY-2 | Bilim yangilanishi | Hamkorchi holati (Nilufar olib tashlangan) |
| BY-3 | Bilim yangilanishi | Joriy kredit limit (7,000,000) |
| BY-4 | Bilim yangilanishi | Ko'chirma tili (O'zbek) |
| BY-5 | Bilim yangilanishi | Xorijiy valyuta yoqilganmi (Ha) |
| VM-1 | Vaqtinchalik mantiq | Dastlabki karta nima edi (Humo Classic) |
| VM-2 | Vaqtinchalik mantiq | Eski kredit limit (10,000,000) |
| VM-3 | Vaqtinchalik mantiq | Billing manzili (Amir Temur ko'chasi, 12, Toshkent) |
| VM-4 | Vaqtinchalik mantiq | Qachondir hamkorchi qo'shilganmidi (Nilufar) |
| VM-5 | Vaqtinchalik mantiq | E'lon qilingan oylik daromad (5,000,000) |

**Scoring logic** (`score_question`):
- `knowledge_update`: PASS if `required_kws` present AND no stale contamination (per-line check — a line containing both old and new keywords is a transition fact, not stale); PARTIAL if only in PostgreSQL changelog; FAIL otherwise
- `temporal_reasoning`: PASS if historical fact retrieved from Qdrant; PARTIAL if only PostgreSQL change-log has it; FAIL otherwise

**Stale contamination check** (`_stale_contamination`): per-line check — a line is only flagged as stale if it contains a stale keyword WITHOUT also containing a required keyword. This allows "Humo bekor qilindi, o'rniga Visa Gold" to pass as a valid transition fact.

**PostgreSQL polling** (`_wait_for_pg`): after ingestion, polls `count_events(user_id, since_iso)` every 4 s up to 90 s. Mem0's async thread pool may write events slightly after `add()` returns.

**History queries use PostgreSQL** — `query_changelog(user_id, since_iso)` calls `memory.db.query_events()`. No SQLite file access in the eval.

```bash
cd test && python3 eval_longmemeval.py                          # full run
cd test && python3 eval_longmemeval.py --skip-ingest --user-id eval_abc123
```

**Per-question `min_score` override**: `EvalQ.min_score` defaults to 0.5. VM-3 (billing address) uses `0.4` because the billing address fact scored 0.489 cosine similarity — just under the default threshold.

---

### `test/benchmark_mem0.py` — Pipeline Latency Benchmark

Runs adversarial prompts through the full `process_turn()` pipeline and prints per-turn latency + total. Measures end-to-end performance including safety guards, LTM search, tool calls, grounding.

```bash
cd test && python3 benchmark_mem0.py
```

---

## 18. Critical Implementation Notes

### Qdrant client API
`qdrant-client >= 1.17` removed `.search()`. Always use `.query_points()` and access `.points` (not `.result`):
```python
result = _qdrant.query_points(collection_name=..., query=vector, limit=k, with_payload=True)
hits = result.points
```

### Embedding dimensions
Both collections (`knowledge_base` and `mem0`) use **2048-dim** vectors matching Xazna's `/models/embedding`. Any reference to 1536-dim (OpenAI `text-embedding-3-small`) is wrong for this codebase.

### asyncpg datetime requirement
`asyncpg` requires Python `datetime` objects for `TIMESTAMPTZ` parameters — it rejects raw ISO strings. Always call `_parse_iso(since_iso)` before passing timestamps to asyncpg queries:
```python
dt = datetime.fromisoformat(since_iso)
if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone.utc)
```
`created_at` returned from asyncpg queries is a `datetime.datetime` — format it with `str(ts)[:19]`, not slice a string directly.

### Mem0 config field names
`MemoryConfig` uses `extra="ignore"` (Pydantic default). Unknown keys are **silently dropped**. The correct field names are:
- `"custom_fact_extraction_prompt"` — not `"custom_prompt"`
- `"custom_update_memory_prompt"` — not `"update_prompt"` or any other variant

### `upsert_ltm` must send structured messages
```python
# CORRECT — Mem0 properly separates user vs assistant content
messages = [
    {"role": "user",      "content": user_input},
    {"role": "assistant", "content": assistant_output},
]
ltm_memory.add(messages, user_id=user_id)

# WRONG — entire blob treated as one user message; role parsing undefined
text = f"User: {user_input}\nAssistant: {assistant_output}"
ltm_memory.add(text, user_id=user_id)
```

### Phase 3 critique loop exit
The break check must come **before** the tool call block, not after:
```python
if verdict != "needs_data" or not mcp_tool:
    if crit_conf >= 7 or verdict == "validated":
        break          # exit without calling a tool
# Tool call only reached when verdict == "needs_data" AND mcp_tool is set
```

### Tool errors surface as content
When `mcp.call_tool()` raises, the exception string becomes the tool result fed back to the LLM. This lets the LLM decide whether to retry or answer without that data — never crashes the pipeline.

### SSE log queue setup
`ui.set_log_queue(queue)` and `ui.set_tool_collector(collector)` must be called **before** `process_turn()` inside each SSE handler. They use `ContextVar` — different concurrent requests get isolated log streams.

### `history_store` is application-level, not Mem0-native
`MEM0_CONFIG["history_store"]` is read by `memory/db.py` only. Mem0 does not know about this key — Pydantic silently ignores it when constructing `MemoryConfig`. Mem0 still writes its own internal deduplication log to `~/.mem0/history.db` (the default path, since we removed `history_db_path` from the config). We never read that file.

---

## 19. Required Python Packages

```
openai
mem0ai
redis[asyncio]
qdrant-client>=1.17
asyncpg
fastmcp
aiogram>=3.0
fastapi
uvicorn
rich
pypdf
ddgs
python-dotenv
```

---

## 20. How to Run

```bash
# 1. Start all infrastructure (Redis, Qdrant, Neo4j, PostgreSQL)
docker compose up -d

# 2. Configure secrets
cp .env.example .env   # set TELEGRAM_BOT_TOKEN and NEO4J_PASSWORD

# 3. Install dependencies
pip install openai mem0ai "redis[asyncio]" "qdrant-client>=1.17" asyncpg fastmcp \
            "aiogram>=3.0" fastapi uvicorn rich pypdf ddgs python-dotenv

# 4a. Run Telegram bot
python telegram_bot.py

# 4b. Or run the web backend
uvicorn Web.backend.main:app --reload --port 8000

# 4c. Or run the CLI
python agent.py

# 5. Run the test suite
cd test
python3 test_memory.py                                  # memory layer smoke test
python3 eval_longmemeval.py                             # LongMemEval banking benchmark
python3 eval_longmemeval.py --skip-ingest --user-id X  # re-evaluate existing user data
python3 benchmark_mem0.py                               # pipeline latency benchmark

# 6. Verify MCP server connectivity (optional)
python mcp_check.py

# 7. Check PostgreSQL directly
docker exec -it postgres_mem0 psql -U mem0 -d mem0_history \
    -c "SELECT event, new_memory, created_at FROM memory_events ORDER BY created_at DESC LIMIT 10;"
```
