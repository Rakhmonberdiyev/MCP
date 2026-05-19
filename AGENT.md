# AGENT.md — Project Blueprint

> Accurate as of May 2026. Describes the live codebase — not a historical snapshot.

---

## 1. What This Project Is

A **production-ready AI assistant for Xazna bank** with a Telegram interface, a FastAPI web dashboard, and a full Rich terminal pipeline display.

**Core features:**
- Two reasoning modes: **System 1** (fast, direct tool-call loop) and **System 2** (Deepthink: Strategy → Self-Critique → Synthesis)
- **Short-term memory** — Redis multi-session history with rolling summaries and active-buffer windowing
- **Long-term memory** — Mem0 backed by Qdrant vectors + Neo4j graph
- **RAG** — users upload PDFs/TXT/MD in Telegram → indexed into Qdrant → searchable
- **8 MCP tool servers** — WebSearch, RAG (local FastMCP) + Deposit, Credit, Pension, Card, Admin, RealTime (remote, proxied through Nginx)
- **Safety guards** on both input and output
- **Grounding / hallucination filter** on every response
- **SSE streaming** — real-time token streaming from the FastAPI web backend
- **Auto-escalation** — user phrases like "deepthink" or Uzbek equivalents override Fast mode for that turn
- **Telegram typing indicator** — refreshed every 4 s while processing

---

## 2. Tech Stack

| Component | Library / Service | Notes |
|---|---|---|
| LLM | Xazna API (OpenAI-compatible) | `https://ai.xazna.uz/llm/v1` |
| Mem0 LLM | Xazna `/models/gemma` | No OpenAI dependency |
| Embeddings | Xazna `/models/embedding` | 2048-dim vectors |
| LTM vector store | Qdrant | collection `mem0`, 2048 dims |
| LTM graph store | Neo4j | relationship triples |
| Mem0 history | SQLite (`mem0_history.db`) | change-log of ADD/UPDATE/DELETE events |
| STM | `redis.asyncio` | multi-session, per-user |
| RAG vector store | Qdrant | collection `knowledge_base`, 2048 dims |
| MCP framework | `fastmcp` | local servers + `create_proxy` for remote |
| Remote MCP gateway | Nginx at `localhost:8080` | routes to 6 bank-domain services |
| Web search | `ddgs` (fallback: `duckduckgo_search`) | |
| PDF extraction | `pypdf` | |
| Telegram bot | `aiogram v3` | `DefaultBotProperties(parse_mode="HTML")` |
| Web backend | `FastAPI` + SSE | `Web/backend/main.py` |
| Terminal UI | `rich` | Console, Panel, Rule, Text, ContextVar queues |
| Infrastructure | Docker Compose | Redis 7, Qdrant, Neo4j |

---

## 3. Project Structure

```
Mem0_full/
├── docker-compose.yml          # Redis + Qdrant + Neo4j
├── .env                        # TELEGRAM_BOT_TOKEN, NEO4J_PASSWORD
├── config.py                   # LLM client, Mem0 config, Qdrant collection bootstrap
├── agent.py                    # Core pipeline: process_turn() + CLI loop
├── telegram_bot.py             # aiogram v3 Telegram interface
├── ui.py                       # Rich terminal + SSE log queue (ContextVar-based)
├── mem0_history.db             # SQLite Mem0 change-log (auto-created)
├── memory/
│   ├── session.py              # Redis: multi-session, active buffer, rolling summaries
│   └── ltm.py                  # Mem0: search (dynamic top-K, score filter, dedup) + upsert
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
└── Web/
    └── backend/
        └── main.py             # FastAPI: SSE chat stream + session management REST API
```

---

## 4. Environment Variables (`.env`)

```env
TELEGRAM_BOT_TOKEN=...          # BotFather token
NEO4J_PASSWORD=password123      # Must match docker-compose NEO4J_AUTH
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

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --appendonly yes
```

Start: `docker compose up -d`

The Nginx gateway (`localhost:8080`) routing Deposit/Credit/Pension/Card/Admin/RealTime MCP services is a separate infrastructure component, not in this docker-compose.

---

## 6. MCP Tool Architecture

### Local tools (defined in this repo)

| Namespaced name | Definition | Description |
|---|---|---|
| `WebSearch_web_search` | `tools/mcp_server.py` | DuckDuckGo live search |
| `RAG_rag_search` | `tools/mcp_server.py` | Qdrant `knowledge_base` semantic search |

### Remote tools (proxied via Nginx at `localhost:8080`)

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

The LLM receives tool schemas like `Pension_get_payment_region_district_street` — the namespace prefix is part of the tool name.

---

## 7. Memory Architecture

### Short-term Memory (Redis)

**Multi-session design** — each user can have up to 20 sessions. Sessions are identified by a UUID-like `s_<hex12>` ID stored in `current_session:{user_id}`.

**Summarized buffer** — instead of sending the full history, the pipeline splits it:
- **Active buffer**: last 5 turns (10 messages) — sent in full every request
- **Session summary**: LLM-generated 2–3 sentence summary of older turns — regenerated every 5 turns in background
- This caps LLM context regardless of session length

Redis key schema:
```
history:{user_id}:{session_id}     → JSON list of messages (TTL 30 days)
sessions:{user_id}                 → JSON list of session metadata (TTL 90 days)
current_session:{user_id}          → active session_id (TTL 90 days)
summary:{user_id}:{session_id}     → rolling session summary text (TTL 30 days)
docs:{user_id}                     → list of uploaded filenames (TTL 30 days)
```

### Long-term Memory (Mem0)

- **Qdrant** (`mem0` collection, 2048 dims): Active atomic facts, auto-deduplicated by Mem0
- **Neo4j**: Relationship graph triples (`source --[rel]--> destination`)
- **SQLite** (`mem0_history.db`): Change-log of every ADD/UPDATE/DELETE Mem0 event
- **Score threshold**: Vector hits below cosine similarity 0.5 are dropped
- **Dynamic top-K** (`memory/ltm.py:query_ltm_limit`):
  - `0` — greetings / very short queries (LTM skipped entirely)
  - `4` — default
  - `7` — complex analytical queries
  - `10` — when deepthink=True
- **Semantic deduplication**: LTM facts with >70% keyword overlap with the active buffer are dropped before injection

---

## 8. Full Pipeline per Turn

```
User Input
    │
    ├── asyncio.gather:
    │     ├── get_active_buffer(user_id)      → Redis → last 5 turns (full)
    │     ├── get_session_summary(user_id)    → Redis → rolling summary text
    │     ├── search_ltm(query, user_id)      → Mem0 → filtered facts string
    │     └── get_user_docs(user_id)          → Redis → list[str]
    │
    ├── deduplicate_ltm_facts(ltm_facts, active_buffer)
    │
    ├── build_messages()
    │     → [system(date+bank_tools+docs+LTM+summary)] + [active_buffer] + [user]
    │
    ├── safety.check_input()    [regex injection guard]
    │
    ├── ROUTING:
    │     deepthink=True or forced → system2.run()
    │     deepthink=False          → system1.run()
    │
    │   System 1 (system1.py):
    │     loop up to 6 rounds:
    │       LLM call (streaming) → tool calls → execute via mcp.call_tool → LLM → ...
    │       → final answer (no tool_calls)
    │
    │   System 2 (system2.py):
    │     Phase 1: _llm() → Thought Signature JSON {goal, approach, confidence, needs_search}
    │     Phase 2: parse — if confidence ≥ 8 and not needs_search → skip Phase 3
    │     Phase 3: loop up to 3 critique rounds:
    │                _llm() → critique JSON {verdict, tool, missing}
    │                if needs_data → _proactive_tool_call() [forces tool via tool_choice API]
    │                               → evidence accumulated
    │                if validated  → break
    │     Phase 4: _llm_with_tools() → synthesis with full tool-execution loop (streaming)
    │
    ├── ground_and_filter(response, evidence, model)
    │     → LLM adds "(unverified)" to claims not supported by evidence
    │     → skipped when evidence == "No external data required."
    │
    ├── safety.check_output()
    │
    └── asyncio.create_task(_persist()):
          ├── save_turn() → Redis
          ├── upsert_ltm() → Mem0
          └── every 5 turns: _generate_summary() → save_session_summary()
```

---

## 9. System Prompt (bank-domain, `pipeline/context_ingestion.py`)

The system prompt instructs the LLM to **always call bank tools first** before answering from general knowledge:

- `Pension_*` → payment schedules (region/district/street)
- `Deposit_*` → deposit products, rates, terms
- `Credit_*` → loan products
- `Card_*` → card products
- `Admin_*` → branches, contacts
- `RealTime_*` → exchange rates, current time
- `RAG_rag_search` → uploaded documents
- `WebSearch_web_search` → last resort for non-bank questions

The system prompt is generated fresh on every request (function, not constant) to inject the current date/time.

---

## 10. System 2 — Key Design Decisions

**Proactive tool call** (`_proactive_tool_call`): the critique LLM names a tool (e.g. `Pension_get_payment_region_district_street`) and the system forces the LLM to call exactly that tool using `tool_choice={"type": "function", "function": {"name": tool_name}}`. This guarantees well-formed arguments instead of relying on the critique LLM to also generate args.

**Tool schema in critique prompt**: the critique prompt receives the full list of available tool names and descriptions so the LLM chooses a real tool name. If the chosen name doesn't exist in the MCP, it falls back to `WebSearch_web_search`.

**Phase 4 must use `_llm_with_tools`**: Final synthesis goes through the full tool loop — if synthesis calls a tool, it executes it. This prevents the LLM from emitting raw tool markup as text.

---

## 11. Web Backend (`Web/backend/main.py`)

FastAPI app. Run with:
```bash
uvicorn Web.backend.main:app --reload --port 8000
```

**SSE chat endpoint** `POST /api/chat/stream`:
- Accepts `{user_id, message, session_id?, deepthink}`
- Streams events: `{type: "token", content: "..."}` during generation, `{type: "done", response: "...", metadata: {...}}` at end
- `metadata` includes `ltm_facts`, `evidence`, `rag_results`, `web_results`, `tool_calls`

**Session endpoints**:
- `GET /api/sessions?user_id=` — list sessions
- `POST /api/sessions?user_id=` — create session
- `POST /api/sessions/{sid}/switch?user_id=` — switch active session
- `GET /api/sessions/{sid}/history?user_id=` — session message history

The `ui.py` module uses `ContextVar` so SSE log events and tool results are captured per-request without global state.

---

## 12. Streaming

Both System 1 and System 2 (Phase 4 Synthesis) support streaming via an `on_chunk(str)` callback:
- Tokens are streamed immediately as they arrive from the LLM
- Tool calls are silently accumulated (not streamed) — only text tokens reach the callback
- Falls back to non-streaming if the LLM or API doesn't support it

The `stream_callback` is threaded from `telegram_bot.py` (used for web UI, not Telegram — Telegram sends one final message).

---

## 13. Telegram Bot (`telegram_bot.py`)

- `aiogram v3`, `parse_mode="HTML"` globally
- Per-user mode stored in `_settings: dict[int, dict]` (in-memory, resets on restart)
- Default mode: **Deepthink ON**
- Inline keyboard on each reply: mode indicator, toggle button, New Session button
- `_keep_typing()`: background task, sends `ChatAction.TYPING` every 4 s
- Document upload (PDF/TXT/MD) → `ingest_document()` → Qdrant + `add_user_doc()` → Redis
- Messages > 4000 chars are split word-by-word

**Commands:**
- `/start` — welcome, reset to Deepthink
- `/mode` — show current mode
- `/newsession` — clear active session history
- `/help` — command list

---

## 14. Config Bootstrap (`config.py`)

On import, `config.py`:
1. Creates the `mem0` Qdrant collection at 2048 dims if it doesn't exist (or recreates it if the dim is wrong)
2. Initializes `ltm_memory = Memory.from_config(MEM0_CONFIG)` — set to `None` if Qdrant is down
3. Both failures are caught and logged — the server can still start without Qdrant

`agent.initialize()` resolves the model ID from the Xazna API at startup; falls back to `/models/gemma` if the API is unreachable.

---

## 15. Critical Implementation Notes

### Qdrant API
`qdrant-client >= 1.17` removed `.search()`. Always use `.query_points()` and access `.points` (not `.result`):
```python
result = _qdrant.query_points(collection_name=..., query=vector, limit=k, with_payload=True)
hits = result.points
```

### Embedding dimensions
Both RAG collection (`knowledge_base`) and Mem0 collection (`mem0`) use **2048-dim** vectors — matching Xazna's `/models/embedding`. The old AGENT.md referenced 1536-dim (OpenAI `text-embedding-3-small`). That is wrong for the current codebase.

### Phase 3 critique loop exit logic
The break check must come **before** the tool call block, not after:
```python
if verdict != "needs_data" or not mcp_tool:
    if crit_conf >= 7 or verdict == "validated":
        break          # ← exit without calling a tool
# Tool call only reached when verdict == "needs_data" AND mcp_tool is set
```

### Tool errors surface as content
When `mcp.call_tool()` raises, the exception becomes the tool result string (`"Tool error: ..."`) and is fed back to the LLM. This lets the LLM decide whether to retry or answer without that data.

### SSE log queue
`ui.set_log_queue(queue)` and `ui.set_tool_collector(collector)` must be called **before** `process_turn()` inside each SSE handler. They use `ContextVar` so different concurrent requests don't interfere.

---

## 16. Required Python Packages

```
openai
mem0ai
redis[asyncio]
qdrant-client>=1.17
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

## 17. How to Run

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Configure secrets
cp .env.example .env   # set TELEGRAM_BOT_TOKEN and NEO4J_PASSWORD

# 3. Install dependencies
pip install openai mem0ai "redis[asyncio]" "qdrant-client>=1.17" fastmcp \
            "aiogram>=3.0" fastapi uvicorn rich pypdf ddgs python-dotenv

# 4a. Run Telegram bot
python telegram_bot.py

# 4b. Or run the web backend
uvicorn Web.backend.main:app --reload --port 8000

# 4c. Or run the CLI
python agent.py

# 5. Verify MCP server connectivity (optional)
python mcp_check.py
```
