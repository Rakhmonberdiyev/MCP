import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()


LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY = "sk-raximberdi-cmF4aW1iZXJkaQ"

llm_client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# Model ID is resolved at startup via initialize() in agent.py
MODEL_ID: str = ""

# --- Mem0 (LTM): Qdrant vectors + Neo4j graph ---
#
# Memory architecture:
#   Qdrant   (via Mem0)  — Active atomic facts, current state only.
#                          Mem0 auto-deduplicates: updates/deletes old facts
#                          when new conflicting facts arrive.
#   PostgreSQL           — Multi-user event log (ADD / UPDATE / DELETE).
#                          Written by upsert_ltm() after each ltm_memory.add().
#                          Replaces the single-file mem0_history.db.
#   Neo4j               — Relationship graph (entity triples).
#
# Both LLM and embedder use Xazna — no OpenAI dependency.
#
MEM0_CONFIG = {
    "llm": {
        "provider": "openai",           # openai-compatible protocol
        "config": {
            "model":            "/models/gemma",
            "api_key":          LLM_API_KEY,
            "openai_base_url":  LLM_BASE_URL,
        },
    },
    "embedder": {
        "provider": "openai",           # openai-compatible protocol
        "config": {
            "model":            "/models/embedding",
            "api_key":          LLM_API_KEY,
            "openai_base_url":  LLM_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host":                   "localhost",
            "port":                   6333,
            "collection_name":        "mem0",
            "embedding_model_dims":   2048,
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
    "history_store": {
        "provider": "postgresql",
        "config": {
            "url": os.getenv(
                "DATABASE_URL",
                "postgresql://mem0:mem0pass@localhost:5432/mem0_history",
            ),
        },
    },
    # ── Fact extraction ────────────────────────────────────────────────────────
    # Replaces USER_MEMORY_EXTRACTION_PROMPT which (a) ignores assistant messages
    # and (b) discards old values on UPDATE.  Both gaps break temporal queries.
    "custom_fact_extraction_prompt": """\
You are a banking-domain memory extractor.
Extract meaningful long-term facts from the FULL conversation — both user AND assistant turns.
Assistant confirmations frequently contain names, values, and transitions that are essential to remember.

OUTPUT FORMAT — return ONLY this JSON, nothing else:
{"facts": ["fact 1", "fact 2", ...]}

VALUE TRANSITIONS (critical rule):
When any value CHANGES — a credit limit, card type, co-holder, address — ALWAYS include
BOTH the old AND new value in the extracted fact.
  Pattern: "[attribute] [old value] dan [new value] ga o'zgartirildi"
  English: "[attribute] changed from [old] to [new]"
Examples:
  "Kredit limiti 10,000,000 dan 7,000,000 UZS ga kamaytirildi"
  "Humo Classic arizasi bekor qilindi, o'rniga Visa Gold ochildi"
  "Hammuallif Nilufar Visa Gold kartasiga qo'shildi, keyin olib tashlandi"
This is mandatory — the only way to answer "what was X before?" later.

INCLUDE:
- Card type (current and transitions), application status
- Credit limits (always with old and new value when changed)
- Co-holders / co-applicants: full name, added/removed events
- Income, billing address, contact preferences
- Service flags: SMS, autopay, foreign-currency, statement language
- Explicit assistant confirmations (approvals, changes, cancellations)

EXCLUDE:
- Greetings, filler phrases, day markers ("1-kun:", "Day 3:")
- General world facts not about this user
- Temporary questions without confirmed answers

Preserve the user's language (Uzbek, English, Russian) in every extracted fact.
""",
    # ── Update-memory prompt ───────────────────────────────────────────────────
    # When a numeric/entity value changes, store the transition text, not just
    # the new value, so historical queries can still be answered from Qdrant.
    "custom_update_memory_prompt": """\
You are a smart memory manager. Perform one of: ADD, UPDATE, DELETE, or NONE.

Compare each new retrieved fact against existing memories:
- ADD    : fact is genuinely new, no existing memory covers it
- UPDATE : fact conflicts with or refines an existing memory — keep the SAME ID
- DELETE : fact explicitly removes a previously stored entity
- NONE   : fact is already captured by an existing memory

TRANSITION RULE (mandatory):
When UPDATing a memory that holds a specific value (amount, card name, co-holder name,
address) and the new fact describes a change FROM the old value TO a new one, store the
FULL TRANSITION in the updated text. Never discard the old value.
  WRONG:   "Kredit limiti 7,000,000 UZS"
  CORRECT: "Kredit limiti 10,000,000 dan 7,000,000 UZS ga kamaytirildi"

Return ONLY this JSON:
{
  "memory": [
    {
      "id": "<existing or new ID>",
      "text": "<memory text>",
      "event": "ADD|UPDATE|DELETE|NONE",
      "old_memory": "<old text — required if event is UPDATE>"
    }
  ]
}
""",
}

# Pre-create the mem0 Qdrant collection at 2048 dims BEFORE Mem0 initializes.
# Wrapped in try/except so the server can start even when Qdrant is temporarily down.
try:
    _qc = QdrantClient(host="localhost", port=6333)
    _existing = {c.name: c for c in _qc.get_collections().collections}
    if "mem0" in _existing:
        _current_dim = _qc.get_collection("mem0").config.params.vectors.size
        if _current_dim != 2048:
            _qc.delete_collection("mem0")
            _qc.create_collection("mem0", vectors_config=VectorParams(size=2048, distance=Distance.COSINE))
    else:
        _qc.create_collection("mem0", vectors_config=VectorParams(size=2048, distance=Distance.COSINE))
    _qc.close()
    del _qc, _existing
except Exception as _e:
    print(f"[config] WARNING: Qdrant unavailable at startup ({_e}) — RAG/LTM will fail until Qdrant is up.")

try:
    ltm_memory = Memory.from_config(MEM0_CONFIG)
except Exception as _e:
    ltm_memory = None
    print(f"[config] WARNING: Mem0 LTM unavailable ({_e}) — long-term memory disabled until Qdrant is up.")

# --- Redis (session history) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SESSION_TTL = 86400          # 24 hours (seconds)
MAX_SESSION_MESSAGES = 40    # rolling window


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
RAG_COLLECTION = "knowledge_base"
EMBED_MODEL = "/models/embedding"
