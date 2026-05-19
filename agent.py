"""
agent.py — Main entry point (CLI + shared core for Telegram bot).

Full pipeline per turn:

  User Input
      │
      ├─ [Redis] load session history
      ├─ [Mem0]  search long-term memory          ← parallel
      │
      ▼
  Unified Context Ingestion
      │
      ▼
  Prompt Ingestion & Safety Guard
      │
      ├─ deepthink=False ──► System 1  (fast, direct tool-call loop)
      └─ deepthink=True  ──► System 2  (Strategy → Critique → Synthesis)
                                  │
      ┌───────────────────────────┘
      ▼
  Grounding & Hallucination Filter
      │
      ├─ Redis: Save Turn   (background)
      └─ Mem0:  Upsert Facts (background)
      │
      ▼
  Output Safety Guard → return response
"""

import asyncio
import re
import time
import config

from fastmcp import FastMCP

import ui
from memory.session import (
    get_active_buffer, get_older_history,
    get_session_summary, save_session_summary,
    get_current_session_meta,
    save_turn, get_user_docs,
)
from memory.ltm import search_ltm, upsert_ltm, query_ltm_limit, deduplicate_ltm_facts
from tools.mcp_server import search_mcp, rag_mcp
from tools.remote_mcp import (
    deposit_mcp, credit_mcp, pension_mcp,
    card_mcp, admin_mcp, realtime_mcp,
)
from pipeline.context_ingestion import build_messages
from pipeline import safety
from pipeline import system1, system2
from pipeline.output_processor import ground_and_filter

# ── Deepthink intent detection ────────────────────────────────────────────────
_DEEPTHINK_RE = re.compile(
    r"\b("
    # explicit deepthink keyword in any language
    r"deepthink|deep\s*think"
    # English research/analysis phrases
    r"|deep\s*research|deep\s*dive|research\s+this|research\s+thoroughly"
    r"|think\s+carefully|think\s+deeply|think\s+step.by.step"
    r"|reason\s+carefully|reason\s+through"
    r"|analyze\s+carefully|careful\s+analysis|thorough\s+analysis|detailed\s+analysis"
    r"|explain\s+in\s+detail|elaborate|in[\s-]depth"
    r")\b"
    # Uzbek phrases (no word boundary needed for Cyrillic/Latin mix)
    r"|yaxshilab|batafsil|chuqur\s*o['']yla|chuqur\s*tahlil|sinchiklab",
    re.IGNORECASE,
)

def _wants_deepthink(text: str) -> bool:
    return bool(_DEEPTHINK_RE.search(text))




# ── Shared MCP server ──────────────────────────────────────────────────────────
main_mcp = FastMCP("Main")
main_mcp.mount(search_mcp,   namespace="WebSearch")
main_mcp.mount(rag_mcp,      namespace="RAG")
main_mcp.mount(deposit_mcp,  namespace="Deposit")
main_mcp.mount(credit_mcp,   namespace="Credit")
main_mcp.mount(pension_mcp,  namespace="Pension")
main_mcp.mount(card_mcp,     namespace="Card")
main_mcp.mount(admin_mcp,    namespace="Admin")
main_mcp.mount(realtime_mcp, namespace="RealTime")


_FALLBACK_MODEL = "/models/gemma"

async def initialize() -> str:
    """Resolve model ID from the Xazna API. Falls back to hardcoded ID if API is down."""
    try:
        models = await config.llm_client.models.list()
        model_id = models.data[0].id
    except Exception as exc:
        model_id = _FALLBACK_MODEL
        ui.warn(f"Could not fetch model list ({exc}) — using fallback: {model_id}")
    config.MODEL_ID = model_id
    ui.console.print(
        f"\n[dim]Model:[/dim] [bold cyan]{model_id}[/bold cyan]"
        f"  [dim]│[/dim]  [bold green]Deepthink ON[/bold green] by default\n"
    )
    return model_id


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def process_turn(
    user_input: str,
    user_id: str,
    deepthink: bool = True,
    stream_callback=None,
    metadata: dict | None = None,
) -> str:
    """Process one user turn through the full pipeline. Returns assistant response.

    If metadata dict is provided it is populated in-place with:
      ltm_facts: list[str]   — facts retrieved from Mem0 LTM
      evidence:  str         — web/RAG evidence gathered by System 2 (if deepthink)
    """
    model = config.MODEL_ID

    t_overall = time.perf_counter()

    ui.blank()
    ui.user_panel(user_input)

    # ── 1. Unified Context Ingestion ────────────────────────────────────────────
    ui.section("Context Ingestion")
    ui.stage("Loading Redis session + Mem0 LTM + user docs in parallel…")

    _times: dict[str, float] = {}

    async def _timed(coro, label: str):
        t = time.perf_counter()
        result = await coro
        _times[label] = time.perf_counter() - t
        return result

    ltm_limit = query_ltm_limit(user_input, deepthink=deepthink)

    active_buffer, session_summary, ltm_facts, user_docs = await asyncio.gather(
        _timed(get_active_buffer(user_id),                        "Redis active buffer"),
        _timed(get_session_summary(user_id),                      "Redis session summary"),
        _timed(search_ltm(user_input, user_id, limit=ltm_limit),  "Mem0 LTM search"),
        _timed(get_user_docs(user_id),                            "User docs (Redis)"),
    )

    for label, elapsed in _times.items():
        ui.timing(label, elapsed)

    n_turns = len(active_buffer) // 2
    ui.kv("Active buffer",   f"{n_turns} turn{'s' if n_turns != 1 else ''} (last 5)")
    ui.kv("Session summary", f"{len(session_summary)} chars" if session_summary else "none yet")
    ui.session_dump(active_buffer[-6:])

    # Deduplicate: remove LTM facts already visible in the active buffer
    ltm_facts = deduplicate_ltm_facts(ltm_facts, active_buffer)

    ui.kv("Mem0 LTM facts", f"{len(ltm_facts.splitlines())} facts (limit={ltm_limit})" if ltm_facts else f"none (limit={ltm_limit})")
    ui.ltm_dump(ltm_facts)

    if metadata is not None:
        metadata["ltm_facts"] = [f.lstrip("- ") for f in ltm_facts.splitlines() if f.strip()]

    ui.kv("Uploaded docs",  ", ".join(user_docs) if user_docs else "none")

    messages = build_messages(
        user_input, active_buffer, ltm_facts,
        session_summary=session_summary,
        user_docs=user_docs,
    )
    ui.kv("Context window", f"{len(messages)} messages sent to LLM")

    # ── Token source breakdown (approximate: chars ÷ 4) ────────────────────────
    def _est(text: str) -> int:
        return max(0, len(text) // 4)

    sys_total_toks   = _est(messages[0]["content"])
    ltm_toks         = _est(ltm_facts) if ltm_facts else 0
    summary_toks     = _est(session_summary) if session_summary else 0
    buffer_toks      = sum(_est(str(m.get("content", ""))) for m in active_buffer)
    user_toks        = _est(user_input)
    ui.token_sources({
        "System prompt":    max(0, sys_total_toks - ltm_toks - summary_toks),
        "Session summary":  summary_toks,
        "Active buffer":    buffer_toks,
        "Mem0 LTM":         ltm_toks,
        "User input":       user_toks,
    })

    ui.section("Messages → LLM")
    ui.messages_dump(messages)

    # ── 2. Safety Guard (input) ─────────────────────────────────────────────────
    ui.section("Prompt Ingestion & Safety Guard")
    is_safe, reason = safety.check_input(user_input)
    if not is_safe:
        ui.err(f"Input BLOCKED — {reason}")
        return "I'm sorry, I can't process that request."
    ui.ok("Input passed safety check")

    # ── 3. Routing ──────────────────────────────────────────────────────────────
    ui.section("Routing")
    evidence = ""

    forced = not deepthink and _wants_deepthink(user_input)
    if forced:
        ui.warn("Fast mode override — user requested deep thinking → System 2")

    t_system = time.perf_counter()
    if deepthink or forced:
        ui.stage("System 2", "Deepthink ON  (Strategy → Critique Loop → Synthesis)")
        response, evidence = await system2.run(messages, main_mcp, model, stream_callback=stream_callback)
        ui.total_time("System 2 total", time.perf_counter() - t_system)
        if metadata is not None:
            metadata["evidence"] = evidence
    else:
        ui.stage("System 1", "Fast Mode  (direct tool-call loop)")
        response, _ = await system1.run(messages, main_mcp, model, stream_callback=stream_callback)
        ui.total_time("System 1 total", time.perf_counter() - t_system)
        if metadata is not None:
            metadata["evidence"] = ""

    # ── 4. Grounding & Hallucination Filter ────────────────────────────────────
    ui.section("Output Processing")
    ui.stage("Grounding & Hallucination Filter")
    t_ground = time.perf_counter()
    response = await ground_and_filter(response, evidence, model)
    ui.timing("Grounding filter", time.perf_counter() - t_ground)
    ui.ok("Grounding complete")
    ui.kv("🪙 Response tokens", f"≈{len(response)//4:,} tok  ({len(response):,} chars)")

    # ── 5. Output Safety Guard ──────────────────────────────────────────────────
    out_safe, out_reason = safety.check_output(response)
    if not out_safe:
        ui.err(f"Output BLOCKED — {out_reason}")
        response = safety.redact_unsafe_output(response)
    else:
        ui.ok("Output passed safety check")

    # ── 6. Persist in background (Redis + Mem0) ─────────────────────────────────
    ui.section("Persisting")
    ui.stage("Saving to Redis (session turn)")
    ui.save_redis(user_input, response)
    ui.stage("Saving to Mem0 (LTM upsert)")
    ui.save_mem0(user_input, response)
    asyncio.create_task(_persist(user_input, response, user_id, model))

    ui.blank()
    ui.response_panel(response)
    ui.total_time("Overall pipeline", time.perf_counter() - t_overall)
    ui.blank()

    return response


async def _generate_summary(user_id: str, model: str) -> None:
    """Summarize older turns and save to Redis. Runs in background every 5 turns."""
    older = await get_older_history(user_id)
    if not older:
        return
    # Format older history as plain text, truncating long assistant replies
    lines: list[str] = []
    for m in older:
        role    = "User" if m["role"] == "user" else "Assistant"
        content = (m.get("content") or "")[:300]
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    try:
        resp = await config.llm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation excerpt in 2–3 concise sentences. "
                        "Focus on what the user asked and what the assistant found or answered. "
                        "Be factual. No filler phrases."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        summary = (resp.choices[0].message.content or "").strip()
        if summary:
            await save_session_summary(user_id, summary)
            ui.console.print(f"[dim]  📝 Session summary updated ({len(summary)} chars)[/dim]")
    except Exception as e:
        ui.warn(f"Summary generation failed (non-fatal): {e}")


async def _persist(user_input: str, response: str, user_id: str, model: str) -> None:
    try:
        await asyncio.gather(
            save_turn(user_id, user_input, response),
            upsert_ltm(user_input, response, user_id),
        )
        ui.console.print("[dim]  💾 Redis ✓  Mem0 ✓[/dim]")
    except Exception as e:
        ui.warn(f"Persist error: {e}")
        return

    # Regenerate session summary every 5 turns once we have older history
    try:
        meta = await get_current_session_meta(user_id)
        turn_count = (meta or {}).get("message_count", 0)
        if turn_count >= 5 and turn_count % 5 == 0:
            await _generate_summary(user_id, model)
    except Exception as e:
        ui.warn(f"Summary trigger failed (non-fatal): {e}")


# ── CLI loop ───────────────────────────────────────────────────────────────────

async def main() -> None:
    from rich.panel import Panel
    ui.console.print(Panel(
        "[bold]AI Agent[/bold] — architecture: Context → Safety → System1/System2 → Grounding → Output\n"
        "Commands: [cyan]/deepthink on[/cyan] | [cyan]/deepthink off[/cyan] | [cyan]exit[/cyan]",
        border_style="cyan",
        padding=(0, 2),
    ))

    await initialize()

    USER_ID   = "user_001"
    deepthink = False

    while True:
        try:
            raw = await asyncio.to_thread(input, "\n[You] → ")
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[dim]Goodbye.[/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue
        if raw.lower() == "exit":
            ui.console.print("[dim]Goodbye.[/dim]")
            break
        if raw.lower() == "/deepthink off":
            deepthink = False
            ui.warn("Deepthink OFF — using System 1 (fast)")
            continue
        if raw.lower() == "/deepthink on":
            deepthink = True
            ui.ok("Deepthink ON — using System 2")
            continue

        try:
            await process_turn(raw, USER_ID, deepthink=deepthink)
        except Exception as e:
            ui.err(f"Pipeline error: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
