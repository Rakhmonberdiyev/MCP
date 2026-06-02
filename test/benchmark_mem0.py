"""
benchmark_mem0.py — Comprehensive Mem0 Agent Memory Rules Benchmark

Mirrors benchmark_letta.py's section/rule structure, adapted to this project's
dual-tier memory architecture:

  TIER 1 — LTM  (Long-Term Memory: Qdrant vectors + Neo4j graph via Mem0)
    Analogous to Letta Core Memory — always accessible across sessions.
    Rule 1.1  New user fact → LTM upsert stores it in Qdrant after the turn
    Rule 1.2  LTM fact visible in a FRESH session (STM empty, only LTM can answer)
    Rule 1.3  Fact change → LTM upsert emits UPDATE, new value stored
    Rule 1.4  Updated LTM fact visible in a fresh session

  TIER 2 — STM  (Short-Term Memory: Redis session history)
    Analogous to Letta Recall Memory — recent N turns verbatim, older turns summarized.
    Rule 2.1  Code planted in Redis session
    Rule 2.2  Immediate recall — code turn is in the active buffer (last 5 turns)
    Rules 2.3–2.9  Noise turns push code turn out of active buffer → rolling summary
    Rule 2.10 Distant recall — code retrievable from rolling session summary

  TIER 3 — Direct LTM Seed & Retrieval
    Analogous to Letta Archival Memory — data seeded before any conversation.
    Rule 3.1  ltm_memory.add() seeds a fact directly into Qdrant/Neo4j
    Rule 3.2  Fresh session: agent retrieves pre-seeded fact via LTM search
              (fact is NOT in any STM — only in the vector store)

Run:
  python test/benchmark_mem0.py
"""

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import process_turn, initialize
import config
from memory.ltm import search_ltm
from memory.session import (
    create_session,
    get_active_buffer,
    get_session_summary,
    ACTIVE_TURNS,
    SUMMARY_OVERFLOW_TURNS,
)

# ── Config ─────────────────────────────────────────────────────────────────────
BENCH_USER    = f"bench_{uuid.uuid4().hex[:8]}"  # isolated user for this run
PERSIST_WAIT  = 4.0   # seconds to wait for upsert_ltm background task
SUMMARY_WAIT  = 15.0  # extra seconds to wait for _generate_summary (LLM call inside _persist)
LTM_MIN_SCORE = 0.3   # relaxed threshold for benchmark verification searches

# Short noise prompt — we need turns to overflow the buffer, not token count.
# ACTIVE_TURNS=5, so after 7 noise turns the planted code is outside the window.
_NOISE = (
    "Write a short Python function (10 lines) that reverses a list "
    "using a loop. This is unrelated to any previous topic."
)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn: int
    section: str
    prompt: str
    response_text: str = ""
    total_latency_s: float = 0.0
    stm_active_turns: int = 0           # active-buffer depth when evaluated

    # Checks
    memory_write_pass: Optional[bool] = None   # LTM stored / summary generated
    answer_pass:       Optional[bool] = None   # response contains expected string

    expected_in_response: Optional[str] = None
    expected_in_ltm:      Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _send(user_id: str, prompt: str) -> tuple[str, float]:
    """Send one pipeline turn; return (response_text, latency_s)."""
    t0       = time.perf_counter()
    response = await process_turn(user_input=prompt, user_id=user_id, deepthink=False)
    return response, time.perf_counter() - t0


async def _wait_persist() -> None:
    """Yield to the event loop long enough for background _persist tasks to finish."""
    await asyncio.sleep(PERSIST_WAIT)


async def _wait_summary() -> None:
    """
    Wait for the full _persist chain including _generate_summary (an LLM call).
    _persist runs: save_turn + upsert_ltm (parallel, ~3-5s) → then _generate_summary
    (another LLM call, ~2-4s). Total can exceed 4s so we use a longer timeout here.
    """
    await asyncio.sleep(SUMMARY_WAIT)


async def _check_ltm(user_id: str, query: str, expected: str) -> tuple[bool, str]:
    """Return (found, raw_text) — searches LTM directly for verification."""
    raw   = await search_ltm(query, user_id, limit=10, min_score=LTM_MIN_SCORE)
    found = expected.lower() in raw.lower()
    return found, raw


def _icon(v: Optional[bool]) -> str:
    if v is True:  return "PASS"
    if v is False: return "FAIL"
    return "N/A "


def _print_turn(r: TurnResult) -> None:
    print(f"  [{r.turn:2d}] {r.section}")
    if r.expected_in_ltm is not None:
        print(f"        LTM check    : [{_icon(r.memory_write_pass)}]"
              f"  looking for '{r.expected_in_ltm}' in Qdrant")
    if r.memory_write_pass is not None and r.expected_in_ltm is None:
        print(f"        Memory check : [{_icon(r.memory_write_pass)}]")
    if r.expected_in_response is not None:
        snippet = r.response_text[:120].replace("\n", " ")
        print(f"        Answer check : [{_icon(r.answer_pass)}]"
              f"  looking for '{r.expected_in_response}'")
        print(f"        Response     : {snippet}…")
    if r.stm_active_turns:
        print(f"        STM buffer   : {r.stm_active_turns} active turns (window={ACTIVE_TURNS})")
    print(f"        Latency      : {r.total_latency_s:.2f}s")
    print()


def _evaluate(r: TurnResult) -> None:
    if r.expected_in_response is not None:
        r.answer_pass = r.expected_in_response.lower() in r.response_text.lower()


# ── Section 1: LTM Persistence ─────────────────────────────────────────────────

async def _ltm_tests(user_id: str, start: int) -> list[TurnResult]:
    results: list[TurnResult] = []
    turn = start

    # 1.1 — New fact → upsert_ltm should store it in Qdrant
    text, lat = await _send(user_id,
        "My name is Temur and I am a senior backend engineer at a fintech company. "
        "Please remember this about me.")
    await _wait_persist()
    ltm_found, _ = await _check_ltm(user_id, "user name profession", "Temur")
    r = TurnResult(
        turn=turn, section="1.1  LTM Write — new user fact stored in Qdrant",
        prompt="My name is Temur…",
        response_text=text, total_latency_s=lat,
        expected_in_ltm="Temur", memory_write_pass=ltm_found,
    )
    _evaluate(r); _print_turn(r); results.append(r); turn += 1

    # 1.2 — Fresh session: STM is empty — only LTM can answer
    await create_session(user_id, "LTM persistence check")
    text, lat = await _send(user_id, "What do you know about me? Tell me everything.")
    r = TurnResult(
        turn=turn, section="1.2  LTM Persistence — fact visible in fresh session (no STM)",
        prompt="What do you know about me?",
        response_text=text, total_latency_s=lat,
        expected_in_response="Temur",
    )
    _evaluate(r); _print_turn(r); results.append(r); turn += 1

    # 1.3 — Fact changes → upsert_ltm should UPDATE the name
    text, lat = await _send(user_id,
        "I changed my name — from now on call me Bobur, not Temur. "
        "Please update your memory to replace my old name.")
    await _wait_persist()
    ltm_found, _ = await _check_ltm(user_id, "user name", "Bobur")
    r = TurnResult(
        turn=turn, section="1.3  LTM Update — fact change stored (Temur → Bobur)",
        prompt="My name changed to Bobur…",
        response_text=text, total_latency_s=lat,
        expected_in_ltm="Bobur", memory_write_pass=ltm_found,
    )
    _evaluate(r); _print_turn(r); results.append(r); turn += 1

    # 1.4 — Fresh session: updated name visible, old name should be gone
    await create_session(user_id, "LTM update verify")
    text, lat = await _send(user_id, "What is my name?")
    r = TurnResult(
        turn=turn, section="1.4  LTM Update Verify — updated name in fresh session",
        prompt="What is my name?",
        response_text=text, total_latency_s=lat,
        expected_in_response="Bobur",
    )
    _evaluate(r); _print_turn(r); results.append(r)

    return results


# ── Section 2: STM Session History ─────────────────────────────────────────────

async def _stm_tests(user_id: str, start: int) -> list[TurnResult]:
    """
    Plant a code, confirm immediate recall from active buffer, then flood with
    7 noise turns so the planted turn overflows the active buffer into older_history.
    SUMMARY_OVERFLOW_TURNS=4 triggers rolling summarization.
    Then verify the code survives in the session summary.

    Buffer math (ACTIVE_TURNS=5, SUMMARY_OVERFLOW_TURNS=4):
      After turn 1 (plant) + 5 noise turns → code turn is in older_history (1 turn).
      After 4 turns in older_history without summarization → summary triggered.
      By noise turn 7 (turn index 8 in this session) → summary should exist.
    """
    results: list[TurnResult] = []
    turn = start

    # Fresh session — clean STM slate for Section 2
    await create_session(user_id, "STM benchmark")

    # 2.1 — Plant the code
    text, lat = await _send(user_id,
        "IMPORTANT: Remember this secret system code — RECALL_CODE=VX-7734. "
        "You will need it later.")
    buf = await get_active_buffer(user_id)
    r = TurnResult(
        turn=turn, section="2.1  STM Plant — code stored in Redis session",
        prompt="RECALL_CODE=VX-7734",
        response_text=text, total_latency_s=lat,
        stm_active_turns=len(buf) // 2,
    )
    _evaluate(r); _print_turn(r); results.append(r); turn += 1

    # 2.2 — Immediate recall: code turn is inside the active buffer
    text, lat = await _send(user_id, "What was the RECALL_CODE I just told you?")
    buf = await get_active_buffer(user_id)
    r = TurnResult(
        turn=turn, section="2.2  STM Immediate — code in active buffer, no LTM needed",
        prompt="What was the RECALL_CODE?",
        response_text=text, total_latency_s=lat,
        expected_in_response="VX-7734",
        stm_active_turns=len(buf) // 2,
    )
    _evaluate(r); _print_turn(r); results.append(r); turn += 1

    # 2.3–2.9 — 7 noise turns:
    #   Each turn pushes one more old turn out of the active window.
    #   When 4 turns accumulate in older_history without a summary, _persist triggers
    #   _generate_summary and the code turn's content is folded into the rolling summary.
    for i in range(7):
        text, lat = await _send(user_id, _NOISE)
        buf = await get_active_buffer(user_id)
        r = TurnResult(
            turn=turn, section=f"2.{3 + i}  STM Flood #{i + 1} — noise turn",
            prompt="(noise)",
            response_text=text, total_latency_s=lat,
            stm_active_turns=len(buf) // 2,
        )
        _evaluate(r); _print_turn(r); results.append(r); turn += 1

    # 2.10 — Distant recall: code is outside active buffer → must come from session summary
    await _wait_summary()  # _generate_summary is an LLM call inside _persist — needs extra time
    summary = await get_session_summary(user_id)
    buf     = await get_active_buffer(user_id)

    print(f"        [pre-2.10] Summary exists : {'yes' if summary else 'NO'}"
          f"  ({len(summary)} chars)")
    if summary:
        print(f"        [pre-2.10] Summary snippet : {summary[:120]}…")
    print()

    text, lat = await _send(user_id,
        "I need that RECALL_CODE from the beginning of our conversation. "
        "What was it exactly?")
    # memory_write_pass: checks the overflow/summary mechanism triggered correctly
    summary_generated = bool(summary)
    r = TurnResult(
        turn=turn, section="2.10 STM Summary — distant code recalled from rolling summary",
        prompt="Find RECALL_CODE from early conversation",
        response_text=text, total_latency_s=lat,
        expected_in_response="VX-7734",
        memory_write_pass=summary_generated,
        stm_active_turns=len(buf) // 2,
    )
    _evaluate(r); _print_turn(r); results.append(r)

    return results


# ── Section 3: Direct LTM Seed & Retrieval ─────────────────────────────────────

async def _direct_ltm_tests(start: int) -> list[TurnResult]:
    """
    Uses a dedicated seed_user so the Project Nexus fact exists ONLY in LTM
    (Qdrant/Neo4j) and never in any session's STM.

    3.1: ltm_memory.add() inserts fact directly — no conversation involved.
    3.2: Fresh session for seed_user → agent must use LTM search to answer.
    """
    results: list[TurnResult] = []
    turn = start

    seed_user = f"bench_seed_{uuid.uuid4().hex[:6]}"

    # 3.1 — Direct LTM seed (bypasses process_turn / STM entirely)
    seed_payload = (
        "Project Nexus — pre-seeded archival record: "
        "Launch date July 15, 2026. Budget $500K. Lead: Kamila Yusupova. "
        "Status: approved. Priority: critical."
    )
    t0       = time.perf_counter()
    seed_ok  = False
    if config.ltm_memory is not None:
        try:
            await asyncio.to_thread(
                config.ltm_memory.add,
                [{"role": "user", "content": seed_payload}],
                user_id=seed_user,
            )
            seed_ok = True
            print(f"  [seed] Project Nexus inserted into Qdrant/Neo4j via ltm_memory.add().")
        except Exception as e:
            print(f"  [seed] LTM seed failed: {e}")
    else:
        print("  [seed] SKIPPED — ltm_memory not initialized (Qdrant down).")
    seed_lat = time.perf_counter() - t0

    ltm_found, _ = await _check_ltm(seed_user, "Project Nexus launch budget", "Nexus")
    r = TurnResult(
        turn=turn, section="3.1  Direct LTM Seed — ltm_memory.add() inserts into Qdrant",
        prompt="(direct API seed — no conversation)",
        total_latency_s=seed_lat,
        expected_in_ltm="Nexus", memory_write_pass=ltm_found and seed_ok,
    )
    _evaluate(r); _print_turn(r); results.append(r); turn += 1

    # 3.2 — Fresh session: Project Nexus is ONLY in LTM, not in any STM
    print()
    await create_session(seed_user, "LTM seed retrieval test")
    text, lat = await _send(seed_user,
        "What do you have stored about Project Nexus? "
        "Search your memory and tell me all details.")
    r = TurnResult(
        turn=turn, section="3.2  LTM Search — fresh session, data only in Qdrant/Neo4j",
        prompt="Search memory for Project Nexus",
        response_text=text, total_latency_s=lat,
        expected_in_response="Nexus",
    )
    _evaluate(r); _print_turn(r); results.append(r)

    return results


# ── Scoring ────────────────────────────────────────────────────────────────────

def _report(all_results: list[TurnResult]) -> None:
    write_checks  = [r for r in all_results if r.memory_write_pass is not None]
    answer_checks = [r for r in all_results if r.answer_pass        is not None]
    write_passed  = sum(1 for r in write_checks  if r.memory_write_pass)
    answer_passed = sum(1 for r in answer_checks if r.answer_pass)
    total_latency = sum(r.total_latency_s for r in all_results)
    avg_latency   = total_latency / len(all_results) if all_results else 0.0

    print("=" * 66)
    print("SCORECARD")
    print("=" * 66)
    print(f"  Memory-write correctness : {write_passed}/{len(write_checks)}")
    print(f"  Answer correctness       : {answer_passed}/{len(answer_checks)}")
    print(f"  Total latency            : {total_latency:.2f}s  "
          f"(avg {avg_latency:.2f}s / turn, {len(all_results)} turns)")
    print()

    print("Per-rule results:")
    for r in all_results:
        w = f"[{_icon(r.memory_write_pass)}]" if r.memory_write_pass is not None else "      "
        a = f"[{_icon(r.answer_pass)}]"       if r.answer_pass        is not None else "      "
        print(f"  Turn {r.turn:2d}  {r.section:<56s}  "
              f"write={w}  answer={a}  {r.total_latency_s:.2f}s")
    print()

    total_checks = len(write_checks) + len(answer_checks)
    total_passed = write_passed + answer_passed
    pct = 100 * total_passed // total_checks if total_checks else 0
    print(f"Overall: {total_passed}/{total_checks} checks passed ({pct}%)")
    if pct == 100:
        print("All memory rules working correctly.")
    elif pct >= 75:
        print("Most rules working — review FAIL rows above.")
    else:
        print("Multiple rules failing — check Qdrant, Neo4j, Redis, and model config.")


# ── Entry point ────────────────────────────────────────────────────────────────

async def run_benchmark() -> None:
    print("Mem0 Agent Memory Rules Benchmark")
    print("Architecture : Redis STM (active buffer + rolling summary)")
    print("             + Mem0 LTM (Qdrant vectors + Neo4j graph)")
    print(f"STM config   : ACTIVE_TURNS={ACTIVE_TURNS}  SUMMARY_OVERFLOW_TURNS={SUMMARY_OVERFLOW_TURNS}")
    print(f"Sections     : LTM Persistence | STM Session History | Direct LTM Seed")
    print(f"Bench user   : {BENCH_USER}")
    print()

    await initialize()

    all_results: list[TurnResult] = []

    print("━" * 66)
    print("SECTION 1 — LTM Persistence  (Qdrant/Neo4j via Mem0)")
    print("━" * 66)
    all_results.extend(await _ltm_tests(BENCH_USER, start=1))

    print("━" * 66)
    print("SECTION 2 — STM Session History  (Redis active buffer + rolling summary)")
    print("━" * 66)
    all_results.extend(await _stm_tests(BENCH_USER, start=len(all_results) + 1))

    print("━" * 66)
    print("SECTION 3 — Direct LTM Seed & Retrieval  (ltm_memory.add + search_ltm)")
    print("━" * 66)
    all_results.extend(await _direct_ltm_tests(start=len(all_results) + 1))

    print("━" * 66)
    _report(all_results)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
