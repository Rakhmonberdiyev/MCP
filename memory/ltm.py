"""Mem0-backed long-term memory (Qdrant vectors + Neo4j graph)."""

import asyncio
import time
import config
from config import MEM0_CONFIG
import ui


def _get_ltm():
    """Return the live ltm_memory, attempting lazy init if it was None at startup.

    config.ltm_memory is None when Qdrant was down at import time. This function
    retries the initialization on every call until it succeeds, so the web backend
    recovers automatically once Qdrant comes up — no restart required.
    """
    if config.ltm_memory is None:
        try:
            from mem0 import Memory
            config.ltm_memory = Memory.from_config(MEM0_CONFIG)
            ui.ok("Mem0 LTM initialized (lazy — Qdrant is now reachable)")
        except Exception as exc:
            pass  # still unavailable — caller will skip gracefully
    return config.ltm_memory

# ── Dynamic top-K ──────────────────────────────────────────────────────────────

_GREETING_WORDS = {
    "hi", "hello", "hey", "ok", "okay", "thanks", "thank", "bye",
    "salom", "assalomu", "alaykum", "rahmat", "xayr", "yaxshimisiz",
    "привет", "пока", "спасибо",
}

def query_ltm_limit(query: str, deepthink: bool = False) -> int:
    """Return an appropriate LTM search limit based on query complexity."""
    if deepthink:
        return 10

    q = query.strip().lower()
    words = q.split()

    # Very short or pure greeting — skip LTM entirely
    if len(q) < 12 or (len(words) <= 3 and all(w in _GREETING_WORDS for w in words)):
        return 0

    # Complex/analytical intent
    complex_kw = (
        "all", "everything", "tell me", "summarize", "explain", "compare",
        "analyze", "why", "how", "what is", "history", "barcha", "hamma",
        "nima", "qanday", "nega", "tahlil", "tushuntirib", "batafsil",
        "всё", "всего", "объясни", "почему", "как",
    )
    if any(kw in q for kw in complex_kw):
        return 7

    return 4  # default


# ── Semantic de-duplication ────────────────────────────────────────────────────

def deduplicate_ltm_facts(ltm_facts: str, recent_messages: list[dict]) -> str:
    """Remove LTM facts whose core content is already present in recent messages."""
    if not ltm_facts or not recent_messages:
        return ltm_facts

    recent_text = " ".join(
        (m.get("content") or "") for m in recent_messages
        if m.get("role") in ("user", "assistant")
    ).lower()

    kept: list[str] = []
    for line in ltm_facts.splitlines():
        fact = line.lstrip("- ").strip().lower()
        if not fact:
            continue
        # Keep words longer than 4 chars as the "fingerprint"
        keywords = [w for w in fact.split() if len(w) > 4]
        if not keywords:
            kept.append(line)
            continue
        overlap = sum(1 for w in keywords if w in recent_text) / len(keywords)
        if overlap < 0.7:          # less than 70 % of key words already in context
            kept.append(line)

    return "\n".join(kept)

_QDRANT_COLLECTION = MEM0_CONFIG["vector_store"]["config"]["collection_name"]


_LTM_TIMEOUT = 8.0   # seconds — give up on Mem0 search if it hangs

def _patch_sub_stores() -> tuple[dict, dict, callable]:
    """
    Temporarily monkey-patch Qdrant vector_store.search and Neo4j graph.search
    so we can capture per-store timing and raw results without modifying mem0.

    Returns (timings_dict, raw_dict, restore_fn).
    The caller must call restore_fn() after the search completes.
    """
    timings: dict[str, float] = {}
    raw:     dict[str, object] = {}
    originals: list[tuple] = []   # (obj, attr, original_fn)

    mem = _get_ltm()
    vs = getattr(mem, "vector_store", None) if mem else None
    if vs and hasattr(vs, "search"):
        orig_vs = vs.search
        originals.append((vs, "search", orig_vs))
        def _vs_search(*a, **kw):
            t = time.perf_counter()
            r = orig_vs(*a, **kw)
            timings["qdrant"] = time.perf_counter() - t
            raw["qdrant"] = r
            return r
        vs.search = _vs_search

    gs = getattr(mem, "graph", None) if mem else None
    if gs and hasattr(gs, "search"):
        orig_gs = gs.search
        originals.append((gs, "search", orig_gs))
        def _gs_search(*a, **kw):
            t = time.perf_counter()
            r = orig_gs(*a, **kw)
            timings["neo4j"] = time.perf_counter() - t
            raw["neo4j"] = r
            return r
        gs.search = _gs_search

    def restore():
        for obj, attr, orig in originals:
            setattr(obj, attr, orig)

    return timings, raw, restore


async def search_ltm(query: str, user_id: str, limit: int = 4, min_score: float = 0.5) -> str:
    """Return relevant facts as a newline-separated string, or empty string."""
    if limit == 0:
        ui.kv("Mem0 LTM", "skipped (greeting / low-complexity query)")
        return ""
    mem = _get_ltm()
    if mem is None:
        ui.warn("Mem0 LTM not initialized (Qdrant unreachable) — skipping LTM search")
        return ""

    ui.console.print()
    ui.console.print(
        f"      [bold white]Mem0 Search[/bold white]"
        f"  [dim]Qdrant:[/dim] [cyan]{_QDRANT_COLLECTION}[/cyan]"
        f"  [dim]Graph:[/dim] [magenta]Neo4j[/magenta]"
    )
    ui.kv("Query →", query[:200])
    ui.kv("User",    user_id)
    ui.kv("Limit",   str(limit))

    timings, _raw, restore = _patch_sub_stores()

    t0 = time.perf_counter()
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(mem.search, query, user_id=user_id, limit=limit),
            timeout=_LTM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        restore()
        ui.warn(f"Mem0 search timed out (>{_LTM_TIMEOUT:.0f}s) — skipping LTM")
        ui.kv("LTM injected ≈", "0 tok — timeout")
        return ""
    except Exception as exc:
        restore()
        ui.warn(f"Mem0 search failed (non-fatal): {exc}")
        ui.kv("LTM injected ≈", "0 tok — search error, skipped")
        return ""
    elapsed = time.perf_counter() - t0
    restore()

    # ── Sub-store timings ──────────────────────────────────────────────────────
    if "qdrant" in timings:
        ui.timing("Qdrant vector search", timings["qdrant"])
    if "neo4j" in timings:
        ui.timing("Neo4j graph search",   timings["neo4j"])
    ui.timing("Mem0 total",               elapsed)

    _MIN_SCORE = min_score   # discard vector hits below this cosine similarity

    # Memory.search() returns {"results": [...vector...], "relations": [...graph...]}
    vector_raw  = (results or {}).get("results",   [])
    graph_raw   = (results or {}).get("relations", [])

    if not vector_raw and not graph_raw:
        ui.kv("Results", "none found")
        return ""

    # ── Vector: filter by score threshold ─────────────────────────────────────
    vector_kept    = [r for r in vector_raw if r.get("score", 0.0) >= _MIN_SCORE]
    vector_dropped = [r for r in vector_raw if r.get("score", 0.0) <  _MIN_SCORE]

    if vector_dropped:
        ui.console.print(f"      [dim]── Dropped (score < {_MIN_SCORE})[/dim]")
        for i, r in enumerate(vector_dropped):
            mem   = r.get("memory", "")
            score = r.get("score", 0.0)
            ui.console.print(
                f"        [dim]{i + 1}.  score=[yellow]{score:.3f}[/yellow]"
                f"  ≈{len(mem)//4} tok  {mem[:300]}[/dim]"
            )

    ui.kv("Results",
          f"{len(vector_raw)} vector ({len(vector_kept)} kept)"
          f"  +  {len(graph_raw)} graph")

    if vector_kept:
        ui.console.print(f"      [bold cyan]── Qdrant / {_QDRANT_COLLECTION}[/bold cyan]")
        for i, r in enumerate(vector_kept):
            mem   = r.get("memory", "")
            score = r.get("score", 0.0)
            ui.console.print(
                f"        [dim]{i + 1}.[/dim]"
                f"  score=[cyan]{score:.3f}[/cyan]"
                f"  ≈[cyan]{len(mem)//4}[/cyan] tok"
            )
            ui.console.print(f"          [dim]{mem[:300]}[/dim]")

    # ── Graph: {"source", "relationship", "destination"} triples ──────────────
    if graph_raw:
        ui.console.print("      [bold magenta]── Neo4j graph[/bold magenta]")
        for i, r in enumerate(graph_raw):
            src  = r.get("source", "")
            rel  = r.get("relationship", "")
            dst  = r.get("destination", "")
            text = f"{src} --[{rel}]--> {dst}"
            ui.console.print(
                f"        [dim]{i + 1}.[/dim]"
                f"  ≈[magenta]{len(text)//4}[/magenta] tok"
            )
            ui.console.print(f"          [dim]{text[:300]}[/dim]")

    if not vector_kept and not graph_raw:
        ui.kv("LTM injected ≈", "0 tok — all vector hits below score threshold")
        return ""

    lines: list[str] = []
    for r in vector_kept:
        lines.append(f"- {r['memory']}")
    for r in graph_raw:
        src = r.get("source", "")
        rel = r.get("relationship", "")
        dst = r.get("destination", "")
        lines.append(f"- {src} {rel} {dst}")

    total_tok = sum(len(l) for l in lines) // 4
    ui.kv("LTM injected ≈", f"{total_tok} tok")

    return "\n".join(lines)


async def upsert_ltm(user_input: str, assistant_output: str, user_id: str) -> None:
    """Extract and upsert facts from a conversation turn (runs in background)."""
    mem = _get_ltm()
    if mem is None:
        return
    messages = [
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": assistant_output},
    ]
    result = await asyncio.to_thread(mem.add, messages, user_id=user_id)

    # Persist ADD / UPDATE / DELETE events to PostgreSQL (non-fatal if DB is down)
    events = (result or {}).get("results", [])
    if events:
        from memory.db import write_events
        await write_events(user_id, events)
