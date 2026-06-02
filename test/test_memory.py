"""
test_memory.py — Xotira qatlamini alohida tekshirish (smoke test).

5 ta soxta bank suhbatini to'g'ridan-to'g'ri quyidagi modullarga yuklaydi:
  memory/session.py  →  Redis qisqa muddatli xotira (STM)
  memory/ltm.py      →  Mem0 uzoq muddatli xotira (Qdrant + Neo4j graph)

Agent marshrutlash, xavfsizlik pipeline yoki LLM chati yo'q — faqat sof xotira qatlami tekshiruvi.
Ishga tushirish:  python test_memory.py
"""

import asyncio
import sqlite3
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.session import save_turn, get_session, get_active_buffer
from memory.ltm import upsert_ltm, search_ltm

# ── Soxta bank suhbati (Humo → Visa o'tishi) ─────────────────────────────────

MOCK_TURNS = [
    (
        "Men Humo karta ochmoqchiman",
        "Ajoyib! Humo Classic karta arizangizni boshlashim mumkin. "
        "Pasport va daromad tasdiqnomasi kerak bo'ladi.",
    ),
    (
        "Mening oylik daromadim 5,000,000 so'm",
        "5,000,000 so'mlik daromadingiz Humo Classic uchun mos keladi. "
        "Ariza boshlandi.",
    ),
    (
        "Humo arizamni bekor qiling — o'rniga Visa Gold karta istayapman",
        "Tushundim. Humo Classic arizangiz bekor qilindi. "
        "Visa Gold arizangizni boshlayman.",
    ),
    (
        "Visa Goldga xotinim Nilufrani hamkorchi sifatida qo'shing",
        "Nilufar Visa Gold arizangizga hamkorchi sifatida qo'shildi.",
    ),
    (
        "Hamkorchini bekor qiling — kartani faqat o'z nomimda qoldiring",
        "Hamkorchi Nilufar o'chirildi. Visa Gold faqat bitta egasi nomiga beriladi.",
    ),
]

SEP = "─" * 60


def _banner(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


async def run_test() -> None:
    user_id = f"test_{uuid.uuid4().hex[:8]}"
    ingest_start = datetime.now(timezone.utc)

    _banner("XOTIRA QATLAMI ALOHIDA TEKSHIRUVI")
    print(f"  Foydalanuvchi ID : {user_id}")
    print(f"  Boshlandi        : {ingest_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # ── 1. Redis (STM) + Mem0 (LTM) ga yuklash ───────────────────────────────
    _banner("1-QADAM — 5 TA SOXTA SUHBATNI YUKLASH")
    for i, (user_msg, asst_msg) in enumerate(MOCK_TURNS, 1):
        print(f"\n  Navbat {i}:")
        print(f"    [Foydalanuvchi] {user_msg}")
        print(f"    [Assistent    ] {asst_msg[:70]}...")
        await save_turn(user_id, user_msg, asst_msg)
        await upsert_ltm(user_msg, asst_msg, user_id)
        print(f"    ✓ Saqlandi → Redis + Qdrant/Mem0")

    # ── 2. Redis sessiyasini o'qib olish (STM) ───────────────────────────────
    _banner("2-QADAM — QISQA MUDDATLI XOTIRA (Redis)")
    session = await get_session(user_id)
    print(f"  Saqlangan navbatlar : {len(session) // 2}")
    print()
    for msg in session:
        role = "F" if msg["role"] == "user" else "A"
        print(f"  [{role}] {msg['content'][:85]}")

    active = await get_active_buffer(user_id)
    print(f"\n  Faol bufer (oxirgi 5 navbat) : {len(active) // 2} ta navbat")

    # ── 3. LTM (Qdrant) dan qidirish ─────────────────────────────────────────
    _banner("3-QADAM — UZOQ MUDDATLI XOTIRA (Qdrant / Mem0)")
    queries = [
        ("karta afzalligi",      ["visa", "humo"]),
        ("oylik daromad so'm",   ["5,000,000", "daromad"]),
        ("hamkorchi Nilufar",    ["nilufar", "o'chirildi", "sole"]),
        ("Humo bekor qilish",    ["humo", "bekor", "visa"]),
    ]

    for query_text, expected_kws in queries:
        print(f"\n  So'rov: \"{query_text}\"")
        facts = await search_ltm(query_text, user_id, limit=4)
        if facts:
            for line in facts.splitlines():
                marker = "✓" if any(kw in line.lower() for kw in expected_kws) else " "
                print(f"   {marker}  {line[:90]}")
        else:
            print("     (natija topilmadi)")

    # ── 4. SQLite o'zgarishlar jurnali ────────────────────────────────────────
    _banner("4-QADAM — SQLITE O'ZGARISHLAR JURNALI (mem0_history.db)")
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mem0_history.db"
    )
    if not os.path.exists(db_path):
        print("  mem0_history.db topilmadi — Mem0 tarixi o'chirilganmi?")
    else:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """
            SELECT event, old_memory, new_memory, created_at
            FROM   history
            WHERE  created_at >= ?
            ORDER  BY created_at ASC
            """,
            (ingest_start.strftime("%Y-%m-%dT%H:%M:%S"),),
        ).fetchall()
        conn.close()

        if rows:
            print(f"  Test boshlanganidan beri {len(rows)} ta hodisa qayd etildi:\n")
            for event, old_mem, new_mem, ts in rows:
                old_label = f"  ESKI : {(old_mem or '—')[:60]}" if old_mem else ""
                new_label = f"  YANGI: {(new_mem or '—')[:60]}"
                ts_short = ts[:19]
                print(f"  [{event:7s}] {ts_short}")
                if old_label:
                    print(f"           {old_label}")
                print(f"           {new_label}")
                print()
        else:
            print("  Hali hodisalar yo'q — LTM ajratish hali davom etmoqda.")
            print("  Maslahat: bir necha soniyadan keyin skriptni qayta ishga tushiring.")

    _banner("TEKSHIRUV YAKUNLANDI")
    print(f"  Foydalanuvchi ID: {user_id} — Qdrant 'mem0' kolleksiyasini tekshiring.")
    print()


if __name__ == "__main__":
    asyncio.run(run_test())
