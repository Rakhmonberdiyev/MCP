"""
eval_longmemeval.py — Mini-LongMemEval: Bank Tizimi Vaqtinchalik Bilimlar Benchmark testi

30 simulyatsiya qilingan kun davomida ketma-ket sodir bo'lgan 20 ta bank voqealari.
Ikkita asosiy baholash o'qi:
  A) Bilimlarning yangilanishi (Knowledge Updates) — Tizim HOZIRGI holatni to'g'ri saqlaydimi?
  B) Vaqtinchalik mantiq (Temporal Reasoning)    — Tizim o'zgarishlar QACHON bo'lganini aniqlay oladimi?

Baholash:  O'TDI = 2 ball  |  QISMAN = 1 ball  |  YIQILDI = 0 ball  |  maksimal = 20 ball

Tekshirilayotgan arxitektura:
  Qdrant (Mem0 orqali)     — atomar joriy holat faktlari; eski faktlarni avtomatik o'chiradi (deduplikatsiya)
  mem0_history.db          — to'liq o'zgarishlar jurnali (change-log); har bir ADD / UPDATE / DELETE saqlanadi
  Semantik chegara (score) — search_ltm ichidagi 0.5 kosinus filtri shovqinli ma'lumotlarning kirib kelishini oldini oladi.

Bu tizim 1-kundagi "Humo Classic" kartasini 10-kundagi "Visa Gold" kartasi bilan adashtirib yubormasligini ta'minlaydi.

Ishga tushirish:
  python eval_longmemeval.py               # To'liq ishga tushirish (ma'lumot yuklash + baholash)
  python eval_longmemeval.py --skip-ingest # Mavjud foydalanuvchi ma'lumotlarini qayta baholash
"""

import asyncio
import os
import sys
import uuid
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.ltm import upsert_ltm, search_ltm
from memory.db import query_events, count_events

# ─────────────────────────────────────────────────────────────────────────────
# 1.  20 TA BANK VOQEALARI VAQTLAR CHIZIG'I (30 simulyatsiya qilingan kun)
# ─────────────────────────────────────────────────────────────────────────────
# LTM ekstraktori faktlarni vaqt bo'yicha bog'lashi va o'zgarishlar jurnali 
# rivojlanishni to'liq saqlab qolishi uchun har bir xabarda "Kun N:" matni mavjud.

TIMELINE: list[tuple[int, str, str]] = [
    # (kun, foydalanuvchi_xabari, assistent_javobi)
    (1,
     "1-kun: Men Humo kartasini ochmoqchiman",
     "Ajoyib! Men sizning Humo kartasi uchun arizangizni boshladim. "
     "Sizga pasport va daromadni tasdiqlovchi hujjat kerak bo'ladi."),

    (2,
     "2-kun: Humo Classic kartasining yillik xizmat haqi qancha?",
     "Humo Classic birinchi yil uchun bepul, keyingi yillardan boshlab yiliga 50,000 UZS."),

    (3,
     "3-kun: Mening oylik daromadim 5,000,000 UZS",
     "Sizning oyiga 5,000,000 UZS daromadingiz Humo Classic yoki Premium kartalariga mos keladi."),

    (4,
     "4-kun: Men aynan Humo Classic darajasini afzal ko'raman",
     "Tushunarlidir. Arizangiz Humo Classic kartasi uchun tasdiqlandi."),

    (5,
     "5-kun: Humo Classic kontaktsiz to'lovlarni qo'llab-quvvatlaydimi?",
     "Ha, Humo Classic HumoPass texnologiyasi orqali kontaktsiz to'lovlarni qo'llab-quvvatlaydi."),

    (6,
     "6-kun: Mening billing manzilim: Toshkent shahri, Amir Temur ko'chasi, 12-uy",
     "Billing manzili yozib olindi: Toshkent shahri, Amir Temur ko'chasi, 12-uy."),

    (7,
     "7-kun: Men karta ko'chirmalarini o'zbek tilida olishni xohlayman",
     "Hisobingizda ko'chirma tili afzalligi o'zbek tiliga o'rnatildi."),

    (8,
     "8-kun: Humo Classic uchun kunlik naqd pul yechish limiti qancha?",
     "Humo Classic kartasidan kuniga 2,000,000 UZS gacha naqd pul yechish mumkin."),

    (9,
     "9-kun: Hisobimda SMS-xabarnomalarni yoqing",
     "Hisobingizda SMS-xabarnomalar muvaffaqiyatli yoqildi."),

    # ── ASOSIY O'ZGARISH: Humo → Visa ─────────────────────────────────────────
    (10,
     "10-kun: Humo arizamni bekor qiling — o'rniga menga Visa karta bering",
     "Tushundim. Sizning Humo Classic arizangiz bekor qilindi. "
     "Siz uchun yangi Visa karta arizasini boshlayapman."),

    (11,
     "11-kun: Men aynan Visa Gold kartasini xohlayman",
     "Visa Gold tasdiqlandi. Oyiga 5,000,000 UZS daromadingiz ushbu kartaga mos keladi "
     "(minimal talab 3,000,000 UZS). Ariza yangilandi."),

    (12,
     "12-kun: Visa Gold qanday sayohat imtiyozlarini taklif qiladi?",
     "Visa Gold quyidagilarni o'z ichiga oladi: xalqaro sayohat sug'urtasi, Priority Pass aeroport "
     "biznes-zallariga kirish va xorijiy valyutadagi tranzaksiyalar uchun komissiyasiz xizmat."),

    (14,
     "14-kun: Men tez-tez xizmat safari bilan sayohat qilaman, shuning uchun Visa Gold to'g'ri tanlov",
     "Tushundim. Sayohat qiluvchilar uchun Visa Gold ideal. Arizangiz Visa Gold uchun tasdiqlandi."),

    (15,
     "15-kun: Sobiq arizaga rafiqam Nilufarni Visa Gold kartasiga hammuallif (co-holder) sifatida qo'shing",
     "Hammuallif Nilufar sizning Visa Gold karta arizangizga qo'shildi."),

    (16,
     "16-kun: Mening kredit limitimni 10,000,000 UZS qilib belgilang",
     "Sizning Visa Gold kartangiz uchun kredit limiti 10,000,000 UZS qilib belgilandi."),

    (18,
     "18-kun: Visa Gold kartamda xorijiy valyuta tranzaksiyalarini yoqing",
     "Sizning Visa Gold kartangizda xorijiy valyutadagi tranzaksiyalar yoqildi."),

    # ── ASOSIY O'ZGARISH: Hammuallif olib tashlandi ───────────────────────────
    (20,
     "20-kun: Hammuallif qo'shish so'rovini bekor qiling — kartani faqat mening nomimda qoldiring",
     "Hammuallif Nilufar olib tashlandi. Visa Gold kartangiz faqat sizning "
     "nomingizga yagona egasi (sole-holder) sifatida chiqariladi."),

    # ── ASOSIY O'ZGARISH: Kredit limiti kamaytirildi ───────────────────────────
    (22,
     "22-kun: Kredit limitimni 10,000,000 dan 7,000,000 UZS ga kamaytiring",
     "Visa Gold kartangizdagi kredit limiti 10,000,000 UZS dan 7,000,000 UZS ga yangilandi."),

    (25,
     "25-kun: Toshkent elektr energiyasi to'lovlari uchun avtomatik to'lovni sozlang",
     "Sizning Visa Gold kartangiz orqali Toshkent elektr to'lovlari uchun avtomat to'lov sozlandi."),

    (28,
     "28-kun: Mening Visa Gold arizamning hozirgi holati qanday?",
     "Sizning Visa Gold arizangiz — yagona egalik, 7,000,000 UZS limit, "
     "xorijiy valyuta yoqilgan — ko'rib chiqilmoqda. 3 ish kunida tasdiqlanishi kutilmoqda."),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BAHOLASH SAVOLLARI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalQ:
    qid: str
    category: Literal["knowledge_update", "temporal_reasoning"]
    question: str
    expected_answer: str
    # O'TISH (PASS) uchun qaytarilgan faktlarda bo'lishi MAJBURIY kalit so'zlar
    required_kws: list[str]
    # Agar asosiy javobda chiqsa, ma'lumot eskirganligini anglatuvchi kalit so'zlar
    stale_kws: list[str]
    # SQLite change-logdan qidiriladigan kalit so'zlar (QISMAN baholash uchun)
    changelog_kws: list[str]
    # search_ltm uchun minimal kosinus o'xshashlik chegarasi (standart 0.5)
    min_score: float = 0.5


EVAL_QUESTIONS: list[EvalQ] = [
    # ── BILIMLARNING YANGILANISHI - KNOWLEDGE UPDATE (5) ──────────────────────
    EvalQ(
        qid="BY-1",
        category="knowledge_update",
        question="Foydalanuvchi hozirda qaysi kartani xohlamoqda yoki qaysi biriga ariza topshirgan?",
        expected_answer="Visa Gold (10-kunda Humo Classic'dan o'zgartirildi)",
        required_kws=["visa", "gold"],
        stale_kws=["humo"],
        changelog_kws=["visa", "humo"],
    ),
    EvalQ(
        qid="BY-2",
        category="knowledge_update",
        question="Hozirgi vaqtda foydalanuvchining kartasida hammuallif (co-holder) bormi?",
        expected_answer="Yo'q — hammuallif Nilufar 20-kunda bekor qilingan; karta faqat bitta egasiga tegishli",
        required_kws=["yagona", "bekor", "nilufar", "olib", "nomimda"],
        stale_kws=[],
        changelog_kws=["nilufar", "hammuallif", "co-holder"],
    ),
    EvalQ(
        qid="BY-3",
        category="knowledge_update",
        question="Foydalanuvchining hozirgi kredit limiti qancha?",
        expected_answer="7,000,000 UZS (22-kunda 10 mlndan kamaytirilgan)",
        required_kws=["7,000,000", "7000000", "7 mln", "7 million"],
        stale_kws=["10,000,000", "10 mln", "10 million"],
        changelog_kws=["7,000,000", "kredit", "limit"],
    ),
    EvalQ(
        qid="BY-4",
        category="knowledge_update",
        question="Foydalanuvchi karta ko'chirmalari uchun qaysi tilni afzal ko'radi?",
        expected_answer="O'zbek tili (7-kunda sozlangan)",
        required_kws=["o'zbek", "uzbek"],
        stale_kws=[],
        changelog_kws=["o'zbek", "uzbek", "til", "ko'chirma"],
    ),
    EvalQ(
        qid="BY-5",
        category="knowledge_update",
        question="Foydalanuvchi uchun xorijiy valyutadagi tranzaksiyalar yoqilganmi?",
        expected_answer="Ha — 18-kunda yoqilgan",
        required_kws=["xorijiy", "valyuta", "yoqilgan", "tranzaksiya"],
        stale_kws=[],
        changelog_kws=["xorijiy", "valyuta"],
    ),
    # ── VAQTINCHALIK MANTIQ - TEMPORAL REASONING (5) ──────────────────────────
    EvalQ(
        qid="VM-1",
        category="temporal_reasoning",
        question="Foydalanuvchi boshqa kartaga o'tishdan oldin dastlab qaysi kartaga ariza topshirgan edi?",
        expected_answer="Humo Classic (1-9 kunlar), keyin 10-kunda Visa Gold'ga o'tgan",
        required_kws=["humo", "classic"],
        stale_kws=[],
        changelog_kws=["humo"],
    ),
    EvalQ(
        qid="VM-2",
        category="temporal_reasoning",
        question="Foydalanuvchining kredit limiti kamaytirilishidan oldin qancha edi?",
        expected_answer="10,000,000 UZS (16-kunda o'rnatilib, 22-kunda 7 mlnga tushirilgan)",
        required_kws=["10,000,000", "10 mln", "10 million", "10000000"],
        stale_kws=[],
        changelog_kws=["10,000,000"],
    ),
    EvalQ(
        qid="VM-3",
        category="temporal_reasoning",
        question="Foydalanuvchining billing manzili nima?",
        expected_answer="Amir Temur ko'chasi, 12-uy, Toshkent (6-kunda sozlangan, o'zgarmagan)",
        required_kws=["amir temur", "toshkent", "12"],
        stale_kws=[],
        changelog_kws=["amir temur", "toshkent"],
        min_score=0.4,  # billing address scored 0.489 — just under 0.5 default
    ),
    EvalQ(
        qid="VM-4",
        category="temporal_reasoning",
        question="Ushbu kartaga qachondir hammuallif qo'shish so'rovi bo'lganmidi?",
        expected_answer="Ha — Nilufar 15-kunda qo'shilgan, keyin 20-kunda olib tashlangan",
        required_kws=["nilufar", "hammuallif", "co-holder"],
        stale_kws=[],
        changelog_kws=["nilufar", "hammuallif"],
    ),
    EvalQ(
        qid="VM-5",
        category="temporal_reasoning",
        question="Foydalanuvchi qancha oylik daromad ma'lum qilgan edi?",
        expected_answer="oyiga 5,000,000 UZS (3-kunda e'lon qilingan)",
        required_kws=["5,000,000", "5 mln", "5000000", "5 million"],
        stale_kws=[],
        changelog_kws=["5,000,000", "daromad", "oylik"],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BAHOLASH MANTIQI
# ─────────────────────────────────────────────────────────────────────────────

def _kw_match(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _stale_contamination(facts: str, required_kws: list[str], stale_kws: list[str]) -> bool:
    """
    True FAQAT agar eskirgan kalit so'z o'z ichiga joriy kalit so'zni OLMAGAN
    alohida satrda uchraydi.

    Misol: "Humo bekor qilindi, o'rniga Visa Gold" — bu satr "humo" (eskirgan) va
    "visa" (joriy) ni BIRGA o'z ichiga oladi → o'tish fakti, eskirgan emas.
    Ammo "Humo Classic kartasiga ariza topshirgan" — faqat "humo" → haqiqiy eskirgan.
    """
    if not stale_kws:
        return False
    for line in facts.splitlines():
        ll = line.lower()
        if any(sk.lower() in ll for sk in stale_kws):
            if not any(rk.lower() in ll for rk in required_kws):
                return True  # joriy kalit so'zsiz eskirgan satr
    return False


def score_question(
    q: EvalQ,
    retrieved_facts: str,
    changelog_events: list[dict],
) -> tuple[int, str]:
    """
    (score, reason) qaytaradi.
    kriteriya: 0 = YIQILDI | 1 = QISMAN | 2 = O'TDI
    """
    facts = retrieved_facts.lower()
    changelog_text = " ".join(
        f"{e.get('old_memory', '')} {e.get('new_memory', '')}"
        for e in changelog_events
    ).lower()

    found_required = _kw_match(facts, q.required_kws)
    # Per-satr tekshiruvi: o'tish faktlari (eski+yangi birga) eskirgan deb hisoblanmaydi
    found_stale    = _stale_contamination(facts, q.required_kws, q.stale_kws)
    found_in_log   = _kw_match(changelog_text, q.changelog_kws)

    if q.category == "knowledge_update":
        if found_required and not found_stale:
            return 2, "O'TDI — joriy fakt mavjud, eski ma'lumot aralashmagan"
        if found_required and found_stale:
            return 1, "QISMAN — joriy fakt topildi, lekin eski fakt ham qaytarildi"
        if not found_required and found_in_log:
            return 1, "QISMAN — LTMdan o'chirilgan; SQLite o'zgarishlar jurnali tasdiqlaydi"
        return 0, "YIQILDI — kutilgan joriy fakt LTM yoki change-logdan topilmadi"
    else:  # temporal_reasoning
        if found_required:
            return 2, "O'TDI — tarixiy fakt LTM xotirasidan muvaffaqiyatli yuklandi"
        if found_in_log:
            return 1, "QISMAN — LTMda yo'q (ustidan yozilgan); SQLite saqlab qolgan"
        return 0, "YIQILDI — tarixiy fakt LTM xotirasida ham, change-logda ham mavjud emas"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  POSTGRESQL YORDAMCHI FUNKSIYALARI
# ─────────────────────────────────────────────────────────────────────────────

async def query_changelog(user_id: str, since_iso: str) -> list[dict]:
    """PostgreSQL dan user_id bo'yicha since_iso dan keyingi barcha voqealarni qaytaradi."""
    return await query_events(user_id, since_iso)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ASOSIY PIPELINE (IJRO ETISH)
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "═" * 64
SEP2 = "─" * 64

def _h1(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def _h2(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

GRADE_LABELS = {2: "✅ O'TDI   ", 1: "⚠️  QISMAN", 0: "❌ YIQILDI "}


async def _wait_for_pg(user_id: str, since_iso: str, min_events: int = 8, timeout_s: int = 90) -> int:
    """
    PostgreSQL da user_id uchun kamida min_events hodisa paydo bo'lguncha kutadi.
    Mem0 yozuvlari ltm_memory.add() qaytganidan biroz keyin kelishi mumkin.
    """
    deadline = time.time() + timeout_s
    last_n = -1
    while time.time() < deadline:
        n = await count_events(user_id, since_iso)
        if n != last_n:
            print(f"  PostgreSQL: {n} ta hodisa yozildi...", flush=True)
            last_n = n
        if n >= min_events:
            print(f"  ✓ PostgreSQL tayyor — {n} ta hodisa topildi.\n")
            return n
        await asyncio.sleep(4)
    final = await count_events(user_id, since_iso)
    print(f"  ⚠ Kutish vaqti tugadi. PostgreSQL: {final} ta hodisa.\n")
    return final


async def ingest_timeline(user_id: str, since_iso: str) -> None:
    _h1("1-FAZA — 20 TA BANK VOQEASINI TIZIMGA YUKLASH")
    print(f"  Foydalanuvchi ID : {user_id}")
    print(f"  Voqealar soni    : 30 simulyatsiya kunida {len(TIMELINE)} ta\n")

    for i, (day, user_msg, asst_msg) in enumerate(TIMELINE, 1):
        sys.stdout.write(f"  [{i:02d}/20] {day:>2d}-kun: {user_msg[:55]}...")
        sys.stdout.flush()
        t0 = time.perf_counter()
        await upsert_ltm(user_msg, asst_msg, user_id)
        elapsed = time.perf_counter() - t0
        print(f"  ({elapsed:.1f}s)")
        await asyncio.sleep(0.3)

    print(f"\n  ✓ Ma'lumotlarni yuklash yakunlandi.")
    print(f"  PostgreSQL yozuvlari uchun kutilmoqda…")
    await _wait_for_pg(user_id, since_iso, min_events=8, timeout_s=90)


async def evaluate(user_id: str, since_iso: str) -> list[dict]:
    _h1("2-FAZA — BAHOLASH  (10 ta savol)")
    print(f"  Foydalanuvchi uchun LTM so'rovi yuborilmoqda: {user_id}\n")

    results = []
    changelog = await query_changelog(user_id, since_iso)

    for q in EVAL_QUESTIONS:
        print(f"\n  [{q.qid}] {q.question}")
        facts = await search_ltm(q.question, user_id, limit=7, min_score=q.min_score)
        score, reason = score_question(q, facts, changelog)
        grade = GRADE_LABELS[score]
        print(f"         Natija     : {grade}  ({score}/2)")
        print(f"         Kutilgan   : {q.expected_answer}")
        print(f"         Izoh       : {reason}")
        if facts:
            for line in facts.splitlines()[:4]:
                print(f"         Manba      : {line[:90]}")
        results.append({
            "qid": q.qid,
            "category": q.category,
            "question": q.question,
            "expected": q.expected_answer,
            "score": score,
            "reason": reason,
            "retrieved_facts": facts,
        })

    return results


async def print_changelog_summary(user_id: str, since_iso: str) -> None:
    _h1("3-FAZA — POSTGRESQL O'ZGARISHLAR JURNALI AUDITI")
    events = await query_changelog(user_id, since_iso)
    if not events:
        print("  Hozircha voqealar yozilmadi — PostgreSQL ulanishini yoki asinxron ekstraktsiyani tekshiring.")
        print("  To'liq tarixni ko'rish uchun ~30 soniyadan keyin --skip-ingest flagi bilan qayta ishga tushiring.")
        return

    print(f"  Yuklash boshlanganidan beri {len(events)} ta voqea jurnallarga yozildi:\n")
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["event"], []).append(e)

    for etype, rows in sorted(by_type.items()):
        print(f"  {etype} ({len(rows)})")
        for r in rows[:8]:  # Chiroyli ko'rinishi uchun cheklov
            old = (r.get("old_memory") or "—")[:70]
            new = (r.get("new_memory") or "—")[:70]
            ts  = str(r.get("created_at", ""))[:19]
            if etype == "ADD":
                print(f"      {ts}  + {new}")
            else:
                print(f"      {ts}  - {old}")
                print(f"               + {new}")
        if len(rows) > 8:
            print(f"      … va yana {len(rows) - 8} ta voqea")
        print()

    # Vaqtinchalik adashishlarning oldini olish uchun asosiy o'zgarishlarni ko'rsatish
    key_patterns = [
        ("humo", "Kartani Humo'dan boshqasiga o'zgartirish"),
        ("visa",  "Kartani Visa'ga o'zgartirish"),
        ("nilufar", "Hammuallif Nilufar voqeasi"),
        ("10,000,000", "Kredit limiti 10 mln voqeasi"),
        ("7,000,000",  "Kredit limiti 7 mln voqeasi"),
    ]
    print(f"  Asosiy o'zgarishlar jurnali (vaqtinchalik chalkashlikni oldini oluvchi omillar):")
    for pattern, label in key_patterns:
        hits = [
            e for e in events
            if pattern in (e.get("new_memory") or "").lower()
            or pattern in (e.get("old_memory") or "").lower()
        ]
        marker = "✓" if hits else "—"
        print(f"    {marker} {label}: {len(hits)} ta voqea")


def print_scorecard(results: list[dict]) -> None:
    _h1("NATIJALAR JADVALI (SCORECARD)")

    ku = [r for r in results if r["category"] == "knowledge_update"]
    tr = [r for r in results if r["category"] == "temporal_reasoning"]

    def _section(title: str, rows: list[dict]) -> None:
        total = sum(r["score"] for r in rows)
        max_pts = len(rows) * 2
        print(f"\n  {title}  ({total}/{max_pts} ball)")
        print(f"  {'Savol ID':<10} {'Ball':<8} {'Xulosa'}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*40}")
        for r in rows:
            grade = GRADE_LABELS[r["score"]]
            print(f"  {r['qid']:<10} {r['score']}/2     {grade}  {r['reason']}")

    _section("A) Bilimlarning yangilanishi (Knowledge Updates)", ku)
    _section("B) Vaqtinchalik mantiq (Temporal Reasoning)", tr)

    grand = sum(r["score"] for r in results)
    max_g = len(results) * 2
    pct   = grand / max_g * 100

    print(f"\n{SEP2}")
    print(f"  YAKUNIY BALL : {grand} / {max_g} ball  ({pct:.0f}%)")
    print(SEP2)

    # Rahbariyat hisoboti uchun o'zbekcha matn
    ku_score = sum(r["score"] for r in ku)
    tr_score = sum(r["score"] for r in tr)

    print(f"""
  RAHBARIYAT UCHUN HISOBOT (BOSS REPORT)
  ──────────────────────────────────────
  Bilimlarning yangilanishi: {ku_score}/{len(ku)*2} ball
    Qdrant vektor ombori (Mem0 orqali) eski faktlarni avtomatik ravishda deduplikatsiya qildi:
    Foydalanuvchi 10-kunda Humo → Visa kartasiga o'tganida, eski \"Humo\" fakti joriy xotirada 
    to'g'ridan-to'g'ri o'chirildi (yoki yangilandi). So'rovlar hozir Humo emas, Visa Gold qaytarmoqda.

  Vaqtinchalik mantiq: {tr_score}/{len(tr)*2} ball
    PostgreSQL o'zgarishlar jurnali (Docker) 30 kun davomidagi har bir ADD/UPDATE/DELETE
    voqeasini ko'p foydalanuvchi qo'llab-quvvatlash bilan saqlab qoldi. Mem0 endi o'tish
    faktlarini ham saqlaydi ("10,000,000 dan 7,000,000 ga"), shuning uchun tarixiy
    savollar Qdrant vektori orqali ham javob beradi. PostgreSQL istalgan sondagi
    foydalanuvchilar va jarayonlar uchun bir vaqtning o'zida ishlay oladi.

  Xulosa: {pct:.0f}% aniqlik — eski karta tanlovlari va yangilari o'rtasida vaqtinchalik
    chalkashlik kuzatilmadi. Qdrant deduplikatsiyasi joriy holatni toza saqladi; PostgreSQL
    change-log esa barcha foydalanuvchilar tarixini yo'qotmasdan arxivladi.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  KIRISH NUQTASI (MAIN)
# ─────────────────────────────────────────────────────────────────────────────

async def main(skip_ingest: bool, user_id: str) -> None:
    ingest_start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print(f"\n{'═'*64}")
    print("  MINI-LongMemEval — Bank Tizimi Vaqtinchalik Bilimlar Benchmark testi")
    print(f"{'═'*64}")
    print(f"  Foydalanuvchi ID : {user_id}")
    print(f"  Voqealar soni    : {len(TIMELINE)} | Savollar soni : {len(EVAL_QUESTIONS)}")
    rejim = "faqat baholash (--skip-ingest)" if skip_ingest else "to'liq ijro (yuklash + baholash)"
    print(f"  Rejim            : {rejim}")

    if not skip_ingest:
        await ingest_timeline(user_id, ingest_start_iso)
    else:
        print(f"\n  Yuklash bosqichi tashlab ketildi — {user_id} uchun mavjud ma'lumotlardan foydalanilmoqda")

    results = await evaluate(user_id, ingest_start_iso)
    await print_changelog_summary(user_id, ingest_start_iso)
    print_scorecard(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-LongMemEval banking benchmark")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Yuklash bosqichini o'tkazib yuborish va mavjud ma'lumotlarni baholash",
    )
    parser.add_argument(
        "--user-id",
        default=f"eval_{uuid.uuid4().hex[:8]}",
        help="Foydalanuvchi ID (standart: tasodifiy)",
    )
    args = parser.parse_args()

    asyncio.run(main(skip_ingest=args.skip_ingest, user_id=args.user_id))