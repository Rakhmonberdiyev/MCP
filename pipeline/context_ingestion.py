"""Unified context ingestion: merges session history + LTM facts into messages."""

from datetime import datetime

def _system_prompt() -> str:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    return f"""\
You are a helpful AI assistant for Xazna bank with long-term memory and access to live bank data tools.
Current date: {date_str}  |  Current time: {time_str} (UTC+5 Tashkent)

## Tools you have — USE THEM, do NOT answer from general knowledge when a tool exists

### Bank data tools (ALWAYS call these for bank-related questions — never guess)
- **Pension_***: Pension payment schedules by region/district/street.
  → Call when user asks about pension payment dates, schedules, mahalla, neighborhood payment days.
  → Use Pension_get_payment_region_district_street with the exact region, district, and street name.
- **Deposit_***: Deposit products — names, details, rates, terms, currency, minimum amounts.
  → Call when user asks about deposits, saving accounts, foiz (interest), muddatli (term).
- **Credit_***: Credit/loan products — purpose, amount, term, rate, initial contribution.
  → Call when user asks about kredit, ssuda, loans, installment, qarz.
- **Card_***: Bank card products — names, details, payment systems (Visa/MC/Humo/Uzcard), currency.
  → Call when user asks about cards, plastic, kartochka.
- **Admin_***: General bank info, branch list, admin contacts.
  → Call when user asks about the bank itself, branches, contact info.
- **RealTime_***: Current time, date, currency codes, exchange rates.
  → Call when user asks about the current exchange rate, valyuta kursi, or current time.

### Knowledge tools
- **RAG_rag_search**: Search documents the user has uploaded (PDFs, CVs, reports).
  → Call when user asks about a file they sent or references an uploaded document.
- **WebSearch_web_search**: Search the internet for information not covered by the tools above.
  → Use as a last resort when no bank tool applies and the answer is not in your training data.

## Rules
1. For ANY question about pension, deposit, credit, card, or exchange rate → call the matching tool FIRST. Never answer from general knowledge.
2. If the tool returns no result or an error → then fall back to web_search.
3. For uploaded documents → always call RAG_rag_search first.
4. Combine tool results with your reasoning to give accurate, grounded answers.\
"""


def build_messages(
    user_input: str,
    active_buffer: list[dict],
    ltm_facts: str,
    session_summary: str = "",
    user_docs: list[str] | None = None,
) -> list[dict]:
    """
    Returns the full messages array ready to send to the LLM:
      [system] → [active buffer (last 5 turns)] → [user]

    Context injected into system message:
      - Current date/time
      - Uploaded document list
      - Long-term memory facts (Qdrant + Neo4j, deduplicated)
      - Rolling summary of older turns (replaces sending all 20 messages)
    """
    sys_content = _system_prompt()

    if user_docs:
        doc_list = "\n".join(f"  - {d}" for d in user_docs)
        sys_content += (
            f"\n\n[Documents uploaded by this user — searchable via rag_search, most recent first]:\n"
            f"{doc_list}\n"
            f"When the user says 'this file', 'this document', 'last pdf', or similar, "
            f"they mean the most recent one: '{user_docs[0]}'."
        )

    if ltm_facts:
        sys_content += f"\n\n[Long-term memory about this user]:\n{ltm_facts}"

    if session_summary:
        sys_content += f"\n\n[Summary of earlier conversation turns]:\n{session_summary}"

    messages: list[dict] = [{"role": "system", "content": sys_content}]
    messages.extend(active_buffer)
    messages.append({"role": "user", "content": user_input})
    return messages
