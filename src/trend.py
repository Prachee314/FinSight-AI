# src/trend.py
from src.utils import get_groq_client, extract_revenue, quarter_order
from src.embedder import search

client = get_groq_client()

# Available quarters per company
COMPANY_QUARTERS = {
    "infosys": ["Q1_2025", "Q2_2025", "Q3_2025", "Q4_2025"],
    "apple":   ["Q1_2025", "Q2_2025"],
}


def get_trend(query, company, quarters=None):
    """
    Fetch all available quarters and build a trend table
    """
    # Use company-specific quarters instead of hardcoded all 4
    quarters = COMPANY_QUARTERS.get(company, ["Q1_2025", "Q2_2025", "Q3_2025", "Q4_2025"])

    all_chunks = []

    for q in quarters:
        press_chunks = search(
            f"delivered million revenues {company}",
            quarter_filter=q,
            company_filter=company,
            top_k=8
        )
        report_chunks = search(
            f"total revenue from operations crore {company}",
            quarter_filter=q,
            company_filter=company,
            top_k=8
        )
        all_chunks.extend(press_chunks)
        all_chunks.extend(report_chunks)

    # Remove duplicates — key on (quarter + first 120 chars) so chunks
    # from different quarters with similar opening text aren't merged
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        q_meta = c.get("metadata", {}).get("quarter", "")
        key = f"{q_meta}::{c['text'][:120]}"
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    values = extract_revenue(unique_chunks)

    if not values:
        return "No trend data found."

    # Deduplicate values per quarter (keep the largest/most confident extraction)
    by_quarter = {}
    for v in values:
        q = v["quarter"]
        if q not in by_quarter:
            by_quarter[q] = v
        else:
            existing_val = by_quarter[q]["usd"]["value"] if by_quarter[q].get("usd") else by_quarter[q]["inr"]["value"]
            new_val = v["usd"]["value"] if v.get("usd") else v["inr"]["value"]
            if new_val > existing_val:
                by_quarter[q] = v
    values = list(by_quarter.values())

    # Sort Q1 → Q4
    values = sorted(values, key=lambda x: quarter_order(x["quarter"]))

    # Build trend table
    trend_table = ""
    for v in values:
        usd = v.get("usd")
        inr = v.get("inr")
        if usd and inr:
            trend_table += (
                f"{v['quarter']}: "
                f"${usd['value']:,.0f}M"
                f" (₹{inr['value']:,.0f} Cr)\n"
            )
        elif usd:
            trend_table += f"{v['quarter']}: ${usd['value']:,.0f}M\n"
        elif inr:
            trend_table += f"{v['quarter']}: ₹{inr['value']:,.0f} Cr\n"

    # --------------------------------------------------------
    # GUARD: only one quarter of data available.
    # A "trend" needs at least 2 points — skip the LLM call
    # entirely rather than asking it to invent best/worst/growth
    # commentary on a single number (this caused the repeated
    # "Q1_2025: $94,036M" filler and meaningless best=worst text).
    # --------------------------------------------------------
    if len(values) < 2:
        only = values[0]
        label = (
            f"${only['usd']['value']:,.0f}M" if only.get("usd")
            else f"₹{only['inr']['value']:,.0f} Cr"
        )
        return (
            f"Revenue Trend — {company.upper()}\n\n"
            f"{trend_table}\n"
            f"Only one quarter of data is currently available "
            f"({only['quarter']}: {label}). A trend needs at least two "
            f"quarters to compare — please check back once more data is added."
        )

    # Compute overall growth
    v1 = values[0]
    v_last = values[-1]
    val1 = v1["usd"]["value"] if v1.get("usd") else v1["inr"]["value"]
    val_last = v_last["usd"]["value"] if v_last.get("usd") else v_last["inr"]["value"]
    unit = "Million" if v1.get("usd") else "Crore"
    overall_change = val_last - val1
    overall_pct = (overall_change / val1 * 100) if val1 != 0 else 0
    growth_summary = (
        f"Overall change from {v1['quarter']} to {v_last['quarter']}: "
        f"{overall_change:+,.0f} {unit} ({overall_pct:+.2f}%)"
    )

    # Best and worst quarters
    best = max(values, key=lambda x: x["usd"]["value"] if x.get("usd") else x["inr"]["value"])
    worst = min(values, key=lambda x: x["usd"]["value"] if x.get("usd") else x["inr"]["value"])

    best_label = (
        f"{best['quarter']}: ${best['usd']['value']:,.0f}M"
        if best.get("usd")
        else f"{best['quarter']}: ₹{best['inr']['value']:,.0f} Cr"
    )
    worst_label = (
        f"{worst['quarter']}: ${worst['usd']['value']:,.0f}M"
        if worst.get("usd")
        else f"{worst['quarter']}: ₹{worst['inr']['value']:,.0f} Cr"
    )

    # Clean prompt — no duplicates
    prompt = f"""You are a senior financial analyst.

Company: {company.upper()}

Revenue Trend:
{trend_table}
{growth_summary}

Pre-computed facts (use exactly, do not recalculate):
- Best quarter:  {best_label}
- Worst quarter: {worst_label}

STRICT RULES:
- Use ONLY the data provided above.
- DO NOT recalculate or second-guess pre-computed facts.
- DO NOT invent figures.
- DO NOT repeat the quarter-by-quarter figures already shown above — write commentary only, not a restated table.
- For Apple: USD only, no INR conversion.
- Be concise — 2-3 sentences total.
- Do NOT add parenthetical notes, caveats, or disclaimers.
- Do NOT mention source filenames.

Write a short paragraph covering:
- Overall growth conclusion (one sentence)
- Best and worst performing quarter (one sentence)
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = (
        f"Revenue Trend — {company.upper()}\n\n"
        f"{trend_table}\n"
        f"{growth_summary}\n\n"
        f"{response.choices[0].message.content}"
    )
    return answer, unique_chunks