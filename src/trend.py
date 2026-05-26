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

    # Remove duplicates
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        key = c["text"][:120]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    values = extract_revenue(unique_chunks)

    if not values:
        return "No trend data found."

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

    # Compute overall growth
    growth_summary = ""
    if len(values) >= 2:
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
- Use ONLY the data provided above
- DO NOT recalculate or second-guess pre-computed facts
- DO NOT invent figures
- For Apple: USD only, no INR conversion
- Be concise — one line per quarter, no extra explanations
- Do not add parenthetical notes or caveats

Write:
1. Quarter-by-quarter trend (use exact figures)
2. Best and worst performing quarter
3. Overall growth conclusion
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return (
        f"Revenue Trend — {company.upper()}\n\n"
        f"{trend_table}\n"
        f"{growth_summary}\n\n"
        f"{response.choices[0].message.content}"
    )