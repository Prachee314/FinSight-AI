# compare.py
from groq import Groq
import re
from src.utils import (
    get_groq_client,
    extract_revenue,
    extract_quarters_from_query,
    quarter_order
)

client = get_groq_client()


def filter_by_quarters(values, query):

    query_quarters = extract_quarters_from_query(query)

    # CASE 1: User specifies 2 quarters
    if len(query_quarters) >= 2:
        filtered = []
        for v in values:
            quarter_value = v["quarter"]
            for q in query_quarters:
                if quarter_value.startswith(q):
                    filtered.append(v)
                    break
        return filtered

    # CASE 2: User specifies 1 quarter
    elif len(query_quarters) == 1:
        current_q = query_quarters[0]
        current_num = int(re.search(r"Q([1-4])", current_q).group(1))  # Bug 2 fix
        prev_q = f"Q{current_num - 1}_2025" if current_num > 1 else None  # Bug 3 fix

        filtered = []
        for v in values:
            quarter_value = v["quarter"]
            if (
                quarter_value.startswith(current_q)
                or (prev_q and quarter_value.startswith(prev_q))
            ):
                filtered.append(v)
        return filtered

    # CASE 3: No quarter mentioned
    values = sorted(values, key=lambda x: quarter_order(x["quarter"]))
    return values[-2:]


def compare_quarters(query, chunks):
    
    values = extract_revenue(chunks)
    values = filter_by_quarters(values, query)

    # Deduplicate
    unique = {}
    for v in values:
        company = (v.get("usd") or v.get("inr") or {}).get("company", "")
        key = f"{v['quarter']}_{company}"
        unique[key] = v
    values = list(unique.values())

    if len(values) < 2:
        return "Not enough data to compare."

    # Sort Q1 → Q4
    values = sorted(values, key=lambda x: quarter_order(x["quarter"]))

    # Detect cross-company
    companies_in_data = list(set([
        (v.get("usd") or v.get("inr") or {}).get("company", "")
        for v in values
    ]))
    is_cross_company = len(companies_in_data) > 1

    # Build comparison text
    comparison_text = ""
    for v in values:
        usd = v.get("usd")
        inr = v.get("inr")
        company = (usd or inr or {}).get("company", "")
        company_label = f" ({company})" if is_cross_company else ""

        if usd and inr and not is_cross_company:
            # Same company — show both currencies
            comparison_text += (
                f"{v['quarter']}{company_label}: "
                f"${usd['value']:,.0f} Million"
                f" (₹{inr['value']:,.0f} Crore)\n"
            )
        elif usd:
            comparison_text += (
                f"{v['quarter']}{company_label}: "
                f"${usd['value']:,.0f} Million\n"
            )
        elif inr:
            comparison_text += (
                f"{v['quarter']}{company_label}: "
                f"₹{inr['value']:,.0f} Crore\n"
            )

    # Compute change using USD if available else INR
    v1, v2 = values[0], values[-1]
    val1 = v1["usd"]["value"] if v1.get("usd") else v1["inr"]["value"]
    val2 = v2["usd"]["value"] if v2.get("usd") else v2["inr"]["value"]
    currency1 = "USD" if v1.get("usd") else "INR"
    unit1 = "Million" if currency1 == "USD" else "Crore"

    change = val2 - val1
    percent = (change / val1) * 100 if val1 != 0 else 0
    
    # Get company names for each quarter
    company1 = (v1.get("usd") or v1.get("inr") or {}).get("company", "")
    company2 = (v2.get("usd") or v2.get("inr") or {}).get("company", "")
    is_cross_company = company1 != company2

    
    if is_cross_company:
        prompt = f"""
    You are a senior financial analyst.

    This is a CROSS-COMPANY comparison — two different companies, different quarters.
    DO NOT treat this as a time-series or growth analysis.

    Company 1: {company1.upper()} — {v1['quarter']}: {comparison_text.splitlines()[0]}
    Company 2: {company2.upper()} — {v2['quarter']}: {comparison_text.splitlines()[1]}

    Scale difference: {abs(change):,.0f} {unit1} ({abs(percent):.1f}x scale gap)
    
    Write:
    1. Scale Comparison — highlight size difference between companies
    2. Context — what this difference means (company size, market, industry)
    3. Conclusion — which is larger and by how much

    IMPORTANT:
    - Do NOT say revenue declined or increased — these are different companies
    - Do NOT calculate growth rates between them
    - Focus on absolute size difference
    """
    else:
        inr_note = f"Note: {company1.title()} figures shown in both USD and INR (Crore) where available." if company1.lower() == "infosys" else ""

        prompt = f"""
You are a senior financial analyst analyzing {company1.upper()} quarterly revenue.

Data:
{comparison_text}

Change from {v1['quarter']} to {v2['quarter']}:
{change:.2f} {unit1} ({percent:.2f}%)
{inr_note}
IMPORTANT:
- DO NOT recalculate percentages
- Use ONLY provided calculations
- Do NOT invent financial metrics
- Mention both USD and INR figures where available
- Use the exact change and percentage given

Write:
1. Key Differences
2. Trend Analysis
3. Conclusion

Keep answer concise and factual.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
