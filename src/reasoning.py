#reasoning.py
from groq import Groq
import re
from src.utils import get_groq_client, extract_revenue, extract_quarters_from_query, quarter_order

client = get_groq_client()


# ------------------------------------------------
# STEP 4: Smart quarter filtering
# ------------------------------------------------
def filter_by_quarters(values, query):

    query_quarters = extract_quarters_from_query(query)

    # --------------------------------------------
    # CASE 1: User explicitly mentions 2 quarters
    # Example:
    # compare Q1 and Q3
    # --------------------------------------------
    if len(query_quarters) >= 2:

        return [
            v for v in values
            if v["quarter"] in query_quarters
        ]

    # --------------------------------------------
    # CASE 2: User mentions only 1 quarter
    # Example:
    # why Q3 higher
    # compare Q2 revenue
    # --------------------------------------------
    elif len(query_quarters) == 1:

        current_q = query_quarters[0]
        current_num = int(
            current_q.replace("Q", "")
        )

        prev_q = f"Q{current_num - 1}_2025" if current_num > 1 else None


        return [
            v for v in values
            if v["quarter"] in [prev_q, current_q]
        ]

    # --------------------------------------------
    # CASE 3: No quarter specified
    # Example:
    # why revenue increased
    # --------------------------------------------
    values = sorted(
        values,
        key=lambda x: quarter_order(x["quarter"])
    )

    return values[-2:]


# ------------------------------------------------
# MAIN REASONING FUNCTION
# ------------------------------------------------
def explain(query, chunks):
    
    if not chunks:
        return "No data found in documents."

    # Try to extract values
    values = extract_revenue(chunks)
    values = filter_by_quarters(values, query)

    # Deduplicate
    unique = {}
    for v in values:
        q = v["quarter"]
        if q not in unique or v["value"] > unique[q]["value"]:
            unique[q] = v
    values = list(unique.values())

    # Build context regardless
    context = "\n\n".join([
        f"[Quarter: {c['metadata']['quarter']} | "
        f"Source: {c['metadata']['source']}]\n{c['text']}"
        for c in chunks
    ])

    # Build numerical summary if available
    numerical_summary = ""
    if len(values) >= 2:
        values = sorted(
            values,
            key=lambda x: quarter_order(x["quarter"])
        )
        v1, v2 = values[0], values[-1]
        change = v2["value"] - v1["value"]
        percent = (change / v1["value"] * 100 
                  if v1["value"] != 0 else 0)
        trend = "increase" if change > 0 else "decrease"
        currency = v1.get("currency", "USD")
        unit = "Crore" if currency == "INR" else "Million"

        numerical_summary = f"""
Numerical Data Found:
- {v1['quarter']}: {v1['value']:,.0f} {unit}
- {v2['quarter']}: {v2['value']:,.0f} {unit}
- Change: {change:,.0f} {unit} ({percent:.2f}% {trend})
"""

    # Always call LLM regardless of numerical data
    prompt = f"""You are a senior financial analyst.

Question: {query}

{numerical_summary}

STRICT RULES:
- NEVER say "unable to determine"
- NEVER say "not enough data"
- Only explain using statements explicitly mentioned in the context.
- Do not infer business strategies unless directly stated.
- Use exact numbers if available
- If numbers not found, explain qualitatively from context
- NEVER invent initiatives, acquisitions, or projects
- Use ONLY facts explicitly present in context
- If a business reason is not mentioned, say "not explicitly stated in reports"
- Do NOT use external knowledge

Answer format:
**Direct Answer:** [one clear sentence]
**Supporting Numbers:** [figures from context]
**Key Reasons:** [business explanation]
**Sources:** [list sources]

Context:
{context}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.choices[0].message.content

    if numerical_summary:
        return f"{numerical_summary}\n{result}"
    return result