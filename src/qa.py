from src.utils import get_groq_client

client = get_groq_client()


def answer(query, chunks):

    if not chunks:
        return "No relevant data found."

    context = "\n\n".join([
        f"[Source: {c['metadata']['source']} | "
        f"Quarter: {c['metadata']['quarter']} | "
        f"Section: {c['metadata']['section']}]\n{c['text']}"
        for c in chunks
    ])
    prompt = f"""
You are a financial analyst assistant.

Rules:
- Use only given context
- Always include source and quarter in answer
- Prefer numerical values when available
- Be concise and logical
- Report TOTAL revenue only — do NOT add product and service segments together
- Use the single total net sales figure directly from the financial statements

Context:
{context}

Question: {query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content