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
- Report TOTAL revenue only.
- If both USD and INR revenue values are available, include BOTH values.
- For Infosys revenue questions, report revenue in both USD (Million) and INR (Crore).
- For Apple revenue questions, report revenue only in USD.
- Prefer exact financial figures from the context.
- Use only the given context below.
- Do NOT mention source filenames, document names, or section labels in your answer.
- Do NOT write phrases like "Source:", "Quarter:", or "Section:" in your answer text.
- Prefer numerical values when available.
- Be concise and logical — answer in 1-3 sentences unless the question needs more detail.
- Report TOTAL revenue only — do NOT add product and service segments together.
- Use the single total net sales figure directly from the financial statements.
- If the answer isn't in the context, say so plainly without guessing.

Context:
{context}

Question: {query}

Answer (plain text only, no source/quarter/section labels):
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content