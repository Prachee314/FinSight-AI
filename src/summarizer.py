from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def summarize(query, chunks):
    if not chunks:
        return "No data found."

    context = "\n\n".join([
        f"[Quarter: {c['metadata']['quarter']} | "
        f"Section: {c['metadata']['section']} | "
        f"Source: {c['metadata']['source']}]\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""You are a senior financial analyst.

Provide a structured summary based on the context.

Format:
**Key Highlights:**
- [bullet points]

**Financial Performance:**
- [revenue, profit, expenses if available]

**Important Points:**
- [risks, management comments, outlook]

**Quarter:** [mention which quarter]
**Source:** [mention source files]

Context:
{context}

Query: {query}

Summary:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content