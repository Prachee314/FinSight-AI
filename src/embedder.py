import re
import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

# Load embedding model (fast + good quality)
#model = SentenceTransformer("all-MiniLM-L6-v2")

#ADD this instead:
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection():
    """
    Create or load ChromaDB collection
    """
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("financial_docs")


def store_chunks(chunks):
    """
    Convert chunks → embeddings → store in DB
    """
    collection = get_collection()

    texts = [c["text"] for c in chunks]

    print("🔄 Creating embeddings...")
    embeddings = get_model().encode(texts).tolist()

    ids = [f"{c['source']}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
    metadatas = [{
        "company": c["company"],
        "quarter": c["quarter"],
        "section": c["section"],
        "source": c["source"],
        "page": str(c["page"]),
        "doc_type": c.get("doc_type", "report")
    } for c in chunks]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")


def search(
    query,
    quarter_filter=None,
    company_filter=None,
    top_k=5
):
    """
    Improved Financial Search:
    - metadata filtering
    - reranking
    - financial boosting
    - reasoning boosting
    - deduplication
    """

    collection = get_collection()

    # -----------------------------------
    # Encode query
    # -----------------------------------
    query_embedding = get_model().encode(
        [query]
    ).tolist()

    # -----------------------------------
    # Build metadata filters
    # -----------------------------------
    where = None

    if quarter_filter and company_filter:

        where = {
            "$and": [
                {"quarter": quarter_filter},
                {"company": company_filter}
            ]
        }

    elif quarter_filter:

        where = {
            "quarter": quarter_filter
        }

    elif company_filter:

        where = {
            "company": company_filter
        }

    # -----------------------------------
    # Retrieve more candidates
    # -----------------------------------
    try:

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 3,
            where=where
        )

    except Exception:

        # fallback search
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 8
        )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    filtered = []

    # -----------------------------------
    # Reranking / scoring
    # -----------------------------------
    for i in range(len(documents)):
        text = documents[i].lower()
        metadata = metadatas[i]
        score = 0

        if any(w in text for w in ["net sales", "revenue", "total revenue", "income"]):
            score += 3
            if re.search(r"[\d]{1,3}(?:,[\d]{3})+|[\d]{1,2}(?:,[\d]{2})+", text):  # Bug 4 fix too
                score += 10
                
        # ← ADD THIS — non-revenue metric boosting
        if any(w in text for w in [
            "operating margin", "margin at",
            "free cash flow", "cash flow"
        ]):
            score += 6

        if any(w in text for w in [
            "large deal", "tcv", "deal wins",
            "net new"
        ]):
            score += 5

        if any(w in text for w in [
            "earnings per share", "eps",
            "net profit", "diluted"
        ]):
            score += 4

        if any(w in text for w in [
            "headcount", "employees",
            "workforce"
        ]):
            score += 3

        if any(w in text for w in ["$", "million", "billion"]):  # must be at THIS indent level
            score += 2

        if any(w in text for w in ["expenses", "cost", "profit"]):
            score += 1

        # -------------------------------
        # Reasoning query boosting
        # -------------------------------
        if any(w in query.lower() for w in [
            "why",
            "reason",
            "cause",
            "increase",
            "decrease",
            "growth"
        ]):

            if any(w in text for w in [
                "management",
                "discussion",
                "growth",
                "decline",
                "increase",
                "decrease",
                "driven by"
            ]):
                score += 4

        # -------------------------------
        # Prefer useful sections
        # -------------------------------
        if metadata.get("section") == "revenue":
            score += 10

        elif metadata.get("section") == "mda":
            score += 5

        elif metadata.get("section") == "general":
            score += 1

        # -------------------------------
        # Skip useless chunks
        # -------------------------------
        if len(text) < 50:
            continue

        filtered.append({
            "text": documents[i],
            "metadata": metadata,
            "score": score
        })

    # -----------------------------------
    # Sort by score
    # -----------------------------------
    filtered = sorted(
        filtered,
        key=lambda x: x["score"],
        reverse=True
    )

    # -----------------------------------
    # Deduplicate
    # -----------------------------------
    seen = set()
    final = []

    for item in filtered:

        snippet = item["text"][:120]

        if snippet not in seen:

            seen.add(snippet)

            final.append({
                "text": item["text"],
                "metadata": item["metadata"]
            })

        if len(final) >= top_k:
            break

    return final
if __name__ == "__main__":
    # test pipeline
    from extract import extract_all_pdfs
    from chunker import chunk_pages

    print("🚀 Running full pipeline...")

    pages = extract_all_pdfs()
    chunks = chunk_pages(pages)

    store_chunks(chunks)

    print("\n🔍 Testing search...\n")

    # Better test queries
    results = search("What is Infosys total revenue Q4 2025?", top_k=3)
    # or
    results = search("Apple net sales Q1 2025", top_k=3)
    # or  
    results = search("Infosys revenue growth reason", top_k=3)
    for r in results:
        print(r["metadata"])
        print(r["text"][:300])
        print("-" * 50)