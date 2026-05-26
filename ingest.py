from src.extract import extract_all_pdfs
from src.chunker import chunk_pages
from src.embedder import store_chunks


def ingest_documents():
    print("🚀 Starting ingestion...\n")

    # Step 1: Extract
    pages = extract_all_pdfs()

    # Step 2: Chunk
    chunks = chunk_pages(pages)

    # Step 3: Store embeddings
    store_chunks(chunks)

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    ingest_documents()