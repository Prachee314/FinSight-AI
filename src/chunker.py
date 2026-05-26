from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages):
    """
    Split extracted pages into smaller chunks
    while preserving metadata
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])

        for i, split in enumerate(splits):
            all_chunks.append({
                "text": split,
                "page": page["page"],
                "section": page["section"],
                "company": page["company"],
                "quarter": page["quarter"],
                "source": page["source"],
                "doc_type": page.get("doc_type", "report")
            })

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # test chunking
    from extract import extract_all_pdfs

    pages = extract_all_pdfs()
    chunks = chunk_pages(pages)

    print("\n🔍 Sample Chunks:\n")

    for c in chunks[:3]:
        print({
            "page": c["page"],
            "section": c["section"],
            "company": c["company"],
            "quarter": c["quarter"]
        })
        print(c["text"][:300])
        print("-" * 50)