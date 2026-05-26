'''import chromadb

# Connect to DB
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Load collection
collection = client.get_or_create_collection(
    "financial_docs"
)

# Get sample records
results = collection.get(limit=10)

# Print results
for i in range(len(results["ids"])):

    print("\n======================")

    print("TEXT:")
    print(results["documents"][i][:200])

    print("\nMETADATA:")
    print(results["metadatas"][i])'''
# save as test_infosys.py in your project root
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("financial_docs")

# Get ALL infosys chunks
results = collection.get(
    where={"company": "infosys"},
    limit=50
)

print(f"Total infosys chunks: {len(results['ids'])}")
print("\n=== SEARCHING FOR REVENUE NUMBERS ===")

for i, doc in enumerate(results['documents']):
    if "crore" in doc.lower():
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Text: {doc[:300]}")
        print("---")