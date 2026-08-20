import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

print("=== Collections in chroma_db ===")
for c in chroma_client.list_collections():
    print(f"  - {c.name}")

col = chroma_client.get_collection("work_orders", embedding_function=embedding_fn)
print(f"\n=== 'work_orders' collection count: {col.count()} ===")

print("\n=== All document IDs in the collection ===")
all_docs = col.get()
print(all_docs["ids"])

query = "What was done for WO-1001?"
print(f"\n=== Querying: '{query}' (n_results=2) ===")
results = col.query(query_texts=[query], n_results=2)

for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"\n--- Result {i+1} (distance: {dist:.4f}) ---")
    print(doc[:300])