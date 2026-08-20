import json
import chromadb
from chromadb.utils import embedding_functions

print(" LOADING WORK ORDERS INTO CHROMADB")


with open("work_orders.json") as f:
    work_orders = json.load(f)

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="work_orders",
    embedding_function=embedding_fn
)

documents, metadatas, ids = [], [], []

for wo in work_orders:
    history_text = "\n".join(
        [f"  [{h['date']}] {h['action']} (by {h['by']})"
         for h in wo["history"]]
    )

    doc_text = f"""
Work Order ID : {wo['work_order_id']}
Title         : {wo['title']}
Status        : {wo['status']}
Priority      : {wo['priority']}
Created At    : {wo['created_at']}
Assigned To   : {wo['assigned_to']}
History:
{history_text}
    """.strip()

    documents.append(doc_text)
    metadatas.append({
        "work_order_id": wo["work_order_id"],
        "status":        wo["status"],
        "priority":      wo["priority"],
        "assigned_to":   wo["assigned_to"]
    })
    ids.append(wo["work_order_id"])

collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

print(f"\n Total records ingested : {collection.count()}")
print(f" Database saved at  : ./chroma_db")
print("\n Ingestion complete!")
