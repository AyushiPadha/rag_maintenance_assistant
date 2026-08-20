import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path=".env", override=True)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def search_collection(name, query, n=3):
    col = chroma_client.get_collection(name, embedding_function=embedding_fn)
    results = col.query(query_texts=[query], n_results=n)
    return "\n\n".join(results["documents"][0])

def ask(query, source):
    # Step 1: Retrieve context based on source choice
    if source == "1":
        context = search_collection("work_orders", query, n=2)
        source_label = "Work Order Records"
    elif source == "2":
        context = search_collection("manual", query, n=3)
        source_label = "X4 7300/7500 Manual"
    else:
        ctx1 = search_collection("work_orders", query, n=2)
        ctx2 = search_collection("manual", query, n=2)
        context = f"WORK ORDER RECORDS:\n{ctx1}\n\nMANUAL CONTENT:\n{ctx2}"
        source_label = "Work Orders + Manual"

    # Step 2: Send to GPT
    prompt = f"""You are a maintenance assistant for the Carrier X4 7300/7500
Trailer Refrigeration Unit.
Answer clearly and accurately based ONLY on the provided context.
Always mention which manual or source the answer comes from.
If the answer is not found, say "I could not find that in the available records."

SOURCE: {source_label}

CONTEXT:
{context}

QUESTION: {query}

Answer:"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

print("   CARRIER X4 7300/7500 — WORK ORDER RAG SYSTEM")
print("\n  What do you want to search?")
print("  1 → Work Order Records")
print("  2 → X4 7300/7500 Service Manual")
print("  3 → Both")

source = input("\n  Enter choice (1/2/3): ").strip()

labels = {"1": "Work Order Records", "2": "X4 7300/7500 Manual", "3": "Both"}
print(f"\n Searching in : {labels.get(source, 'Both')}")
print("  Type 'exit' to quit")

while True:
    try:
        query = input("\n Type a Question (or press Enter / type 'exit' to quit): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n Thank you!\n")
        break

    # Exit conditions
    if query.lower() in ["exit", "quit", "q", "bye", "no", "n", ""]:
        print("\n Thank you!\n")
        break

    print("\n Answer:")
    answer = ask(query, source)
    for line in answer.strip().split("\n"):
        print(f"  {line}")