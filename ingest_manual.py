import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import os

# ── settings ────
MANUALS_FOLDER = "manuals"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50

def extract_text_from_pdf(pdf_path):
    print(f"\n  📄 Reading: {pdf_path}")
    reader = PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            full_text += f"\n[Page {i+1}]\n{text}"
    print(f"  ✅ {len(reader.pages)} pages extracted")
    return full_text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

# setup ChromaDB 
client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name="manual",
    embedding_function=embedding_fn
)

# find all PDFs in manuals folder
pdf_files = [
    f for f in os.listdir(MANUALS_FOLDER)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    print("  ⚠️  No PDF files found in manuals/ folder!")
    exit()

print("\n" + "="*55)
print("   INGESTING ALL MANUALS INTO CHROMADB")
print("="*55)
print(f"\n  📁 Found {len(pdf_files)} PDF(s) in '{MANUALS_FOLDER}/':")
for f in pdf_files:
    print(f"     • {f}")

# process each PDF
total_chunks = 0

for pdf_file in pdf_files:
    pdf_path = os.path.join(MANUALS_FOLDER, pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]  # filename without .pdf

 # Extract text
    full_text = extract_text_from_pdf(pdf_path)

    if not full_text.strip():
        print(f"  ⚠️  No text found in {pdf_file} — skipping")
        continue

    # Chunk text
    chunks = chunk_text(full_text)
    print(f"  ✅ Split into {len(chunks)} chunks")

# Prepare for ChromaDB
    documents, metadatas, ids = [], [], []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "source":      pdf_name,    # which PDF it came from
            "filename":    pdf_file,
            "chunk_index": i
        })
        ids.append(f"{pdf_name}_chunk_{i}")  # unique ID per PDF

 # Store in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    total_chunks += len(chunks)
    print(f"  📥 Stored {len(chunks)} chunks for: {pdf_file}")

print(f"All manuals ingested successfully!")
print(f"Total PDFs processed : {len(pdf_files)}")
print(f" Total chunks stored  : {collection.count()}")
print(f"Database saved at : ./chroma_db")
