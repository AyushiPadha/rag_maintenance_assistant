# Work Order RAG — Maintenance Assistant

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about equipment work orders and service manuals, grounding every answer in retrieved source documents rather than the LLM's own (unverifiable) knowledge.

Built as a maintenance assistant for a Carrier X4 7300/7500 trailer refrigeration unit, but the pipeline generalizes to any combination of structured records (JSON/database) + unstructured documents (PDF manuals).

## Why RAG, not just a raw LLM prompt?

A general-purpose LLM has no idea what's inside *your* specific equipment's 342-page service manual, and it can't see your work order history at all. Two options exist to fix that:

1. **Fine-tune a model** on your documents — expensive, slow to update, and still doesn't reliably cite sources.
2. **RAG** — embed your documents into a vector database, retrieve the most relevant chunks for a given question at query time, and feed *only those chunks* to the LLM as context. The model answers strictly from what it's given, and can honestly say "I don't know" when the answer isn't in the retrieved context.

This project uses option 2.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│ work_orders.json │     │  manuals/*.pdf   │
└────────┬─────────┘     └─────────┬────────┘
         │                         │
   ingest.py                ingest_manual.py
   (structure per-WO        (extract text → chunk
    record as text)          into ~500-char pieces)
         │                         │
         ▼                         ▼
   ┌─────────────────────────────────────┐
   │       ChromaDB (local, persistent)   │
   │  ┌────────────────┐ ┌──────────────┐ │
   │  │ "work_orders"  │ │  "manual"    │ │
   │  │  collection    │ │  collection  │ │
   │  └────────────────┘ └──────────────┘ │
   │  embeddings via sentence-transformers │
   │       (all-MiniLM-L6-v2, local)       │
   └──────────────────┬────────────────────┘
                       │
                  ask.py (query time)
                       │
        1. Embed the user's question
        2. Retrieve top-N similar chunks
           from one or both collections
        3. Build a grounded prompt with
           only the retrieved context
        4. Send to OpenAI GPT-3.5 for the
           final natural-language answer
                       │
                       ▼
              Answer, grounded in
              retrieved source text
```

**Key design choices:**
- **Local embeddings** (`sentence-transformers`, `all-MiniLM-L6-v2`) instead of an embedding API call — free, fast, and keeps the ingestion pipeline usable offline.
- **Two separate ChromaDB collections** (`work_orders` and `manual`) rather than one merged collection, so retrieval can be scoped to just records, just the manual, or both — mixing unrelated content types in one similarity search tends to dilute retrieval quality.
- **Chunking with overlap** (500 chars, 50-char overlap) for the manual, so answers that span a chunk boundary in the source PDF aren't lost.
- **The prompt explicitly instructs the model to say "I could not find that in the available records"** rather than guess — critical for a maintenance tool, where a confidently wrong answer is worse than an honest "I don't know."

## Setup

```bash
# Install dependencies (this project uses uv)
uv sync

# Set up your API key
cp .env.example .env
# then edit .env and add your real OPENAI_API_KEY

# Add a PDF manual to ingest (see manuals/README.md)
# Then run both ingestion scripts:
uv run ingest.py           # loads work_orders.json into ChromaDB
uv run ingest_manual.py    # loads any PDF(s) in manuals/ into ChromaDB

# Ask questions
uv run ask.py
```

At the prompt, choose whether to search work order records, the manual, or both, then ask natural-language questions like:

- *"What was done for WO-1001?"*
- *"What's the refrigerant charging procedure?"*
- *"Which critical-priority work orders are still open?"*

## Tech Stack

- **ChromaDB** — local, persistent vector database
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embedding model, no API cost
- **OpenAI GPT-3.5** — final answer generation from retrieved context
- **pypdf** — PDF text extraction
- **python-dotenv** — environment variable management
- **uv** — Python dependency management

## Notes on this repo

- `chroma_db/` is excluded from git — it's regenerable binary index data, not source code. Run the ingest scripts to rebuild it locally.
- The original 342-page service manual used during development isn't included, since it's third-party copyrighted material. `manuals/README.md` explains how to substitute your own.
- `work_orders.json` contains synthetic sample data only.
