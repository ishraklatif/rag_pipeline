# 🧠 Hybrid RAG Pipeline — RAG vs Long Context

A modular, local-first pipeline built with **LangChain**, **Hugging Face models**, and **ChromaDB** that lets you directly compare two fundamental approaches to context injection in LLMs: **Retrieval-Augmented Generation (RAG)** and **Long Context Windows**.

> Companion project to the blog post: *The Core Limitation of Large Language Models*

---

## 🧠 The Problem This Project Explores

LLMs are frozen in time — they know nothing about your private documents, internal data, or anything after their training cutoff. To answer questions about your own data, you must inject the right context into the model at query time.

Two major architectural approaches solve this:

### Approach 1 — Retrieval-Augmented Generation (RAG)

Instead of sending all documents to the model, RAG retrieves only the most relevant pieces.

```
Documents → Chunks → Embeddings → Vector DB → (query) → Top-K chunks → LLM → Answer
```

**Strengths:**
- Compute efficient — only a few chunks processed per query
- Scales to massive datasets (terabytes, petabytes)
- Forces the model to focus on relevant signal

**The key risk — silent retrieval failure:**
The correct answer exists in the database, but the semantic search fails to fetch it. The model never sees the information and may generate an incorrect answer without warning.

---

### Approach 2 — Long Context Windows

Instead of retrieving snippets, the full document is injected directly into the model's context window. The LLM uses its attention mechanism to find relevant information itself.

```
Documents → Prompt → LLM → Answer
```

**Strengths:**
- No retrieval step — no retrieval failure
- Enables global reasoning across entire documents
- Simpler architecture (no vector DB, no chunking strategy)

**The key limitation — the "whole book" problem:**
Some questions require reasoning about what is *absent* from a document, not just what is present. For example:

> *"Which security requirements were omitted from the final release?"*

RAG will retrieve chunks about security requirements — but it cannot retrieve the *absence* of information. Long context enables this by placing both documents in the prompt simultaneously.

**The constraint:**
Even million-token windows are small compared to enterprise data lakes. And larger prompts are expensive — a 500-page manual is ~250k tokens processed on every single query.

---

### Why This Project Uses Both

Neither approach replaces the other. They solve different parts of the context problem:

| | Standard RAG | Long Context |
|---|---|---|
| Dataset size | Unlimited | Bounded (fits in window) |
| Query type | Factual lookups, summaries | Comparisons, gap analysis |
| Infrastructure | ChromaDB + embeddings | Prompt only |
| Failure mode | Silent retrieval miss | Input truncation |
| Cost | Low (few chunks per query) | Higher (full doc each time) |

This pipeline lets you switch between both modes on the same query so you can observe the difference directly.

---

## 🚀 Features

- 🔀 **Hybrid routing** — auto-detects comparison queries and routes to long-context mode
- 🎛️ **Manual mode switching** — force `/rag` or `/lc` mode at any time mid-session
- 📄 **Multi-format ingestion** — loads `.pdf`, `.txt`, and `.md` from local `/data` folder
- 🌐 **Web ingestion** — fetches and embeds content from URLs defined in `config.py`
- 🧩 **BGE embeddings** — `BAAI/bge-base-en-v1.5` running fully locally
- 🧠 **Phi-3-mini-128k** — 128k context window, MPS-compatible on Apple Silicon
- 💾 **Persistent ChromaDB** — embeddings stored locally, rebuilt only when needed
- ⚙️ **Single config file** — all models, paths, and URLs in `config.py`

---

## 🧩 Project Structure

```
rag_pipeline/
│
├── config.py              # All settings: models, paths, URLs, chunking
├── ingest.py              # Loads pdf/txt/md from disk and web URLs
├── embeddings_store.py    # Builds or loads ChromaDB vectorstore
├── llm.py                 # Loads Phi-3-mini via HuggingFace pipeline
├── rag_pipeline.py        # Both chains + router + interactive loop
├── run_rag.py             # CLI entry point
├── data/                  # Drop your documents here
└── chroma_db/             # Auto-generated, ignored by git
```

---

## ⚙️ Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install pypdf langchain-huggingface
```

---

## ▶️ Run

```bash
python run_rag.py
```

Choose your data source:

```
[1] Web
[2] Local documents (/data folder)
[3] Both
```

Then query in any mode:

```
Hybrid RAG ready.
Current mode: Auto (keyword routing)
Type /help to see mode commands.

Question: Who is Ishrak?
  → [Standard RAG]
  Answer: Ishrak is a Master of AI student at Monash University...

Question: /lc
  Mode switched to: Long Context (forced)

Question: Who is Ishrak?
  → [Long Context]
  Answer: Based on the full document...

Question: /auto
  Mode switched to: Auto (keyword routing)

Question: What skills are missing from the CV compared to the job description?
  → [Long Context]   ← auto-routed: detected comparison query
```

---

## 🎛️ Mode Commands

| Command | Effect |
|---|---|
| `/rag` | Lock into Standard RAG for all subsequent questions |
| `/lc` | Lock into Long Context for all subsequent questions |
| `/auto` | Return to automatic keyword-based routing (default) |
| `/mode` | Print the current active mode |
| `/help` | Show all commands |
| `exit` | Quit |

Auto-routing triggers long-context mode when your question contains words like: `compare`, `difference`, `missing`, `versus`, `contrast`, `omitted`, `across documents`, etc.

---

## ⚙️ Configuration

All settings live in `config.py`:

```python
LLM_MODEL_NAME       = "microsoft/Phi-3-mini-128k-instruct"  # 128k context window
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
HF_DEVICE            = "mps"          # "mps" for Apple Silicon, "cuda" or "cpu" otherwise
CHUNK_SIZE           = 800
CHUNK_OVERLAP        = 150
TOP_K                = 4              # chunks retrieved per RAG query

URLS = [                              # web sources to ingest
    "https://ishraklatif.github.io",
]
```

To switch models, change `LLM_MODEL_NAME`. Delete `chroma_db/` afterwards so the vectorstore rebuilds cleanly.

---

## 📦 Data & Vectorstore

- Drop documents into `/data` — supports `.pdf`, `.txt`, `.md`
- Embeddings are persisted in `/chroma_db` and reloaded on subsequent runs
- Both folders are gitignored

---

## 🧾 License

MIT License — free to use, modify, and extend.

---

## 🧑‍💻 Author

Developed by **Ishrak Latif**  
Master of Artificial Intelligence — Monash University  
🔗 [ishraklatif.github.io](https://ishraklatif.github.io)