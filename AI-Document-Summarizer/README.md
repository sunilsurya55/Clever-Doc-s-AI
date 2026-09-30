# AI Document Question Answering (Local RAG + Flask + Ollama + MongoDB)

A full-stack, locally-run **document Q&A and summarisation system**:

- Secure signup/login with per-user sessions
- File uploads: PDF, PPTX, PPT, DOCX, TXT, Markdown
- Text extraction (including tables in DOCX/PPTX, page/slide tagging for citations)
- Chunking + embeddings (SentenceTransformer)
- Vector indexing and retrieval via FAISS
- Local LLM summarisation and RAG-based Q&A via **Ollama**
- Per-user document history stored in **local MongoDB**
- Ownership-isolated documents (users can never see/query each other's files)
- Responsive, animated UI with live model selection and a "My Documents" picker

Everything (embeddings, FAISS, and the LLM) runs **entirely on your machine** — no cloud AI calls, no API keys.

## Architecture

```
Browser (index.html/login.html/signup.html)
      │  fetch() JSON
      ▼
Flask app (app/app.py)
      │
      ├── app/extract.py     -> pulls raw text out of PDF/DOCX/PPTX/TXT/MD
      ├── app/chunking.py    -> splits text into overlapping token chunks
      ├── app/embeddings.py  -> SentenceTransformer embeddings + FAISS index (outputs/)
      ├── app/retrieval.py   -> top-k retrieval for a query
      └── ollama.generate()  -> local LLM for summarisation / Q&A
      │
      ▼
MongoDB (users, uploads, summaries collections)
```

## Project structure

```
.
├── app/
│   ├── app.py                  # Flask routes: auth, upload, index, summarize, ask
│   ├── extract.py              # Text extraction per file type
│   ├── chunking.py             # Token-based chunking (offline-safe)
│   ├── embeddings.py           # Embedding model + FAISS persistence
│   ├── retrieval.py            # Query embedding + top-k FAISS search
│   └── templates/
│       ├── index.html          # Main dashboard (upload / summarize / ask)
│       ├── login.html
│       └── signup.html
├── uploads/                     # Saved uploaded files (created at runtime)
├── outputs/                     # FAISS indexes, chunks, embeddings, summaries (created at runtime)
├── requirements.txt
├── run.sh
└── README.md
```

## Prerequisites

- Python 3.10+
- MongoDB running locally (or reachable via `MONGO_URI`)
- [Ollama](https://ollama.com) installed and running, with at least one model pulled
- Optional: LibreOffice (`soffice` on PATH) only if you need to support legacy `.ppt` files
- Optional: `tesseract` binary only if you want OCR on image files

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Start MongoDB (pick whichever matches your OS/setup):

```bash
# macOS (Homebrew service)
brew services start mongodb-community
# Linux (systemd)
sudo systemctl start mongod
# or run directly
mongod --dbpath /path/to/your/db
```

Start Ollama and pull the models you want available:

```bash
ollama serve
ollama pull gemma3:1b
ollama pull qwen2.5
ollama pull llama3.2
ollama pull mistral
```

Run the app:

```bash
chmod +x run.sh
./run.sh
```

Then open **http://127.0.0.1:5000**

## Configuration (environment variables, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `ai_summariser` | Database name |
| `FLASK_SECRET_KEY` | `dev-secret-change-me` | **Change this in production** — used to sign session cookies |
| `EMBEDDING_MODEL_LOCAL_PATH` | `~/pdf_summariser_models/all-MiniLM-L6-v2` | Local path for the embedding model; if absent, it's downloaded from Hugging Face Hub on first use and cached |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face Hub model name used as a fallback |

## Notes on security & scope

- Passwords are hashed with Werkzeug's PBKDF2 (`generate_password_hash`) — not plaintext, not a weak/reversible scheme.
- Every document-bearing API route checks that the logged-in user actually owns the `prefix` being requested before touching any file, FAISS index, or Ollama call.
- The password policy is intentionally a 4-digit PIN by original design — fine for local/personal use, but you should tighten this (longer passwords, rate limiting, email verification) before any multi-user or public deployment.
- There is no server-side session idle timeout configured beyond Flask's default cookie behavior — add `PERMANENT_SESSION_LIFETIME` if you need one.
