# app/app.py
import os
import time
import subprocess
import re
import datetime
from functools import wraps
from pathlib import Path
from typing import List

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ServerSelectionTimeoutError, DuplicateKeyError

from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_from_directory,
    redirect,
    url_for,
    session,
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge

import ollama

# local modules
from app.extract import extract_text
from app.chunking import prepare_chunks
from app.embeddings import (
    embed_chunks,
    build_faiss_index,
    persist_index,
    load_index_and_chunks,
    load_embeddings_if_exists,
)
from app.retrieval import retrieve_top_k, assemble_context

# -------------------------
# Config / folders
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".pdf", ".txt", ".pptx", ".ppt", ".docx", ".md"}

# Models offered for selection. Users can also type a custom model name that
# is already pulled locally via `ollama pull <model>`.
AVAILABLE_MODELS = [
    "gemma3:1b",
    "qwen2.5:latest",
    "llama3.2:latest",
    "mistral:latest",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]

app = Flask(__name__, template_folder=str(BASE_DIR / "app" / "templates"))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
# 50 MB max upload size; anything larger gets a clean JSON 413 instead of a crash.
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Session secret key (for login sessions). Override with FLASK_SECRET_KEY in production.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# -------------------------
# MongoDB (local, offline)
# -------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "ai_summariser")
MONGO_USERS_COL = os.environ.get("MONGO_USERS_COL", "users")
MONGO_UPLOADS_COL = os.environ.get("MONGO_UPLOADS_COL", "uploads")
MONGO_SUMMARIES_COL = os.environ.get("MONGO_SUMMARIES_COL", "summaries")

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
mongo_db = mongo_client[MONGO_DB_NAME]
users_col = mongo_db[MONGO_USERS_COL]
uploads_col = mongo_db[MONGO_UPLOADS_COL]
summaries_col = mongo_db[MONGO_SUMMARIES_COL]

try:
    mongo_client.admin.command("ping")
    users_col.create_index([("email", ASCENDING)], unique=True)
    uploads_col.create_index([("user_id", ASCENDING), ("prefix", ASCENDING)], unique=True)
    print("[startup] Connected to MongoDB at", MONGO_URI)
except ServerSelectionTimeoutError as e:
    print(f"[startup] WARNING: could not connect to MongoDB at {MONGO_URI}: {e}")
    print("[startup] Signup/Login/history features will fail until MongoDB is reachable.")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def hash_password(raw_password: str) -> str:
    """Hash a password using Werkzeug (PBKDF2)."""
    raw = str(raw_password or "")
    return generate_password_hash(raw)


def verify_password(raw_password: str, hashed: str) -> bool:
    """Verify password against stored hash."""
    raw = str(raw_password or "")
    if not hashed:
        return False
    try:
        return check_password_hash(hashed, raw)
    except Exception:
        return False


# -------------------------
# Auth helpers
# -------------------------
def login_required(f):
    """
    Protects JSON API routes. Unlike the page routes (which redirect to /login),
    API routes return a clean 401 JSON payload so the frontend JS can show an
    error message instead of receiving an HTML redirect it can't parse.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"ok": False, "error": "Authentication required. Please log in."}), 401
        return f(*args, **kwargs)
    return wrapper


def current_user_id() -> str:
    return session.get("user_id")


def require_owns_prefix(prefix: str):
    """
    Returns None if the current session user owns this prefix, otherwise
    returns a Flask response (403) that the caller should return immediately.
    """
    user_id = current_user_id()
    record = uploads_col.find_one({"prefix": prefix, "user_id": user_id})
    if not record:
        return jsonify({"ok": False, "error": "Document not found or you do not have access to it."}), 403
    return None


# -------------------------
# Ollama helpers
# -------------------------
def list_local_ollama_models() -> List[str]:
    """
    Returns the list of model names currently pulled/available in the local
    Ollama installation. Returns [] if Ollama can't be reached (daemon not
    running, not installed, etc.) instead of raising.
    """
    try:
        resp = ollama.list()
        # The python client returns either a dict {"models": [...]}
        # or (in newer versions) a ListResponse object with a `.models` attribute.
        models = resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", [])
        names = []
        for m in models or []:
            name = m.get("name") if isinstance(m, dict) else getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(name)
        return names
    except Exception as e:
        print(f"[ollama] Could not list local models: {e}")
        return []


def call_ollama(model_name: str, prompt: str, timeout: int = 300) -> str:
    """
    Calls the local Ollama model and returns the generated text.

    IMPORTANT: the `ollama` python package does NOT have a `.run()` method
    (a bug in the original code caused every call to fail and silently fall
    back to a slow subprocess every time). The correct call is
    `ollama.generate(model=..., prompt=...)`.

    This function tries the python client first, and only falls back to the
    `ollama` CLI if the client genuinely cannot reach the daemon (e.g. it's
    not running, or the python package isn't installed correctly) -- not
    because of a wrong method name.
    """
    try:
        resp = ollama.generate(model=model_name, prompt=prompt)
        if isinstance(resp, dict):
            text = resp.get("response", "")
        else:
            text = getattr(resp, "response", "")
        text = str(text).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text
    except Exception as client_err:
        # Fallback path: only used if the python client itself couldn't
        # complete the call (e.g. connection refused to the Ollama daemon).
        try:
            proc = subprocess.run(
                ["ollama", "run", model_name, "--nowordwrap"],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="ignore") if isinstance(proc.stderr, bytes) else str(proc.stderr)
                raise RuntimeError(
                    f"Ollama could not generate a response for model '{model_name}'. "
                    f"Client error: {client_err}. CLI error: {stderr[:500]}"
                )
            out = proc.stdout
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            out = str(out).strip()
            if not out:
                raise RuntimeError(f"Ollama CLI returned an empty response for model '{model_name}'.")
            return out
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama is not installed or not found in PATH. Install it from https://ollama.com "
                "and make sure `ollama serve` is running."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Ollama timed out after {timeout}s while generating with model '{model_name}'.")


def validate_model_or_error(model_name: str):
    """
    Best-effort check that the requested model is actually pulled locally.
    Returns None if OK (or if we couldn't check), or a Flask response with a
    helpful message if we positively know the model isn't available.
    """
    local_models = list_local_ollama_models()
    if not local_models:
        # Couldn't reach Ollama to check -- let call_ollama surface the real error.
        return None
    # Ollama model names sometimes include/omit the ":latest" tag; compare loosely.
    normalized = {m.split(":")[0] for m in local_models}
    if model_name.split(":")[0] not in normalized and model_name not in local_models:
        return jsonify({
            "ok": False,
            "error": (
                f"Model '{model_name}' is not pulled locally. "
                f"Run `ollama pull {model_name}` in a terminal, then try again. "
                f"Currently available: {', '.join(local_models) if local_models else 'none'}"
            )
        }), 400
    return None


# -------------------------
# Error handlers
# -------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"ok": False, "error": "File is too large. Maximum upload size is 50MB."}), 413


@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api") or request.is_json:
        return jsonify({"ok": False, "error": "Not found"}), 404
    return redirect(url_for("index"))


# -------------------------
# HOME
# -------------------------
@app.route("/", methods=["GET"])
def index():
    # Require login to use the summariser dashboard
    if not session.get("user_id"):
        return redirect(url_for("login_page"))

    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL,
        user_name=session.get("user_name", ""),
        user_email=session.get("user_email", ""),
    )


# -------------------------
# AUTH PAGES (UI only)
# -------------------------
@app.route("/signup", methods=["GET"])
def signup_page():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("signup.html")


@app.route("/login", methods=["GET"])
def login_page():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("login.html")


# -------------------------
# SESSION INFO (for frontend UI)
# -------------------------
@app.route("/api/session", methods=["GET"])
def api_session():
    if not session.get("user_id"):
        return jsonify({"ok": True, "authenticated": False})
    return jsonify({
        "ok": True,
        "authenticated": True,
        "user": {
            "name": session.get("user_name", ""),
            "email": session.get("user_email", ""),
        }
    })


# -------------------------
# AVAILABLE MODELS (real Ollama models on this machine)
# -------------------------
@app.route("/available_models", methods=["GET"])
@login_required
def available_models():
    local_models = list_local_ollama_models()
    # Merge defaults with whatever's actually pulled, de-duplicated, defaults first.
    merged = list(dict.fromkeys(AVAILABLE_MODELS + local_models))
    return jsonify({
        "ok": True,
        "models": merged,
        "locally_available": local_models,
        "default_model": DEFAULT_MODEL,
    })


# -------------------------
# AUTH APIs (MongoDB)
# -------------------------
@app.route("/signup", methods=["POST"])
def signup_api():
    """
    JSON body:
    {
      "name": "Full Name",
      "email": "user@example.com",
      "password": "1234"
    }
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = normalize_email(data.get("email"))
    password = str(data.get("password") or "").strip()

    if len(name) < 2:
        return jsonify({"ok": False, "error": "Name must be at least 2 characters long"}), 400

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"ok": False, "error": "Please enter a valid email address"}), 400

    if not re.fullmatch(r"\d{4}", password):
        return jsonify({"ok": False, "error": "Password must be exactly 4 digits"}), 400

    try:
        existing = users_col.find_one({"email": email})
        if existing:
            return jsonify({"ok": False, "error": "This email is already registered"}), 400

        user_doc = {
            "name": name,
            "email": email,
            "password_hash": hash_password(password),
            "created_at": datetime.datetime.utcnow(),
            "last_login_at": None,
        }
        result = users_col.insert_one(user_doc)

        # Auto-login the new user so "Redirecting..." on the signup page
        # actually lands them in a logged-in dashboard instead of bouncing
        # straight back to /login with no session set.
        session["user_id"] = str(result.inserted_id)
        session["user_email"] = email
        session["user_name"] = name

        return jsonify({"ok": True, "message": "Account created successfully"}), 201

    except DuplicateKeyError:
        return jsonify({"ok": False, "error": "This email is already registered"}), 400
    except ServerSelectionTimeoutError:
        return jsonify({"ok": False, "error": "Cannot reach the database. Please try again shortly."}), 503
    except Exception as e:
        return jsonify({"ok": False, "error": f"Signup failed: {e}"}), 500


@app.route("/login", methods=["POST"])
def login_api():
    """
    JSON body:
    {
      "email": "user@example.com",
      "password": "1234"
    }
    """
    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email"))
    password = str(data.get("password") or "").strip()

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"ok": False, "error": "Please enter a valid email address"}), 400

    if not re.fullmatch(r"\d{4}", password):
        return jsonify({"ok": False, "error": "Password must be exactly 4 digits"}), 400

    try:
        user = users_col.find_one({"email": email})
        if not user:
            return jsonify({"ok": False, "error": "Invalid email or password"}), 401

        if not verify_password(password, user.get("password_hash", "")):
            return jsonify({"ok": False, "error": "Invalid email or password"}), 401

        users_col.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login_at": datetime.datetime.utcnow()}},
        )

        session["user_id"] = str(user["_id"])
        session["user_email"] = user["email"]
        session["user_name"] = user.get("name", "")

        token = f"session-{session['user_id']}"
        return jsonify({"ok": True, "message": "Login successful", "token": token}), 200

    except ServerSelectionTimeoutError:
        return jsonify({"ok": False, "error": "Cannot reach the database. Please try again shortly."}), 503
    except Exception as e:
        return jsonify({"ok": False, "error": f"Login failed: {e}"}), 500


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    if request.method == "POST":
        return jsonify({"ok": True, "message": "Logged out"}), 200
    return redirect(url_for("login_page"))


# -------------------------
# UPLOAD
# -------------------------
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"ok": False, "error": "No filename"}), 400

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"ok": False, "error": "Invalid filename"}), 400

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            return jsonify({
                "ok": False,
                "error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
            }), 400

        user_id = current_user_id()

        # Namespace the saved file + prefix per-user so two users uploading
        # files with the same name never collide or expose each other's data.
        stem = Path(filename).stem.replace(" ", "_")
        unique_suffix = str(int(time.time() * 1000))
        stored_filename = f"u{user_id}_{unique_suffix}_{filename}"
        save_path = UPLOAD_FOLDER / stored_filename
        file.save(save_path)

        prefix = f"u{user_id}_{stem}_{unique_suffix}"
        model_name = request.form.get("model", DEFAULT_MODEL)

        uploads_col.insert_one({
            "user_id": user_id,
            "prefix": prefix,
            "original_filename": filename,
            "stored_filename": stored_filename,
            "filepath": str(save_path),
            "model": model_name,
            "uploaded_at": datetime.datetime.utcnow(),
            "indexed": False,
        })

        return jsonify({
            "ok": True,
            "status": "uploaded",
            "filename": filename,
            "prefix": prefix,
            "model": model_name,
        })
    except ServerSelectionTimeoutError:
        return jsonify({"ok": False, "error": "Cannot reach the database. Please try again shortly."}), 503
    except Exception as e:
        return jsonify({"ok": False, "error": f"Upload failed: {e}"}), 500


# -------------------------
# LIST MY DOCUMENTS
# -------------------------
@app.route("/my_documents", methods=["GET"])
@login_required
def my_documents():
    user_id = current_user_id()
    try:
        docs = list(
            uploads_col.find({"user_id": user_id}).sort("uploaded_at", -1).limit(50)
        )
        out = [{
            "prefix": d["prefix"],
            "filename": d.get("original_filename"),
            "uploaded_at": d.get("uploaded_at").isoformat() if d.get("uploaded_at") else None,
            "indexed": d.get("indexed", False),
        } for d in docs]
        return jsonify({"ok": True, "documents": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _resolve_owned_filepath(prefix: str):
    """
    Resolves an upload's real filepath server-side from MongoDB, after
    verifying the current session user owns it. This is the ONLY way any
    route should ever obtain a filesystem path for a document -- routes
    must NEVER accept a raw filepath string from the client, since that
    would allow a path-traversal / arbitrary-file-read attack (a user could
    otherwise pass e.g. "/etc/passwd" or another user's uploaded file path).

    Returns (filepath_str, None) on success, or (None, error_response) on failure.
    """
    user_id = current_user_id()
    record = uploads_col.find_one({"prefix": prefix, "user_id": user_id})
    if not record:
        return None, (jsonify({"ok": False, "error": "Document not found or you do not have access to it."}), 403)
    filepath = record.get("filepath")
    if not filepath or not Path(filepath).exists():
        return None, (jsonify({"ok": False, "error": "The original uploaded file is missing on the server."}), 404)
    return filepath, None


# -------------------------
# EXTRACTION TEST
# -------------------------
@app.route("/extract_test", methods=["POST"])
@login_required
def extract_test():
    data = request.json or {}
    prefix = data.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": 'Provide JSON {"prefix": "<your document prefix>"}'}), 400

    filepath, err = _resolve_owned_filepath(prefix)
    if err:
        return err

    try:
        text = extract_text(filepath)
        return jsonify({"ok": True, "length": len(text), "snippet": text[:200]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# CHUNKING TEST
# -------------------------
@app.route("/chunk_test", methods=["POST"])
@login_required
def chunk_test():
    data = request.json or {}
    prefix = data.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": "prefix required"}), 400

    filepath, err = _resolve_owned_filepath(prefix)
    if err:
        return err

    try:
        text = extract_text(filepath)
        chunks = prepare_chunks(text)
        return jsonify({
            "ok": True,
            "chunks": len(chunks),
            "chunk_preview": chunks[0][:300] if chunks else ""
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# INDEXING ENDPOINT (embeddings + faiss) - optimized with cache reuse & logs
# -------------------------
@app.route("/index_file", methods=["POST"])
@login_required
def index_file():
    data = request.json or {}
    prefix = data.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": "prefix is required"}), 400

    # filepath is intentionally NOT accepted from the client -- it is looked
    # up server-side from the ownership-checked upload record, closing a
    # path-traversal / arbitrary-file-read hole that previously let a caller
    # pass any filesystem path for indexing.
    filepath, ownership_error = _resolve_owned_filepath(prefix)
    if ownership_error:
        return ownership_error

    try:
        start_total = time.perf_counter()
        text = extract_text(filepath)
        chunks = prepare_chunks(text)
        if not chunks:
            return jsonify({"ok": False, "error": "No text could be extracted from this document."}), 422

        out_dir = OUTPUT_FOLDER
        faiss_path_candidate = out_dir / f"{prefix}_faiss.idx"
        chunks_path_candidate = out_dir / f"{prefix}_chunks.pkl"
        emb_path_candidate = out_dir / f"{prefix}_embeddings.npy"

        if faiss_path_candidate.exists() and chunks_path_candidate.exists():
            print(f"[index_file] Existing index found for prefix='{prefix}', skipping re-index.")
            elapsed_total = time.perf_counter() - start_total
            return jsonify({
                "ok": True,
                "chunks": len(chunks),
                "faiss_path": str(faiss_path_candidate),
                "chunks_path": str(chunks_path_candidate),
                "embeddings_path": str(emb_path_candidate) if emb_path_candidate.exists() else None,
                "prefix": prefix,
                "info": "index_exists",
                "time_s": elapsed_total
            })

        emb_time = 0.0
        embeddings = load_embeddings_if_exists(prefix)
        if embeddings is not None and embeddings.shape[0] == len(chunks):
            print(f"[index_file] Reusing cached embeddings for prefix='{prefix}' (vectors={embeddings.shape[0]}).")
        else:
            print(f"[index_file] Computing embeddings for {len(chunks)} chunks...")
            t0 = time.perf_counter()
            embeddings = embed_chunks(chunks, batch_size=64, show_progress=True)
            emb_time = time.perf_counter() - t0
            print(f"[index_file] Embeddings computed in {emb_time:.2f}s, shape={embeddings.shape}.")

        t0 = time.perf_counter()
        index = build_faiss_index(embeddings)
        faiss_path, chunks_path, embeddings_path = persist_index(index, chunks, prefix, embeddings=embeddings)
        idx_time = time.perf_counter() - t0

        total_time = time.perf_counter() - start_total
        print(f"[index_file] Indexing complete: chunks={len(chunks)}, emb_time={emb_time:.2f}s, index_time={idx_time:.2f}s, total={total_time:.2f}s")

        uploads_col.update_one(
            {"prefix": prefix, "user_id": current_user_id()},
            {"$set": {"indexed": True, "chunk_count": len(chunks), "indexed_at": datetime.datetime.utcnow()}}
        )

        return jsonify({
            "ok": True,
            "chunks": len(chunks),
            "faiss_path": faiss_path,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "prefix": prefix,
            "time_s": total_time
        })
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        print("[index_file] error:", str(e))
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# SUMMARIZE endpoint (two-stage tuned, with logging)
# -------------------------
@app.route("/summarize", methods=["POST"])
@login_required
def summarize_endpoint():
    data = request.json or {}
    prefix = data.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": "prefix required (the index prefix)"}), 400

    ownership_error = require_owns_prefix(prefix)
    if ownership_error:
        return ownership_error

    query = data.get("query", "Produce headlines and a short paragraph under each headline summarising the document.")
    model_name = data.get("model", DEFAULT_MODEL)
    top_k = int(data.get("top_k", 6))
    save_flag = bool(data.get("save", True))

    model_error = validate_model_or_error(model_name)
    if model_error:
        return model_error

    try:
        start_total = time.perf_counter()
        index, chunks = load_index_and_chunks(prefix)
        n_chunks = len(chunks)
        print(f"[summarize] Starting summarization: prefix={prefix}, n_chunks={n_chunks}, model={model_name}")

        TWO_STAGE_THRESHOLD = 14
        if n_chunks > TWO_STAGE_THRESHOLD:
            # Stage 1: mini summaries in batches
            BATCH_SIZE = 8
            mini_summaries = []
            mini_start = time.perf_counter()
            total_batches = (n_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_i, i in enumerate(range(0, n_chunks, BATCH_SIZE), start=1):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                batch_context = "\n\n---\n\n".join(batch_chunks)
                batch_prompt = f"""You are an expert summariser. For the context below, produce a single high-quality mini-summary (80-120 words) using only the provided text.

Context:
{batch_context}

Mini-summary (80-120 words):
"""
                t0 = time.perf_counter()
                batch_text = call_ollama(model_name, batch_prompt, timeout=300)
                elapsed = time.perf_counter() - t0
                print(f"[summarize] mini-batch {batch_i}/{total_batches} done in {elapsed:.2f}s")
                mini_summaries.append(batch_text)
                time.sleep(0.05)

            mini_time = time.perf_counter() - mini_start
            print(f"[summarize] All mini-summaries done ({len(mini_summaries)}) in {mini_time:.2f}s")

            # Stage 2: aggregate mini summaries into final structured summary
            combined_text = "\n\n".join(mini_summaries)
            aggregate_prompt = f"""
You are an expert editor. Using ONLY the combined mini-summaries below, produce a final structured summary.

Requirements:
- 10-15 short, meaningful headlines.
- Under each headline write a substantive paragraph (100-160 words).
- Finish with a "Conclusion" section (350-500 words).
- Base content only on the provided mini-summaries.
Combined mini-summaries:
{combined_text}

Final structured summary:
"""
            t0 = time.perf_counter()
            final_text = call_ollama(model_name, aggregate_prompt, timeout=600)
            print(f"[summarize] Aggregate done in {time.perf_counter() - t0:.2f}s")
        else:
            # small doc: single-shot summarization using top_k retrieval
            selected_chunks, indices = retrieve_top_k(prefix, query, top_k=top_k)
            context = assemble_context(selected_chunks, max_chars=6500)
            direct_prompt = f"""You are an expert summariser. Given the context below produce structured headlines and for each headline a substantive paragraph (90-150 words). End with a 'Conclusion' (150-250 words). Use only the context.

Context:
{context}
"""
            t0 = time.perf_counter()
            final_text = call_ollama(model_name, direct_prompt, timeout=600)
            print(f"[summarize] Direct summarization done in {time.perf_counter() - t0:.2f}s")

        # save final
        summary_filename = f"{prefix}_summary.md"
        if save_flag:
            out_path = OUTPUT_FOLDER / summary_filename
            out_path.write_text(final_text, encoding="utf-8")
            download_url = f"/download_summary/{summary_filename}"

            summaries_col.insert_one({
                "user_id": current_user_id(),
                "prefix": prefix,
                "model": model_name,
                "chunk_count": n_chunks,
                "created_at": datetime.datetime.utcnow(),
                "download_filename": summary_filename,
            })
        else:
            download_url = None

        total_elapsed = time.perf_counter() - start_total
        print(f"[summarize] Completed summarization for prefix='{prefix}' total_time={total_elapsed:.2f}s")

        return jsonify({
            "ok": True,
            "summary": final_text,
            "download_url": download_url,
            "prefix": prefix,
            "chunks": n_chunks,
            "time_s": total_elapsed
        })
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "This document hasn't been indexed yet. Please index it first."}), 404
    except RuntimeError as e:
        # Errors raised by call_ollama() are already user-friendly.
        return jsonify({"ok": False, "error": str(e)}), 502
    except Exception as e:
        print("[summarize] error:", str(e))
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# ASK endpoint (RAG Q&A)
# -------------------------
def extract_pages_from_chunk_text(chunk_text: str) -> List[int]:
    """Return a list of page numbers found in the chunk text (if any)."""
    pages = [int(m) for m in re.findall(r"\[page:(\d+)\]", chunk_text)]
    if pages:
        return sorted(set(pages))
    slides = [int(m) for m in re.findall(r"\[slide:(\d+)\]", chunk_text)]
    if slides:
        return sorted(set(slides))
    return []


@app.route("/ask", methods=["POST"])
@login_required
def ask_endpoint():
    """
    POST JSON:
    {
      "prefix": "u1_resume_170000",   # required: index prefix
      "question": "What is virtualization?",
      "model": "gemma3:1b",           # optional
      "top_k": 5                      # optional
    }
    """
    data = request.json or {}
    prefix = data.get("prefix")
    question = data.get("question")
    if not prefix or not question:
        return jsonify({"ok": False, "error": "prefix and question required"}), 400

    ownership_error = require_owns_prefix(prefix)
    if ownership_error:
        return ownership_error

    model_name = data.get("model", DEFAULT_MODEL)
    top_k = int(data.get("top_k", 5))

    model_error = validate_model_or_error(model_name)
    if model_error:
        return model_error

    try:
        # retrieve top_k chunks (uses SentenceTransformer + FAISS)
        selected_chunks, indices = retrieve_top_k(prefix, question, top_k=top_k)
        if not selected_chunks:
            return jsonify({"ok": False, "error": "No chunks retrieved for this document."}), 500

        # build context including short markers for page numbers if present
        context_blocks = []
        for idx, chunk in zip(indices, selected_chunks):
            pages = extract_pages_from_chunk_text(chunk)
            page_hint = f"(pages: {pages[0]}-{pages[-1]})" if pages and len(pages) > 1 else (f"(page: {pages[0]})" if pages else "")
            context_blocks.append(f"CHUNK_INDEX:{idx} {page_hint}\n{chunk}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a precise assistant answering questions using ONLY the provided context. Do NOT hallucinate or add information not present in the context. If the document does not contain the answer, respond: "The document does not contain the requested information."

Context:
{context}

Question:
{question}

Answer concisely. At the end, provide a short "SOURCES" line listing page numbers or chunk indices used, e.g. "SOURCES: pages 12-14" or "SOURCES: CHUNK_INDEX:3".
"""
        answer_text = call_ollama(model_name, prompt, timeout=300)

        # Attempt to extract SOURCES line from model output
        sources = []
        m = re.search(r"SOURCES\:\s*(.*)$", answer_text, re.I | re.M)
        if m:
            sources_text = m.group(1).strip()
            sources = [s.strip() for s in re.split(r"[,;]", sources_text) if s.strip()]
        else:
            # fallback: use the pages extracted from selected chunks
            pages_set = set()
            for ch in selected_chunks:
                pages = extract_pages_from_chunk_text(ch)
                for p in pages:
                    pages_set.add(p)
            if pages_set:
                pages_list = sorted(pages_set)
                if len(pages_list) > 1:
                    sources = [f"pages {pages_list[0]}-{pages_list[-1]}"]
                else:
                    sources = [f"page {pages_list[0]}"]

        return jsonify({
            "ok": True,
            "answer": answer_text,
            "sources": sources,
            "used_chunk_indices": indices,
            "prefix": prefix
        })
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "This document hasn't been indexed yet. Please index it first."}), 404
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# DOWNLOAD summary file
# -------------------------
@app.route("/download_summary/<filename>", methods=["GET"])
@login_required
def download_summary(filename):
    safe = Path(filename).name
    # filenames are "<prefix>_summary.md" -- recover the prefix to check ownership
    prefix = safe[:-len("_summary.md")] if safe.endswith("_summary.md") else None
    if not prefix:
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    ownership_error = require_owns_prefix(prefix)
    if ownership_error:
        return ownership_error

    if not (OUTPUT_FOLDER / safe).exists():
        return jsonify({"ok": False, "error": "Summary file not found"}), 404

    return send_from_directory(str(OUTPUT_FOLDER), safe, as_attachment=True)


# RUN
# -------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
