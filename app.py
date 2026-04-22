"""
LocalKB - Local Knowledge Base Manager
Inspired by DeepTutor's KB structure (HKUDS/DeepTutor)
Data structure: data/{kb_name}/raw/, chunks.json, index.json, metadata.json
"""

import os
import json
import re
import math
import uuid
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from collections import Counter

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Config ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "data"
BASE_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
CHUNK_SIZE = 500        # chars per chunk
CHUNK_OVERLAP = 80


# ── Text Extraction ──────────────────────────────────────────
def extract_text_from_file(filepath: Path) -> str:
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(filepath))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return f"[PDF parse error: {e}]"
    elif ext in (".txt", ".md", ".markdown"):
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                return filepath.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return "[Encoding error]"
    return ""


def split_into_chunks(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += size - overlap
    return chunks


# ── BM25 Index ───────────────────────────────────────────────
def tokenize(text: str):
    """Simple CJK + ASCII tokenizer."""
    # split on CJK chars individually, keep ASCII words
    tokens = []
    for part in re.split(r"([\u4e00-\u9fff\u3040-\u30ff])", text.lower()):
        if not part:
            continue
        if re.match(r"[\u4e00-\u9fff\u3040-\u30ff]", part):
            tokens.append(part)
        else:
            tokens.extend(re.findall(r"[a-z0-9]+", part))
    return tokens


class BM25:
    """Minimal BM25 implementation (no external deps needed as fallback)."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = []
        self.df = {}
        self.idf = {}
        self.avgdl = 0
        self.N = 0

    def fit(self, docs):
        self.corpus_tokens = [tokenize(d) for d in docs]
        self.N = len(self.corpus_tokens)
        self.avgdl = sum(len(t) for t in self.corpus_tokens) / max(self.N, 1)
        self.df = {}
        for tokens in self.corpus_tokens:
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                    for t, df in self.df.items()}

    def score(self, query: str, top_k=10):
        q_tokens = tokenize(query)
        scores = []
        for i, tokens in enumerate(self.corpus_tokens):
            tf_map = Counter(tokens)
            dl = len(tokens)
            sc = 0.0
            for t in q_tokens:
                if t not in self.idf:
                    continue
                tf = tf_map.get(t, 0)
                sc += self.idf[t] * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                )
            scores.append((i, sc))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ── KB helpers ───────────────────────────────────────────────
def kb_path(kb_name: str) -> Path:
    return BASE_DIR / kb_name


def load_metadata(kb_name: str) -> dict:
    p = kb_path(kb_name) / "metadata.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return {}


def save_metadata(kb_name: str, meta: dict):
    p = kb_path(kb_name) / "metadata.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(kb_name: str) -> list:
    p = kb_path(kb_name) / "chunks.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return []


def save_chunks(kb_name: str, chunks: list):
    p = kb_path(kb_name) / "chunks.json"
    p.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def build_bm25(kb_name: str) -> BM25:
    chunks = load_chunks(kb_name)
    bm = BM25()
    bm.fit([c["text"] for c in chunks])
    return bm


# ── API: KB management ───────────────────────────────────────
@app.route("/api/kbs", methods=["GET"])
def list_kbs():
    result = []
    if BASE_DIR.exists():
        for d in BASE_DIR.iterdir():
            if d.is_dir():
                meta = load_metadata(d.name)
                chunks = load_chunks(d.name)
                raw_dir = d / "raw"
                file_count = len(list(raw_dir.glob("*"))) if raw_dir.exists() else 0
                result.append({
                    "name": d.name,
                    "description": meta.get("description", ""),
                    "created_at": meta.get("created_at", ""),
                    "file_count": file_count,
                    "chunk_count": len(chunks),
                })
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify(result)


@app.route("/api/kbs", methods=["POST"])
def create_kb():
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    # sanitize
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    kp = kb_path(name)
    if kp.exists():
        return jsonify({"error": "KB already exists"}), 409
    (kp / "raw").mkdir(parents=True)
    meta = {
        "name": name,
        "description": data.get("description", ""),
        "created_at": datetime.now().isoformat(),
    }
    save_metadata(name, meta)
    return jsonify({"ok": True, "name": name})


@app.route("/api/kbs/<kb_name>", methods=["DELETE"])
def delete_kb(kb_name):
    kp = kb_path(kb_name)
    if not kp.exists():
        return jsonify({"error": "Not found"}), 404
    shutil.rmtree(kp)
    return jsonify({"ok": True})


@app.route("/api/kbs/<kb_name>/files", methods=["GET"])
def list_files(kb_name):
    raw_dir = kb_path(kb_name) / "raw"
    if not raw_dir.exists():
        return jsonify([])
    files = []
    for f in raw_dir.iterdir():
        if f.is_file():
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "mtime": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(files)


@app.route("/api/kbs/<kb_name>/upload", methods=["POST"])
def upload_file(kb_name):
    kp = kb_path(kb_name)
    if not kp.exists():
        return jsonify({"error": "KB not found"}), 404

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported type: {ext}"}), 400

    raw_dir = kp / "raw"
    raw_dir.mkdir(exist_ok=True)
    save_path = raw_dir / file.filename
    file.save(str(save_path))

    # Extract + chunk
    text = extract_text_from_file(save_path)
    new_chunks = split_into_chunks(text)

    existing = load_chunks(kb_name)
    # remove old chunks from this file
    existing = [c for c in existing if c.get("source") != file.filename]
    for i, chunk_text in enumerate(new_chunks):
        existing.append({
            "id": str(uuid.uuid4()),
            "source": file.filename,
            "chunk_index": i,
            "text": chunk_text,
        })
    save_chunks(kb_name, existing)

    # update metadata
    meta = load_metadata(kb_name)
    meta["updated_at"] = datetime.now().isoformat()
    save_metadata(kb_name, meta)

    return jsonify({
        "ok": True,
        "filename": file.filename,
        "chunks_added": len(new_chunks),
        "total_chunks": len(existing),
    })


@app.route("/api/kbs/<kb_name>/files/<filename>", methods=["DELETE"])
def delete_file(kb_name, filename):
    raw_dir = kb_path(kb_name) / "raw"
    fpath = raw_dir / filename
    if not fpath.exists():
        return jsonify({"error": "File not found"}), 404
    fpath.unlink()

    # Remove chunks
    chunks = [c for c in load_chunks(kb_name) if c.get("source") != filename]
    save_chunks(kb_name, chunks)
    return jsonify({"ok": True, "remaining_chunks": len(chunks)})


# ── API: Search ───────────────────────────────────────────────
@app.route("/api/kbs/<kb_name>/search", methods=["POST"])
def search(kb_name):
    kp = kb_path(kb_name)
    if not kp.exists():
        return jsonify({"error": "KB not found"}), 404

    data = request.json or {}
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", 8)), 20)

    if not query:
        return jsonify({"error": "Query required"}), 400

    chunks = load_chunks(kb_name)
    if not chunks:
        return jsonify({"results": [], "total": 0})

    bm = BM25()
    bm.fit([c["text"] for c in chunks])
    hits = bm.score(query, top_k=top_k)

    results = []
    for idx, score in hits:
        if score <= 0:
            continue
        c = chunks[idx]
        # highlight query terms
        snippet = c["text"][:300]
        results.append({
            "id": c["id"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "score": round(score, 4),
            "snippet": snippet,
            "text": c["text"],
        })

    return jsonify({"results": results, "total": len(results), "query": query})


# ── API: Stats ────────────────────────────────────────────────
@app.route("/api/kbs/<kb_name>/stats", methods=["GET"])
def kb_stats(kb_name):
    kp = kb_path(kb_name)
    if not kp.exists():
        return jsonify({"error": "Not found"}), 404

    chunks = load_chunks(kb_name)
    raw_dir = kp / "raw"
    files = list(raw_dir.glob("*")) if raw_dir.exists() else []

    total_chars = sum(len(c["text"]) for c in chunks)
    sources = list({c["source"] for c in chunks})

    return jsonify({
        "kb_name": kb_name,
        "file_count": len(files),
        "chunk_count": len(chunks),
        "total_chars": total_chars,
        "sources": sources,
        "metadata": load_metadata(kb_name),
    })


# ── Serve frontend ───────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    print("LocalKB running at http://localhost:5800")
    app.run(host="0.0.0.0", port=5800, debug=False)
