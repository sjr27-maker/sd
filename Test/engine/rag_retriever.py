import os
import json
import pickle
import numpy as np
from pathlib import Path
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_DATA_DIR    = Path(__file__).parent.parent / "data"
_CHUNKS_DIR  = _DATA_DIR / "ncert_chunks"
_INDEX_PATH  = _DATA_DIR / "faiss_index.pkl"
_META_PATH   = _DATA_DIR / "chunk_metadata.json"

# Lazy-loaded at first call
_index    = None
_metadata = None   # list of {text, subject, grade, topic, chunk_id}

# ── Embedding ─────────────────────────────────────────────────────────
def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings using Google's native embedding model via the Client."""
    try:
        # ── THE FIX: Use the client.models interface ──
        result = _client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
            config={
                "task_type": "retrieval_document"
            }
        )
        
        # The new SDK returns a list of objects. We extract the 'values' (floats) from each.
        return np.array([e.values for e in result.embeddings], dtype="float32")
        
    except Exception as e:
        print(f"❌ [RAG] Embedding Error: {e}")
        # Fallback to zero-vectors to prevent the whole system from crashing
        return np.zeros((len(texts), 768), dtype="float32")
# ── Index loading ─────────────────────────────────────────────────────
def _load_index():
    global _index, _metadata

    if _index is not None:
        return  # already loaded

    if not _INDEX_PATH.exists() or not _META_PATH.exists():
        print("  [RAG] Index not found — building from NCERT chunks...")
        _build_index()
        return

    with open(_INDEX_PATH, "rb") as f:
        _index = pickle.load(f)

    with open(_META_PATH) as f:
        _metadata = json.load(f)

    print(f"  [RAG] Index loaded — {len(_metadata)} chunks")

def _build_index():
    """
    Build FAISS index from text files in data/ncert_chunks/.
    Each .txt file = one chunk. Filename convention:
        {subject}_{grade}_{topic}_{chunk_num}.txt
    e.g. mathematics_9_quadratic_equations_01.txt
    """
    global _index, _metadata

    try:
        import faiss
    except ImportError:
        print("  [RAG] faiss-cpu not installed. pip install faiss-cpu")
        _index    = None
        _metadata = []
        return

    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_files = list(_CHUNKS_DIR.glob("*.txt"))

    if not chunk_files:
        print("  [RAG] No chunk files found in data/ncert_chunks/")
        print("  [RAG] Add NCERT text chunks as .txt files to enable RAG.")
        _index    = None
        _metadata = []
        return

    texts = []
    meta  = []

    for f in sorted(chunk_files):
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue

        # Parse filename for metadata
        parts = f.stem.split("_")
        subject = parts[0] if len(parts) > 0 else "general"
        grade   = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        topic   = "_".join(parts[2:-1]) if len(parts) > 3 else "general"

        texts.append(text)
        meta.append({
            "chunk_id": f.stem,
            "text":     text,
            "subject":  subject,
            "grade":    grade,
            "topic":    topic,
        })

    if not texts:
        _index    = None
        _metadata = []
        return

    print(f"  [RAG] Embedding {len(texts)} chunks...")

    # Embed in batches of 100
    all_embeddings = []
    batch_size     = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs  = _embed(batch)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings)
    dim        = embeddings.shape[1]

    # Build FAISS flat L2 index
    index = faiss.IndexFlatIP(dim)     # inner product = cosine on normalised vecs

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    with open(_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    _index    = index
    _metadata = meta
    print(f"  [RAG] Index built and saved — {len(meta)} chunks")

# ── Retrieval ─────────────────────────────────────────────────────────
def retrieve_context(query: str,
                     subject: str,
                     grade: int,
                     top_k: int = 3) -> str:
    """
    Retrieve top_k relevant NCERT chunks for query.
    Filters by subject and grade when possible.
    Returns formatted string for prompt injection.
    """
    _load_index()

    if _index is None or not _metadata:
        return ""   # RAG unavailable — tutor continues without it

    try:
        import faiss
    except ImportError:
        return ""

    # Embed query
    q_emb = _embed([query])
    faiss.normalize_L2(q_emb)

    # Search
    scores, indices = _index.search(q_emb, top_k * 3)  # over-fetch then filter

    results = []
    seen    = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        chunk = _metadata[idx]

        # Prefer matching subject and grade
        subject_match = chunk["subject"].lower() == subject.lower()
        grade_match   = chunk["grade"] == grade or chunk["grade"] == 0

        if chunk["chunk_id"] in seen:
            continue
        seen.add(chunk["chunk_id"])

        results.append({
            "text":    chunk["text"],
            "topic":   chunk["topic"],
            "score":   float(score),
            "priority": 2 if (subject_match and grade_match)
                        else 1 if subject_match
                        else 0,
        })

    # Sort by priority then score
    results.sort(key=lambda x: (x["priority"], x["score"]), reverse=True)
    top = results[:top_k]

    if not top:
        return ""

    # Format for prompt
    context_parts = []
    for r in top:
        topic_label = r["topic"].replace("_", " ").title()
        context_parts.append(
            f"[{topic_label}]\n{r['text']}"
        )

    return "\n\n".join(context_parts)

# ── Index management ──────────────────────────────────────────────────
def rebuild_index():
    """Force rebuild the index. Run this after adding new NCERT chunks."""
    global _index, _metadata
    _index    = None
    _metadata = None

    if _INDEX_PATH.exists():
        _INDEX_PATH.unlink()
    if _META_PATH.exists():
        _META_PATH.unlink()

    _build_index()
    print("  [RAG] Index rebuilt successfully.")

def add_ncert_pdf(pdf_path: str,
                  subject: str,
                  grade: int,
                  chunk_size: int = 400):
    """
    Utility: extract text from a NCERT PDF, chunk it,
    and save chunks to data/ncert_chunks/.
    Run this once per PDF then call rebuild_index().
    """
    try:
        import pdfplumber
    except ImportError:
        print("pip install pdfplumber  to use PDF ingestion")
        return

    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )

    # Simple sentence-aware chunking
    words    = full_text.split()
    chunks   = []
    current  = []

    for word in words:
        current.append(word)
        if len(current) >= chunk_size and word.endswith("."):
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    subject_clean = subject.lower().replace(" ", "_")
    saved = 0

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:  # skip tiny chunks
            continue
        filename = f"{subject_clean}_{grade}_general_{i+1:03d}.txt"
        (  _CHUNKS_DIR / filename).write_text(chunk, encoding="utf-8")
        saved += 1

    print(f"  [RAG] Saved {saved} chunks from {pdf_path}")
    print(f"  [RAG] Now run: from engine.rag_retriever import rebuild_index; rebuild_index()")