import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE = ROOT / "knowledge_base"
if not KNOWLEDGE_BASE.exists():
    KNOWLEDGE_BASE = ROOT.parent / "knowledge_base"
SOURCE_ROOT = ROOT.parent if KNOWLEDGE_BASE == ROOT.parent / "knowledge_base" else ROOT
OUTPUT_DIR = ROOT / "index"
MODEL_CACHE = ROOT / "models"
MODEL_NAME = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
CHUNK_SIZE = 2200
CHUNK_OVERLAP = 200
MIN_WORDS = 100
MAX_WORDS = 300
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

def load_documents():
    documents = []
    for path in sorted(KNOWLEDGE_BASE.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        heading = re.match(r"^\s*#\s+(.+?)\s*(?:\r?\n|$)", text)
        if heading:
            title = heading.group(1).strip()
            body = text[heading.end():]
            body_start = heading.end()
        else:
            title, body, body_start = path.stem, text, 0
        if not body.strip():
            continue
        source = path.resolve().relative_to(SOURCE_ROOT).as_posix()
        documents.append({
            "path": path, "source": source, "title": title,
            "body": body, "body_start": body_start,
        })
    if not documents:
        raise ValueError("No knowledge-base documents found")
    return documents

def make_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=SEPARATORS,
    )
    chunks = []
    for document in documents:
        parts = splitter.create_documents([document["body"]])
        for index, part in enumerate(parts):
            text = part.page_content.strip()
            word_count = len(re.findall(r"\S+", text))
            if not MIN_WORDS <= word_count <= MAX_WORDS:
                raise ValueError(
                    f"{document['source']}#{index + 1}: {word_count} words"
                )
            start = document["body_start"] + int(part.metadata["start_index"])
            chunks.append({
                "text": text, "chunk_id": f"{document['source']}:{index}",
                "chunk_index": index, "total_chunks": len(parts),
                "source_path": document["source"], "file_name": document["path"].name,
                "title": document["title"], "start_char": start,
                "end_char": start + len(text), "word_count": word_count,
            })
    if not chunks:
        raise ValueError("No chunks were created")
    return chunks

def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {OUTPUT_DIR}")
    started = time.perf_counter()
    documents = load_documents()
    chunks = make_chunks(documents)
    model = SentenceTransformer(
        MODEL_NAME, cache_folder=str(MODEL_CACHE), device="cpu"
    )
    passages = [
        f"{PASSAGE_PREFIX}{chunk['title']}\n{chunk['text']}"
        for chunk in chunks
    ]
    embeddings = np.asarray(
        model.encode(
            passages,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))
    with (OUTPUT_DIR / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    manifest = {
        "embedding": {"model": MODEL_NAME, "dimension": int(embeddings.shape[1])},
        "document_count": len(documents), "chunk_count": len(chunks),
        "word_count_range": {"min": MIN_WORDS, "max": MAX_WORDS},
        "encoding_seconds": round(time.perf_counter() - started, 3),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
    print(f"Index saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
