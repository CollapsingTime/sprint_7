import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from build_index import MODEL_CACHE, MODEL_NAME, QUERY_PREFIX, ROOT

INDEX_DIR = ROOT / "index"
QUERIES = [
    "Что такое Район?",
    "Кто такой Лук Прогуливающийся?",
    "Какое оружие использует Смертоносный Шар?",
]


def load_chunks():
    with (INDEX_DIR / "chunks.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    chunks = load_chunks()
    model = SentenceTransformer(
        MODEL_NAME, cache_folder=str(MODEL_CACHE), device="cpu"
    )
    vectors = np.asarray(
        model.encode(
            [f"{QUERY_PREFIX}{query}" for query in QUERIES],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    _, ids = index.search(vectors, 1)
    for query, chunk_id in zip(QUERIES, ids[:, 0]):
        print(f"\n{query}")
        print(chunks[int(chunk_id)]["text"][:300])


if __name__ == "__main__":
    main()
