from pathlib import Path
import os

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


ES_URL = os.getenv("FOVEA_ES_URL", "http://localhost:9200")
INDEX_NAME = "esci-products"
SHARD_DIR = Path(__file__).resolve().parent / "embeddings"
STATE_FILE = Path(__file__).resolve().parent / "shards_loaded.txt"
TITLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "embeddings_input"
    / "shopping_queries_dataset"
    / "shopping_queries_dataset_products.parquet"
)
EXPECTED_DIMENSIONS = 1_024


def load_titles() -> dict[str, dict[str, str]]:
    if not TITLE_FILE.exists():
        raise FileNotFoundError(f"Title dataset not found: {TITLE_FILE}")

    titles_df = pd.read_parquet(TITLE_FILE)
    required = {"product_id", "product_title", "product_description"}
    missing = sorted(required - set(titles_df.columns))
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {TITLE_FILE}; "
            f"missing {missing}; found {titles_df.columns.tolist()}"
        )

    titles_df = titles_df[["product_id", "product_title", "product_description"]].dropna()
    titles_df = titles_df.drop_duplicates(subset=["product_id"])

    result: dict[str, dict[str, str]] = {}
    for _, row in titles_df.iterrows():
        product_id = str(row["product_id"])
        result[product_id] = {
            "title": str(row["product_title"]),
            "description": str(row["product_description"]),
        }
    return result

def load_shard(
    es: Elasticsearch, emb_path: Path, ids_path: Path, titles: dict[str, dict[str, str]]
) -> tuple[int, list[dict]]:
    vectors = np.load(emb_path).astype(np.float32)
    ids_df = pd.read_parquet(ids_path)

    if "product_id" not in ids_df.columns:
        raise ValueError(
            f"Expected a product_id column in {ids_path.name}; "
            f"found {ids_df.columns.tolist()}"
        )

    ids = ids_df["product_id"].astype(str).tolist()
    if len(vectors) != len(ids):
        raise ValueError(
            f"Shard mismatch in {emb_path.name}: "
            f"{len(vectors)} vectors vs {len(ids)} IDs"
        )
    if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Bad dimensions in {emb_path.name}: expected (?, "
            f"{EXPECTED_DIMENSIONS}), got {vectors.shape}"
        )

    def actions():
        for product_id, vector in zip(ids, vectors):
            product_info = titles.get(product_id, {"title": "", "description": ""})
            yield {
                "_index": INDEX_NAME,
                "_id": product_id,
                "_source": {
                    "product_id": product_id,
                    "title": product_info["title"],
                    "description": product_info["description"],
                    "embedding": vector.tolist(),
                },
            }

    return helpers.bulk(
        es.options(request_timeout=180, max_retries=3),
        actions(),
        chunk_size=500,
        raise_on_error=False,
    )


def main() -> None:
    es = Elasticsearch(ES_URL).options(request_timeout=120)
    if not es.ping():
        raise ConnectionError(f"Could not connect to Elasticsearch at {ES_URL}")

    emb_files = sorted(SHARD_DIR.glob("emb_*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No embedding shards found in {SHARD_DIR}")

    titles = load_titles()
    print(f"Loaded {len(titles)} titles from {TITLE_FILE}")
    loaded_shards = (
        set(STATE_FILE.read_text().splitlines()) if STATE_FILE.exists() else set()
    )

    for emb_path in tqdm(emb_files, desc="Loading embedding shards"):
        shard_id = emb_path.stem.removeprefix("emb_")
        if shard_id in loaded_shards:
            continue

        ids_path = SHARD_DIR / f"ids_{shard_id}.parquet"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing ID shard for {emb_path.name}: {ids_path}")

        success, errors = load_shard(es, emb_path, ids_path, titles)
        if errors:
            print(
                f"Shard {shard_id}: {success} indexed, "
                f"{len(errors)} failed; it will be retried next run"
            )
            print(errors[:3])
            continue

        with STATE_FILE.open("a") as state_file:
            state_file.write(shard_id + "\n")
        loaded_shards.add(shard_id)
        print(f"Shard {shard_id}: {success} indexed successfully")

    es.indices.refresh(index=INDEX_NAME)
    total = es.count(index=INDEX_NAME)["count"]
    untitled = es.count(index=INDEX_NAME, query={"term": {"title": ""}})["count"]
    print(f"\nIndexed documents: {total:,}")
    print(f"Documents with an empty title: {untitled:,}")
    print("Done. Re-run this script to retry any failed shards.")


if __name__ == "__main__":
    main()
