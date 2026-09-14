from pathlib import Path

import numpy as np
import pandas as pd


SHARD_DIR = Path(__file__).resolve().parent / "embeddings"
EXPECTED_EMBEDDING_SHAPE = (50_000, 1_024)
EXPECTED_EMBEDDING_DTYPE = np.float16


def main() -> None:
    emb_files = sorted(SHARD_DIR.glob("emb_*.npy"))
    ids_files = sorted(SHARD_DIR.glob("ids_*.parquet"))
    print(f"{len(emb_files)} emb files, {len(ids_files)} ids files")

    if not emb_files or not ids_files:
        raise FileNotFoundError(
            f"Expected emb_*.npy and ids_*.parquet files in {SHARD_DIR}"
        )

    embedding_sample = np.load(emb_files[0], mmap_mode="r")
    print(embedding_sample.shape, embedding_sample.dtype)
    if embedding_sample.shape != EXPECTED_EMBEDDING_SHAPE:
        print(f"Warning: expected shape {EXPECTED_EMBEDDING_SHAPE}")
    if embedding_sample.dtype != EXPECTED_EMBEDDING_DTYPE:
        print(f"Warning: expected dtype {EXPECTED_EMBEDDING_DTYPE}")

    ids_sample = pd.read_parquet(ids_files[0])
    print(ids_sample.columns.tolist(), ids_sample.shape)
    if "product_id" in ids_sample.columns:
        print("ID column: product_id")
    else:
        print("product_id was not found; use one of the columns listed above.")


if __name__ == "__main__":
    main()
