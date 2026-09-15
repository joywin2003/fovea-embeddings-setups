# Fovea

Fovea loads product embeddings into a local Elasticsearch index. The large
datasets and generated embedding files are intentionally kept out of Git.

## Repository layout

```text
fovea/
├── embeddings_input/
│   └── shopping_queries_dataset/
│       ├── shopping_queries_dataset_products.parquet  # Required product data
│       ├── shopping_queries_dataset_examples.parquet  # Optional source data
│       └── shopping_queries_dataset_sources.csv       # Optional source data
├── embeddings_output/
│   ├── embeddings/
│   │   ├── emb_00000.npy
│   │   ├── ids_00000.parquet
│   │   ├── emb_00001.npy
│   │   ├── ids_00001.parquet
│   │   └── ...
│   ├── inspect_shards.py
│   ├── load_embeddings.py
│   └── shards_loaded.txt  # Local load state
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## What belongs in the Parquet files

### Product data

Keep this file at:

```text
embeddings_input/shopping_queries_dataset/shopping_queries_dataset_products.parquet
```

The loader requires these columns:

| Column | Purpose |
| --- | --- |
| `product_id` | Unique product identifier. Used to match IDs and Elasticsearch documents. |
| `product_title` | Product title added to the Elasticsearch document. |

Additional columns are allowed, but the loader only uses these two columns.
`product_id` values should be stable and should match the IDs in every embedding
shard.

### Embedding ID shards

Keep the ID files beside the embedding files:

```text
embeddings_output/embeddings/ids_00000.parquet
embeddings_output/embeddings/ids_00001.parquet
...
```

Each ID Parquet file must contain a `product_id` column. Its row order must
match the corresponding NumPy file exactly:

```text
emb_00000.npy  <->  ids_00000.parquet
emb_00001.npy  <->  ids_00001.parquet
```

The current loader expects each embedding file to contain a two-dimensional
array with shape `(50_000, 1_024)`. The checked-in inspection script expects
`float16`; the loader converts vectors to `float32` before sending them to
Elasticsearch. The number of rows in the ID Parquet file must equal the number
of vectors in its matching `.npy` file.

Do not rename or reorder either side of a shard pair. A mismatch causes the
loader to stop rather than silently indexing the wrong products.

## What is not stored in Git

The following paths are ignored by `.gitignore` because they are large,
machine-specific, generated, or contain local progress state:

- `embeddings_input/shopping_queries_dataset/` - downloaded/raw datasets,
	including the Parquet files described above.
- `embeddings_output/embeddings/` - generated `.npy` vectors and ID Parquet
	shards.
- `embeddings_output/shards_loaded.txt` - records shards already loaded into
	Elasticsearch on one machine.
- `.venv/`, `__pycache__/`, build output, and package metadata.

The Elasticsearch data volume is also local Docker state. It is not included in
the repository or in the Parquet files.

## Sharing the project

When sharing with a friend, send the Git repository plus the data separately.
They should place the files at the exact paths above. The data directory is
ignored, so copying it into the checkout will not make it appear in Git.

If the data is too large for a normal file transfer, use a shared drive or an
artifact/object store and document the download location and checksum. Do not
commit the Parquet or embedding files unless the repository's storage policy
has explicitly been changed to support large files.

## Setup and loading

Install the Python environment with `uv`, then start Elasticsearch:

```bash
uv sync
docker compose up -d
```

Check the first embedding/ID pair before loading everything:

```bash
uv run python embeddings_output/inspect_shards.py
```

Load all available shards:

```bash
uv run python embeddings_output/load_embeddings.py
```

The loader skips shard IDs listed in `embeddings_output/shards_loaded.txt`.
To load the same shards into a fresh Elasticsearch volume, remove that state
file and start with an empty index.

## Minimum data checklist

Before sharing or loading, confirm that:

- `shopping_queries_dataset_products.parquet` exists in the input directory.
- It contains `product_id` and `product_title`.
- Every `emb_XXXXX.npy` has a matching `ids_XXXXX.parquet`.
- Every ID Parquet contains `product_id`.
- Each pair has the same number of rows.
- Product IDs in the ID shards have matching product rows.
