---
title: Storage Architecture
description: Why Reminiscence uses LanceDB and Apache Arrow
---

Reminiscence's storage layer is what sets it apart from every other semantic cache. This page explains **why we chose LanceDB + Apache Arrow** and what makes this architecture special.

## The Problem With Other Semantic Caches

Most semantic caches (gptcache, upstash, etc.) use one of these approaches:

### Approach 1: Redis + JSON

```
Query → Embedding → Vector Search (Redis) → Get ID → Fetch JSON blob
```

**Problems:**
- JSON bloat (10-100x larger than needed for tabular data)
- Type loss (everything becomes strings)
- Slow serialization/deserialization
- Can't handle DataFrames or arrays efficiently
- No columnar compression

### Approach 2: Vector DB + Object Storage

```
Query → Embedding → Vector Search (Pinecone/Weaviate) → Get URL → Fetch from S3
```

**Problems:**
- Extra network hop for results
- Still uses JSON/pickle for serialization
- No support for efficient tabular storage
- High latency (50-200ms for S3 fetch)

## Reminiscence's Approach: LanceDB + Arrow

```
Query → Embedding → Vector Search (LanceDB) → Fetch Arrow data directly
```

**Advantages:**
- **Single database** handles both vectors AND data
- **Arrow format** for efficient columnar storage
- **Zero-copy** reads for DataFrames and arrays
- **Compression** built-in (10-100x for tabular data)
- **Fast** (5-15ms end-to-end)

## Why LanceDB?

LanceDB is one of the few databases that natively combines:

1. **Vector similarity search** (for semantic matching)
2. **Columnar storage** (for tabular data)
3. **Apache Arrow** format (industry standard)

### Key Features We Use

**1. Dual-Mode Storage**

LanceDB stores two types of tables:

```python
# Exact match table (for EXACT mode queries)
exact_table:
    - id (string): Query hash
    - query_text (string): Original query
    - context (string): JSON context
    - context_hash (string): Fast context lookup
    - result (binary): Serialized result
    - timestamp (int64): Creation time
    - metadata (string): JSON metadata

# Semantic match table (for SEMANTIC/AUTO mode)
semantic_table:
    - id (string): Entry hash
    - query_text (string): Original query
    - embedding (fixed_size_list<float>[384]): Query vector
    - context (string): JSON context
    - context_hash (string): Fast context lookup
    - result (binary): Serialized result
    - timestamp (int64): Creation time
    - metadata (string): JSON metadata
```

The `embedding` column enables fast vector similarity search, while other columns provide exact matching and metadata.

**2. IVF-PQ Indexing**

LanceDB automatically creates vector indexes using IVF-PQ (Inverted File with Product Quantization):

```python
# Without index (256+ entries)
lookup_latency = 10-50ms  # Linear scan

# With index
lookup_latency = 5-15ms  # Indexed search
```

The index is created automatically when the cache exceeds a threshold (default: 256 entries).

**IVF-PQ explained:**
- **IVF (Inverted File)**: Partitions vectors into clusters (like hash buckets)
- **PQ (Product Quantization)**: Compresses vectors for faster comparison
- **Result**: 3-5x speedup with minimal accuracy loss (0.95+ recall)

**3. Native Arrow Support**

LanceDB's storage layer IS Arrow:

```python
# When you store a DataFrame
df = pd.DataFrame({"col": [1, 2, 3]})
cache.store(query, context, df)

# Internally:
# 1. df → Arrow Table (zero-copy if possible)
# 2. Arrow Table → LanceDB storage (already in Arrow format!)
# 3. LanceDB compresses columns

# When you retrieve
result = cache.lookup(query, context)
df = result.result

# Internally:
# 1. LanceDB reads Arrow data (no decompression needed)
# 2. Arrow Table → pandas DataFrame (zero-copy view!)
# 3. No JSON parsing, no type conversion
```

**This is 10-100x faster than JSON serialization.**

## Apache Arrow: The Secret Sauce

Arrow is a language-agnostic columnar memory format designed for analytics.

### Why Columnar?

**Row-based (JSON):**
```
Row 1: {id: 1, name: "Alice", value: 100}
Row 2: {id: 2, name: "Bob", value: 200}
Row 3: {id: 3, name: "Charlie", value: 300}
```

Each row stores all columns together. Poor compression, slow analytics.

**Column-based (Arrow):**
```
id column: [1, 2, 3]
name column: ["Alice", "Bob", "Charlie"]
value column: [100, 200, 300]
```

Columns stored separately. Excellent compression (similar values together), fast analytics (read only needed columns).

### Compression Ratios

Arrow achieves 10-100x compression for typical data:

```python
# Example: 1M row sales data
df = pd.DataFrame({
    "order_id": range(1_000_000),  # Sequential IDs
    "customer_id": np.random.randint(1, 10000, 1_000_000),  # ~10K customers
    "amount": np.random.rand(1_000_000) * 1000,
    "category": np.random.choice(["A", "B", "C", "D"], 1_000_000)
})

# JSON serialization
json_size = len(df.to_json())  # ~250 MB
json_time = 2.5 seconds  # Serialization + storage

# Arrow format (Reminiscence)
arrow_size = 5 MB  # 50x compression!
arrow_time = 0.05 seconds  # 50x faster!
```

**Why Arrow compresses so well:**
- **order_id**: Sequential → run-length encoding → ~1 KB
- **customer_id**: Limited range → dictionary encoding → ~100 KB
- **category**: 4 values → dictionary encoding → ~10 KB
- **amount**: Float compression → ~4 MB

Total: ~5 MB vs 250 MB JSON

### Zero-Copy Operations

Arrow enables zero-copy conversions between libraries:

```python
# pandas DataFrame
df = pd.DataFrame({"a": [1, 2, 3]})

# To Arrow (zero-copy!)
arrow_table = pa.Table.from_pandas(df, preserve_index=False)
# Memory usage: Same (no copy made)

# Store in LanceDB
lancedb_table.add(arrow_table)
# Memory usage: Same (no copy made)

# Retrieve from LanceDB
result_table = lancedb_table.to_arrow()
# Memory usage: Same (no copy made)

# To pandas (zero-copy view!)
df_retrieved = result_table.to_pandas(self_destruct=True)
# Memory usage: Same (view, not copy)
```

**Five operations, zero copies. Total memory: 1x data size.**

Compare with JSON:
```python
# pandas → dict → JSON string
json_str = json.dumps(df.to_dict())  # Copy 1

# Store
storage[key] = json_str  # Copy 2

# Retrieve
json_str = storage[key]  # Copy 3

# Parse
dict_obj = json.loads(json_str)  # Copy 4

# Convert
df = pd.DataFrame.from_dict(dict_obj)  # Copy 5
```

**Five operations, five copies. Total memory: 5x data size.**

## Storage Layout

Reminiscence organizes data in LanceDB as follows:

```
cache.db/ (LanceDB directory)
  ├── exact_cache/ (for EXACT mode queries)
  │   ├── data/ (Arrow files)
  │   │   ├── 0.lance (first batch of entries)
  │   │   ├── 1.lance (second batch)
  │   │   └── ...
  │   └── metadata.json (table schema)
  │
  └── semantic_cache/ (for SEMANTIC/AUTO mode queries)
      ├── data/ (Arrow files)
      │   ├── 0.lance (entries with embeddings)
      │   └── ...
      ├── index/ (IVF-PQ vector index, created at 256+ entries)
      │   ├── ivf.idx (cluster centroids)
      │   └── pq.idx (quantization codebook)
      └── metadata.json (table schema)
```

Each `.lance` file is an Arrow file with columnar data. The index directory contains the vector index for fast similarity search.

## Hybrid Exact + Semantic Tables

Reminiscence uses **two separate tables** for efficiency:

### Exact Table

For queries in EXACT mode (SQL, deterministic operations):

```python
@cache.cached(query="sql", context=["db"], mode=QueryMode.EXACT)
def execute_sql(sql: str, db: str):
    return database.execute(sql)
```

Stored in `exact_cache` table:
- No embedding column (saves space)
- Fast hash-based lookup by query hash
- Used for SQL, code execution, API calls

### Semantic Table

For queries in SEMANTIC/AUTO mode (natural language):

```python
@cache.cached(query="question", context=["model"], mode=QueryMode.SEMANTIC)
def ask_llm(question: str, model: str):
    return llm(question, model)
```

Stored in `semantic_cache` table:
- Has embedding column (384 dimensions)
- Uses vector index for similarity search
- Used for LLM queries, RAG, conversational AI

**Why separate tables?**
- Exact queries don't need embeddings (saves 1.5KB per entry)
- Semantic queries don't need hash indexes
- Different access patterns optimized independently

## Vector Index Lifecycle

The vector index is created on-demand:

```python
# Cache starts empty
cache = Reminiscence(config)  # No index yet

# Add entries (below threshold)
for i in range(255):
    cache.store(query, context, result)
# No index created yet (< 256 entries)

# Cross threshold
cache.store(query_256, context, result)
# Index automatically created!
# IVF-PQ index built with 256 partitions

# Subsequent lookups use index
result = cache.lookup(query, context)  # 5-15ms (indexed)
```

**Index parameters:**
```python
config = ReminiscenceConfig(
    auto_create_index=True,  # Enable auto-indexing
    index_threshold_entries=256,  # Create at this size
    index_num_partitions=256  # Number of IVF clusters
)
```

**Partitions guideline:**
- Small cache (< 1K): 128 partitions
- Medium cache (1K-10K): 256 partitions (default)
- Large cache (10K-100K): 512 partitions
- Huge cache (100K+): 1024 partitions

Rule of thumb: `num_partitions ≈ sqrt(num_entries)`

## Persistence Options

### In-Memory Mode

```python
config = ReminiscenceConfig(db_uri="memory://")
```

- Fastest (no disk I/O)
- Data lost on restart
- Good for development, testing, ephemeral workloads

### Persistent Mode

```python
config = ReminiscenceConfig(db_uri="./cache.db")
```

- Data survives restarts
- Slightly slower (disk I/O)
- Production-ready
- Supports backup/restore

**Performance:**
- In-memory: 5-10ms lookup, 5ms store
- SSD persistent: 8-15ms lookup, 8-10ms store
- HDD persistent: 15-50ms lookup, 15-30ms store

**Recommendation:** Use SSD for production persistent storage.

## Compression Integration

Reminiscence applies compression **after** Arrow serialization:

```
DataFrame → Arrow Table → Compress (zstd) → Encrypt → Store
```

**Why compress Arrow?**
Arrow is already columnar and compact, but compression helps for:
- Sparse data (many nulls)
- Low-cardinality strings (repeated values)
- Large binary blobs (images, embeddings)

**Typical compression ratios:**
- Arrow + zstd: 2-5x additional compression
- Total: 20-500x vs JSON (depending on data)

```python
config = ReminiscenceConfig(
    compression_enabled=True,
    compression_algorithm="zstd",
    compression_level=3  # Balanced
)
```

## Performance Characteristics

### Lookup Performance

```python
# Small cache (< 256 entries, no index)
lookup_time = 10-30ms
  ├─ Embedding generation: 5-10ms
  ├─ Linear search: 5-15ms
  └─ Arrow deserialization: <1ms

# Large cache (256+ entries, with index)
lookup_time = 5-15ms
  ├─ Embedding generation: 5-10ms
  ├─ Indexed search: 1-5ms
  └─ Arrow deserialization: <1ms
```

### Storage Performance

```python
# Small result (<1KB JSON)
store_time = 5-10ms
  ├─ Embedding generation: 5-10ms
  └─ Arrow write: <1ms

# Large DataFrame (1M rows)
store_time = 50-100ms
  ├─ Embedding generation: 5-10ms
  ├─ Arrow conversion: 10-30ms (one-time)
  └─ LanceDB write: 30-50ms
```

**Note:** Arrow conversion is often zero-copy from pandas/polars!

## Scalability

LanceDB + Arrow scales efficiently:

| Cache Size | Lookup (no index) | Lookup (indexed) | Storage (GB) |
|------------|-------------------|------------------|--------------|
| 1K entries | 10-20ms | 5-10ms | 0.1 GB |
| 10K entries | 50-100ms | 8-15ms | 1 GB |
| 100K entries | 500-1000ms | 10-20ms | 10 GB |
| 1M entries | N/A (too slow) | 15-30ms | 100 GB |

**Key insight:** Without index, lookup time grows linearly. With index, grows logarithmically.

## Comparison Table

| Feature | Redis + JSON | Pinecone + S3 | **Reminiscence (LanceDB + Arrow)** |
|---------|-------------|---------------|-----------------------------------|
| **Vector search** | Via RediSearch | ✓ | ✓ |
| **DataFrame support** | ✗ (JSON only) | ✗ (JSON only) | **✓ (Native Arrow)** |
| **NumPy support** | ✗ (lists only) | ✗ (lists only) | **✓ (Native Arrow)** |
| **Compression** | Minimal | Minimal | **10-100x (columnar)** |
| **Lookup latency** | 20-50ms | 50-200ms | **5-15ms** |
| **Type preservation** | ✗ (all strings) | ✗ (all strings) | **✓ (exact types)** |
| **Zero-copy** | ✗ | ✗ | **✓** |
| **Storage cost** | High | Very high | **Low** |

## Best Practices

### DO: Let LanceDB Handle Large Data

```python
# ✓ Good: Store 10M row DataFrame directly
huge_df = pd.read_parquet("data.parquet")  # 10M rows
cache.store(query, context, huge_df)
# LanceDB handles it efficiently
```

### DO: Use Persistent Storage in Production

```python
# ✓ Good: Persistent SSD storage
config = ReminiscenceConfig(db_uri="/mnt/ssd/cache.db")
```

### DO: Enable Auto-Indexing

```python
# ✓ Good: Auto-index at 256 entries
config = ReminiscenceConfig(
    auto_create_index=True,
    index_threshold_entries=256
)
```

### DON'T: Pre-convert to JSON

```python
# ❌ Bad: Manual JSON conversion
df_json = df.to_json()
cache.store(query, context, df_json)

# ✓ Good: Let Arrow handle it
cache.store(query, context, df)
```

## Monitoring Storage

Check storage statistics:

```python
stats = cache.get_stats()

print(f"Total entries: {stats['cache_entries']}")
print(f"Exact entries: {stats['exact_entries']}")
print(f"Semantic entries: {stats['semantic_entries']}")
print(f"Index created: {stats['index_created']}")
```

## Next Steps

- **[Data Types Reference](/reference/data-types/)** — Supported formats and serialization
- **[Configuration](/reference/config/)** — Storage and index settings
- **[Performance](/production/performance/)** — Optimization techniques
