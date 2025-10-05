# Utils Module

Utility functions for Memora. Each module is independent and focused.

## Structure

```
utils/
├── __init__.py           # Public exports
├── embeddings.py         # Vector operations (cosine similarity, etc.)
├── fingerprint.py        # Context fingerprinting and normalization
├── serialization.py      # Type-safe serialization (orjson + Arrow)
└── hashing.py            # Content-addressable hashing (future use)
```

## `embeddings.py`

Vector operations for semantic search.

### Functions

#### `cosine_similarity(vec1, vec2) -> float`

Computes cosine similarity between two L2-normalized vectors.

For unit vectors: `cosine_similarity = dot_product`

```python
from memora.utils import cosine_similarity

vec1 = [0.1, 0.2, 0.3]
vec2 = [0.15, 0.25, 0.28]

sim = cosine_similarity(vec1, vec2)
print(sim)  # 0.998 (very similar)
```

**Args:**
- `vec1`, `vec2`: Lists of floats or PyArrow arrays

**Returns:**
- Float in range [-1, 1] (typically [0, 1] for normalized vectors)

#### `cosine_similarity_batch(query_vec, candidate_vecs) -> List[float]`

Batch version for multiple candidates.

```python
query = [0.1, 0.2, 0.3]
candidates = [
    [0.15, 0.25, 0.28],
    [0.5, 0.1, 0.2],
    [-0.1, 0.9, 0.1],
]

similarities = cosine_similarity_batch(query, candidates)
# [0.998, 0.456, 0.234]
```

#### `euclidean_distance(vec1, vec2) -> float`

Euclidean L2 distance.

```python
dist = euclidean_distance([1, 2, 3], [4, 5, 6])
# 5.196...
```

#### `normalize_embedding(vec) -> List[float]`

L2 normalization (unit vector).

```python
vec = [3, 4]  # magnitude = 5
normalized = normalize_embedding(vec)
# [0.6, 0.8]  # magnitude = 1
```

---

## `fingerprint.py`

Context hashing with type normalization.

### Problem It Solves

Different representations of the same context should produce the same hash:

```python
ctx1 = {"timeout": 30, "retry": 3}
ctx2 = {"retry": 3, "timeout": 30.0}  # Different order, float vs int

# Both produce same fingerprint
create_fingerprint(ctx1) == create_fingerprint(ctx2)  # True
```

### Functions

#### `create_fingerprint(context: Dict[str, Any]) -> str`

Generates SHA256 hash of normalized context.

**Normalization rules:**
- Dict keys sorted alphabetically
- Floats that are integers → converted to int (30.0 → 30)
- Floats rounded to 10 decimals
- Lists maintain order (order matters)
- Sets converted to sorted lists

```python
from memora.utils import create_fingerprint

context = {
    "agent": "sql",
    "timeout": 30.0,
    "tools": ["search", "calculator"],
}

fingerprint = create_fingerprint(context)
# "a4d5a016dcc745a9e8f3b2c1..." (64 chars)
```

**Args:**
- `context`: Dictionary with any JSON-serializable values

**Returns:**
- SHA256 hex string (64 characters)

#### `fingerprint_matches(ctx1, ctx2) -> bool`

Helper to check if two contexts produce the same fingerprint.

```python
ctx1 = {"db": "prod", "timeout": 30}
ctx2 = {"timeout": 30.0, "db": "prod"}

assert fingerprint_matches(ctx1, ctx2)
```

### Implementation Details

**Why normalize?**

Without normalization:
```python
{"timeout": 30} != {"timeout": 30.0}  # Different hashes!
```

With normalization:
```python
create_fingerprint({"timeout": 30}) == \
create_fingerprint({"timeout": 30.0})  # Same hash
```

**Edge cases handled:**
- Empty context → `{"__default__": True}`
- Unicode strings → UTF-8 encoded
- Nested dicts → Recursive normalization
- Non-JSON types → Converted to string

---

## `serialization.py`

Type-safe, secure serialization without pickle.

### Strategy

1. **Small data** → orjson (fast, secure, cross-language)
2. **Large DataFrames (>10MB)** → Arrow IPC (zero-copy)
3. **Custom types** → Handler functions

### Functions

#### `serialize(data: Any) -> bytes`

Serializes Python objects to bytes.

```python
from memora.utils import serialize, deserialize

# Basic types
data = {"status": "ok", "items": [1, 2, 3]}
serialized = serialize(data)  # bytes

# DataFrames
import pandas as pd
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
serialized = serialize(df)

# Numpy arrays
import numpy as np
arr = np.array([1, 2, 3])
serialized = serialize(arr)
```

**Supported types:**
- Basic: str, int, float, bool, None, dict, list
- Pandas: DataFrame, Series
- Polars: DataFrame
- Numpy: ndarray (dtype/shape preserved)
- Nested combinations of above

**Returns:**
- bytes (orjson JSON or Arrow IPC)

**Raises:**
- `TypeError`: If type not serializable

#### `deserialize(data: bytes) -> Any`

Deserializes bytes back to Python objects.

```python
original = {"key": "value"}
roundtrip = deserialize(serialize(original))
assert roundtrip == original
```

**Args:**
- `data`: bytes from `serialize()`

**Returns:**
- Original Python object

#### `is_serializable(data: Any) -> bool`

Checks if an object can be serialized.

```python
assert is_serializable({"key": "value"})
assert is_serializable(pd.DataFrame())
assert not is_serializable(lambda x: x)  # Functions not supported
```

### Implementation Details

#### orjson Handler

For custom types in orjson:

```python
def _orjson_handler(obj):
    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "pandas_df",
            "data": obj.to_dict(orient="split"),
            "index_name": obj.index.name,
        }
    # ... other types
```

On deserialization:

```python
def _reconstruct_nested(obj):
    if isinstance(obj, dict) and "__type__" in obj:
        if obj["__type__"] == "pandas_df":
            return pd.DataFrame.from_dict(
                obj["data"], 
                orient="split"
            )
    # ... other types
```

#### Arrow IPC Path

Large DataFrames use Arrow IPC for efficiency:

```python
# Pandas → Arrow Table → IPC Stream
table = pa.Table.from_pandas(df)
sink = pa.BufferOutputStream()
with pa.ipc.new_stream(sink, table.schema) as writer:
    writer.write_table(table)

# Prefix to identify format
return b"__arrow_pandas__:" + sink.getvalue().to_pybytes()
```

#### Why No Pickle?

Pickle is avoided because:
1. **Security**: Arbitrary code execution risk
2. **Portability**: Python-only, version-dependent
3. **Debugging**: Binary format, hard to inspect

orjson + Arrow IPC provides:
- Cross-language compatibility
- Human-readable (JSON) or efficient (Arrow)
- Security (no code execution)

### Testing

See `tests/test_serialization.py` for comprehensive type coverage:

```bash
pytest tests/test_serialization.py -v
```

Tests include:
- Basic types roundtrip
- DataFrame preservation (pandas/polars)
- Numpy array dtype/shape preservation
- Nested structures
- Unicode handling
- Edge cases (empty, large, deep nesting)

---

## `hashing.py`

Content-addressable hashing for future step cache functionality.

### Functions

#### `content_hash(agent_id, agent_version, config, input_data, dependencies) -> str`

Generates deterministic hash for caching computation steps.

Similar to Docker layer hashing: `hash(agent + version + config + inputs + deps)`

```python
from memora.utils import content_hash, short_hash

hash_val = content_hash(
    agent_id="sql_agent",
    agent_version="v1.2.3",
    config={"db": "prod", "timeout": 30},
    input_data="SELECT * FROM sales",
    dependencies=["step1_hash", "step2_hash"]
)

print(hash_val)  # "a3f5b2c1d4e6..." (64 chars)
print(short_hash(hash_val, length=12))  # "a3f5b2c1d4e6" (Docker-style)
```

**Args:**
- `agent_id`: Agent identifier
- `agent_version`: Code version (semver or git hash)
- `config`: Agent configuration dict
- `input_data`: Input data (query, parameters, etc.)
- `dependencies`: List of upstream step hashes (optional)

**Returns:**
- SHA256 hex string (64 characters)

#### `short_hash(full_hash, length=12) -> str`

Truncates hash to Docker/Git style short form.

```python
short = short_hash("a3f5b2c1d4e6...", length=8)
# "a3f5b2c1"
```

#### `verify_content_hash(...) -> bool`

Validates a hash against given parameters.

```python
is_valid = verify_content_hash(
    provided_hash="a3f5b2c1...",
    agent_id="sql_agent",
    agent_version="v1.2.3",
    config={...},
    input_data="...",
)
```

#### `dependency_chain_hash(step_hashes) -> str`

Hashes a chain of dependencies for cascade invalidation.

```python
chain = ["hash1", "hash2", "hash3"]
chain_hash = dependency_chain_hash(chain)
# If any step changes, chain hash changes
```

### Use Case: Step Cache

Future feature for caching multi-step pipelines:

```python
# Step 1: Fetch data
step1_hash = content_hash(
    agent_id="fetcher",
    agent_version="v1.0.0",
    config={"source": "api"},
    input_data={"query": "..."},
)

# Step 2: Process (depends on step 1)
step2_hash = content_hash(
    agent_id="processor",
    agent_version="v2.1.0",
    config={"format": "json"},
    input_data={...},
    dependencies=[step1_hash]  # Cascade invalidation
)

# If step1 input/config changes → step1_hash changes → step2_hash changes
```

---

## Testing Utils

Each util module has comprehensive tests:

```bash
# All util tests
pytest tests/test_*

# Specific module
pytest tests/test_fingerprint.py -v
pytest tests/test_serialization.py -v
pytest tests/test_embeddings.py -v
```

Test coverage:

```bash
pytest --cov=src/utils --cov-report=html
open htmlcov/index.html
```

---

## Common Patterns

### Pattern 1: Type-Safe Caching

```python
from memora.utils import serialize, deserialize

# Store
data = {"complex": [1, 2, {"nested": True}]}
cached_bytes = serialize(data)

# Retrieve
restored = deserialize(cached_bytes)
assert restored == data
```

### Pattern 2: Context Grouping

```python
from memora.utils import create_fingerprint

# Group by normalized context
contexts = [
    {"db": "prod", "timeout": 30},
    {"timeout": 30.0, "db": "prod"},  # Same fingerprint
    {"db": "staging", "timeout": 30},  # Different fingerprint
]

fingerprints = [create_fingerprint(ctx) for ctx in contexts]
assert fingerprints[0] == fingerprints[1]  # Normalized to same
assert fingerprints[0] != fingerprints[2]  # Different context
```

### Pattern 3: DataFrame Handling

```python
import pandas as pd
from memora.utils import serialize, deserialize

# Large DataFrame → Arrow IPC
df = pd.DataFrame({"col": range(1_000_000)})
serialized = serialize(df)  # Uses Arrow IPC automatically

# Small DataFrame → orjson
small_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
serialized = serialize(small_df)  # Uses orjson

# Both deserialize the same way
restored = deserialize(serialized)
```

### Pattern 4: Vector Similarity Search

```python
from memora.utils import cosine_similarity_batch

query_embedding = [0.1, 0.2, 0.3, ...]

candidates = [
    [0.15, 0.25, 0.28, ...],
    [0.5, 0.1, 0.2, ...],
    # ... more candidates
]

similarities = cosine_similarity_batch(query_embedding, candidates)
best_idx = similarities.index(max(similarities))
best_match = candidates[best_idx]
```

---

## Performance Considerations

### Serialization

**orjson** is 2-10x faster than standard `json`:

```python
# Benchmark: 10k iterations
import json
import orjson

data = {"key": "value", "items": [1, 2, 3] * 100}

# json.dumps(): ~150ms
# orjson.dumps(): ~25ms
```

**Arrow IPC** for large DataFrames avoids copies:

```python
# 1M row DataFrame
df = pd.DataFrame({"col": range(1_000_000)})

# to_dict() → orjson: ~500ms, ~50MB
# to_arrow() → IPC: ~100ms, ~8MB (compressed)
```

### Fingerprinting

SHA256 hashing is fast even for large contexts:

```python
# 10k key-value pairs
large_context = {f"key_{i}": i for i in range(10_000)}

# create_fingerprint(): ~5ms
```

Normalization overhead is negligible (<1ms).

### Vector Operations

Cosine similarity on normalized vectors is just a dot product:

```python
# 384-dim vectors
vec1 = [random.random() for _ in range(384)]
vec2 = [random.random() for _ in range(384)]

# cosine_similarity(): ~0.02ms
# cosine_similarity_batch(100 candidates): ~2ms
```

---

## Extending Utils

### Adding Custom Serialization

Edit `serialization.py`:

```python
# In _orjson_handler()
if isinstance(obj, MyCustomClass):
    return {
        "__type__": "my_custom",
        "data": obj.to_dict(),
    }

# In _reconstruct_nested()
if obj_type == "my_custom":
    return MyCustomClass.from_dict(obj["data"])
```

### Adding Vector Operations

Edit `embeddings.py`:

```python
def manhattan_distance(vec1, vec2) -> float:
    """L1 distance."""
    return sum(abs(a - b) for a, b in zip(vec1, vec2))

def dot_product(vec1, vec2) -> float:
    """Raw dot product."""
    return sum(a * b for a, b in zip(vec1, vec2))
```

---

## Troubleshooting

### Serialization Errors

**Error**: `TypeError: Type X is not serializable`

**Solution**: Add handler for your type or convert to supported type:

```python
# Convert to dict before caching
result = my_custom_object.to_dict()
memora.store(query, context, result)
```

### Fingerprint Collisions

**Symptom**: Different contexts producing same hash (extremely rare)

**Debug**:

```python
ctx1 = {...}
ctx2 = {...}

fp1 = create_fingerprint(ctx1)
fp2 = create_fingerprint(ctx2)

print(fp1, fp2)
print(json.dumps(ctx1, sort_keys=True))
print(json.dumps(ctx2, sort_keys=True))
```

SHA256 collision probability is negligible (2^-256).

### Vector Similarity Issues

**Problem**: Expected matches not being found

**Debug**:

```python
from memora.utils import cosine_similarity, normalize_embedding

vec1 = [...]
vec2 = [...]

# Check if normalized
import math
mag1 = math.sqrt(sum(x**2 for x in vec1))
print(f"Magnitude: {mag1}")  # Should be ~1.0

# Check similarity
sim = cosine_similarity(vec1, vec2)
print(f"Similarity: {sim}")  # 0.0-1.0
```

---

## Dependencies

Utils modules have minimal external dependencies:

```python
# embeddings.py
# - No external deps (pure Python)

# fingerprint.py
# - hashlib (stdlib)
# - json (stdlib)

# serialization.py
# - orjson (required)
# - pyarrow (required)
# - pandas (optional)
# - polars (optional)
# - numpy (optional)

# hashing.py
# - hashlib (stdlib)
# - json (stdlib)
```

Install optional dependencies:

```bash
pip install pandas polars numpy
```

---

## API Reference Summary

### embeddings.py

```python
cosine_similarity(vec1, vec2) -> float
cosine_similarity_batch(query_vec, candidates) -> List[float]
euclidean_distance(vec1, vec2) -> float
normalize_embedding(vec) -> List[float]
```

### fingerprint.py

```python
create_fingerprint(context: Dict) -> str
fingerprint_matches(ctx1: Dict, ctx2: Dict) -> bool
```

### serialization.py

```python
serialize(data: Any) -> bytes
deserialize(data: bytes) -> Any
is_serializable(data: Any) -> bool
```

### hashing.py

```python
content_hash(agent_id, version, config, input_data, deps) -> str
short_hash(full_hash: str, length: int) -> str
verify_content_hash(...) -> bool
dependency_chain_hash(step_hashes: List[str]) -> str
```