---
title: RAG Pipelines
description: Caching Retrieval-Augmented Generation systems with Reminiscence
---

RAG (Retrieval-Augmented Generation) systems are expensive—both in latency and cost. Each query triggers document retrieval, embedding generation, vector search, and LLM inference. Reminiscence addresses this by caching the entire pipeline, but RAG presents unique challenges that most semantic caches can't handle.

## Why RAG Caching Is Different

Traditional semantic caches fail at RAG because RAG results aren't simple text strings. A typical RAG response includes:

- **Structured metadata**: Document IDs, relevance scores, timestamps (often as pandas DataFrames)
- **Complex outputs**: Answers with source citations, confidence scores, retrieved chunks
- **Multimodal content**: Images from documents, table data, charts
- **Large datasets**: Hundreds of source documents with metadata

**Other caches (gptcache, upstash) break here** because they only store JSON strings. If you return a DataFrame of source documents or a NumPy array of relevance scores, they'll serialize it to JSON—massive bloat, type loss, and slow retrieval.

**Reminiscence solves this** with Arrow format. Your DataFrame of sources? Stored in columnar format with 10-100x compression. NumPy relevance scores? Stored as Arrow arrays with zero-copy reads. This is why RAG systems need Reminiscence.

## The RAG Caching Challenge

Consider a typical RAG query: "What were Q3 2024 sales in the US region?"

**Without caching**, every query executes:
1. Embedding generation: 5-10ms
2. Vector search: 20-50ms
3. Document retrieval: 10-30ms
4. LLM inference: 1-3 seconds
5. **Total: 1-3 seconds, plus API costs**

**With Reminiscence**, similar queries like "Show me third quarter 2024 US revenue" hit the cache:
- Cache lookup: 5-15ms (semantic matching finds the similar query)
- Context verification: Instant (exact context match on region/year)
- Result retrieval: <1ms (Arrow format, zero-copy)
- **Total: 10-20ms, $0 API cost**

That's a **100x speedup** and **100% cost savings** on cache hits.

But here's the catch: RAG queries with the same text but different retrieved documents should NOT match. That's where layered caching and context isolation come in.

## Layered Caching Strategy

The most effective RAG caching strategy caches at multiple levels:

### Level 1: Cache the Entire Pipeline

Cache the complete RAG result, including answer and sources. This works when the same question with the same context returns the same documents.

```python
from reminiscence import Reminiscence
from openai import OpenAI
import chromadb

cache = Reminiscence()
client = OpenAI()
vector_db = chromadb.Client()

@cache.cached(query="query", context=["collection", "model", "top_k"])
def rag_query(
    query: str,
    collection: str,
    model: str = "gpt-4",
    top_k: int = 5
):
    """Complete RAG pipeline with caching"""
    # Retrieve relevant documents
    collection_obj = vector_db.get_collection(collection)
    results = collection_obj.query(
        query_texts=[query],
        n_results=top_k
    )

    # Extract documents
    documents = results["documents"][0]
    context = "\n\n".join(documents)

    # Generate answer
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": results["metadatas"][0]  # Metadata as dict
    }

# First call: Full pipeline execution
result1 = rag_query("What is quantum computing?", collection="physics")

# Similar query: Cache hit (semantic matching)
result2 = rag_query("Explain quantum computers", collection="physics")
```

**Why this works**: The decorator caches based on semantic similarity of the query plus exact context match (collection, model, top_k). "What is quantum computing?" semantically matches "Explain quantum computers" with ~0.91 similarity, so you get a cache hit.

**Key insight**: The `sources` metadata is stored efficiently as a dict. If you have a DataFrame of sources, Reminiscence stores it in Arrow format—no JSON bloat.

### Level 2: Cache Retrieval and Generation Separately

More sophisticated caching splits retrieval and generation. This allows reusing the same retrieved context for different questions.

```python
import hashlib

@cache.cached(query="query", context=["collection", "top_k"])
def retrieve_documents(query: str, collection: str, top_k: int = 5):
    """Cache document retrieval separately"""
    collection_obj = vector_db.get_collection(collection)
    results = collection_obj.query(
        query_texts=[query],
        n_results=top_k
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0]
    }

@cache.cached(
    query="question",
    context=["context_hash", "model"]
)
def generate_answer(question: str, context: str, context_hash: str, model: str = "gpt-4"):
    """Cache answer generation with context hash"""
    prompt = f"""Answer based on context:

Context:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def rag_pipeline(query: str, collection: str, model: str = "gpt-4"):
    """RAG with layered caching"""
    # Cached retrieval
    retrieval = retrieve_documents(query, collection, top_k=5)

    # Build context
    context = "\n\n".join(retrieval["documents"])
    context_hash = hashlib.sha256(context.encode()).hexdigest()[:16]

    # Cached generation (keyed by context hash)
    answer = generate_answer(query, context, context_hash, model)

    return {
        "answer": answer,
        "sources": retrieval["metadatas"]
    }

# Query 1: "What is quantum computing?" → retrieves docs → generates answer
result1 = rag_pipeline("What is quantum computing?", "physics")

# Query 2: "Quantum computing definition" → might retrieve SAME docs
# Retrieval cache hit → same context hash → generation cache hit!
result2 = rag_pipeline("Quantum computing definition", "physics")
```

**Why this works**: If two different questions retrieve the same documents, the `context_hash` matches, so the generation step hits the cache. You get partial caching even when queries differ.

**When to use this**: When your document collection is stable and different phrasings retrieve the same context. Common in FAQ systems and knowledge bases.

## Caching Structured RAG Results

This is where Reminiscence shines over competitors. Modern RAG systems return structured data—DataFrames, arrays, rich metadata—that JSON caches can't handle efficiently.

### Example: RAG with DataFrame Sources

```python
import pandas as pd

@cache.cached(query="query", context=["collection", "model"])
def rag_with_dataframe_sources(query: str, collection: str, model: str = "gpt-4"):
    """RAG that returns DataFrame of sources"""
    # Retrieve documents
    collection_obj = vector_db.get_collection(collection)
    results = collection_obj.query(query_texts=[query], n_results=5)

    # Build DataFrame of sources (metadata with scores)
    sources_df = pd.DataFrame({
        "doc_id": [m["id"] for m in results["metadatas"][0]],
        "title": [m["title"] for m in results["metadatas"][0]],
        "relevance_score": [1 - d for d in results["distances"][0]],  # Convert distance to score
        "chunk": results["documents"][0]
    })

    # Generate answer
    context = "\n\n".join(results["documents"][0])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources_df  # DataFrame stored in Arrow format!
    }

# First call: Full execution
result = rag_with_dataframe_sources("What is machine learning?", "ml_docs")

# Result contains a DataFrame
print(type(result["sources"]))  # <class 'pandas.core.frame.DataFrame'>
print(result["sources"].shape)  # (5, 4)

# Second similar query: Cache hit with DataFrame
result2 = rag_with_dataframe_sources("Explain ML concepts", "ml_docs")
# DataFrame retrieved instantly from Arrow storage - no JSON conversion!
```

**What happens under the hood:**
1. Reminiscence detects the DataFrame in the result
2. Stores it as an Arrow table (columnar format)
3. On retrieval, returns it as a DataFrame (zero-copy)
4. **No JSON serialization** → 10-100x smaller storage, 10x faster retrieval

**Other caches would**: Convert DataFrame to JSON (massive), lose column types, slow everything down.

## Multi-Collection RAG

RAG systems often search across multiple collections (papers, docs, wikis). Context isolation ensures each collection combination caches separately.

```python
@cache.cached(
    query="query",
    context=["collections_list", "model", "top_k_per_collection"]
)
def multi_collection_rag(
    query: str,
    collections: list[str],
    model: str = "gpt-4",
    top_k_per_collection: int = 3
):
    """Search across multiple collections"""
    all_documents = []
    all_sources = []

    # Retrieve from each collection
    for collection_name in collections:
        collection = vector_db.get_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=top_k_per_collection
        )

        all_documents.extend(results["documents"][0])
        all_sources.extend([
            {**meta, "collection": collection_name}
            for meta in results["metadatas"][0]
        ])

    # Generate from combined context
    context = "\n\n".join(all_documents)

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": all_sources
    }

# Cached per collection combination
result = multi_collection_rag(
    "Explain machine learning",
    collections=["cs_papers", "textbooks", "wikipedia"]
)
```

**Why context matters**: Searching `["cs_papers", "textbooks"]` vs `["cs_papers", "wikipedia"]` should cache separately—different collections = different results.

## Re-ranking with Caching

Re-ranking improves retrieval quality but adds latency. Caching the entire re-ranked pipeline saves significant time.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@cache.cached(
    query="query",
    context=["collection", "rerank_top_k"]
)
def rag_with_reranking(
    query: str,
    collection: str,
    initial_top_k: int = 20,
    rerank_top_k: int = 5,
    model: str = "gpt-4"
):
    """RAG with document re-ranking"""
    # Initial retrieval (broad)
    collection_obj = vector_db.get_collection(collection)
    results = collection_obj.query(
        query_texts=[query],
        n_results=initial_top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Re-rank with cross-encoder
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    # Sort by re-ranker score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:rerank_top_k]

    # Top documents after re-ranking
    top_documents = [documents[i] for i in ranked_indices]
    top_sources = [metadatas[i] for i in ranked_indices]

    # Generate answer
    context = "\n\n".join(top_documents)

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": top_sources
    }

# Entire re-ranking pipeline is cached
result = rag_with_reranking("What is transfer learning?", "ml_papers")
```

**Performance impact**: Re-ranking adds 50-100ms per query. Caching the final result saves this overhead on similar queries.

## Hybrid Search RAG

Combining semantic and keyword search improves recall. Caching prevents redundant searches.

```python
from elasticsearch import Elasticsearch

es = Elasticsearch()

@cache.cached(
    query="query",
    context=["collection", "index", "model", "alpha"]
)
def hybrid_rag(
    query: str,
    collection: str,  # Vector DB collection
    index: str,       # Elasticsearch index
    alpha: float = 0.5,  # Weight: 0=pure keyword, 1=pure semantic
    model: str = "gpt-4"
):
    """Hybrid search RAG (semantic + keyword)"""
    # Semantic search
    vector_results = vector_db.get_collection(collection).query(
        query_texts=[query],
        n_results=10
    )

    # Keyword search
    keyword_results = es.search(
        index=index,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"]
                }
            },
            "size": 10
        }
    )

    # Combine with weighted scores
    combined_docs = []

    # Add semantic results
    for i, doc in enumerate(vector_results["documents"][0]):
        combined_docs.append({
            "text": doc,
            "score": (1 - alpha) * (1 - vector_results["distances"][0][i]),
            "source": "semantic"
        })

    # Add keyword results
    for hit in keyword_results["hits"]["hits"]:
        combined_docs.append({
            "text": hit["_source"]["content"],
            "score": alpha * hit["_score"],
            "source": "keyword"
        })

    # Sort by combined score
    combined_docs.sort(key=lambda x: x["score"], reverse=True)
    top_docs = combined_docs[:5]

    # Generate answer
    context = "\n\n".join([doc["text"] for doc in top_docs])

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": top_docs
    }

# Hybrid search pipeline cached
result = hybrid_rag(
    "machine learning algorithms",
    collection="ml_docs",
    index="ml_text",
    alpha=0.5
)
```

**Why alpha is in context**: Different alpha values produce different results, so they should cache separately.

## Conversational RAG

Multi-turn conversations add complexity—conversation history affects retrieval and generation.

```python
import hashlib

@cache.cached(
    query="current_question",
    context=["collection", "conversation_hash", "model"]
)
def conversational_rag(
    current_question: str,
    conversation_history: list,
    collection: str,
    model: str = "gpt-4"
):
    """RAG with conversation context"""
    # Create conversation hash for context
    conv_hash = hashlib.sha256(
        str(conversation_history).encode()
    ).hexdigest()[:16]

    # Retrieve based on current question
    collection_obj = vector_db.get_collection(collection)
    results = collection_obj.query(
        query_texts=[current_question],
        n_results=5
    )

    context = "\n\n".join(results["documents"][0])

    # Build messages with history
    messages = conversation_history + [
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {current_question}"
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": results["metadatas"][0]
    }

# Conversation
history = []

# Turn 1
response1 = conversational_rag(
    "What is neural network?",
    conversation_history=history,
    collection="ml_docs"
)
history.append({"role": "user", "content": "What is neural network?"})
history.append({"role": "assistant", "content": response1["answer"]})

# Turn 2 (uses conversation context)
response2 = conversational_rag(
    "How does it learn?",  # "it" refers to neural network from history
    conversation_history=history,
    collection="ml_docs"
)
```

**Caching strategy**: Each turn caches based on the current question + conversation hash. Same conversation path → cache hit.

## Metadata Filtering

Filter retrieved documents by metadata (year, author, category). Context ensures filtered results cache separately.

```python
import json

@cache.cached(
    query="query",
    context=["collection", "filters_json", "model"]
)
def filtered_rag(
    query: str,
    collection: str,
    filters: dict,
    model: str = "gpt-4"
):
    """RAG with metadata filtering"""
    collection_obj = vector_db.get_collection(collection)

    # Query with metadata filters
    results = collection_obj.query(
        query_texts=[query],
        n_results=5,
        where=filters  # e.g., {"year": {"$gte": 2020}}
    )

    if not results["documents"][0]:
        return {"answer": "No relevant documents found.", "sources": []}

    context = "\n\n".join(results["documents"][0])

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": results["metadatas"][0]
    }

# Cache filtered queries separately
recent = filtered_rag(
    "machine learning advances",
    collection="papers",
    filters={"year": {"$gte": 2023}}
)

older = filtered_rag(
    "machine learning advances",
    collection="papers",
    filters={"year": {"$lt": 2020}}
)
```

**Why filters in context**: Same query with different filters retrieves different documents, so they must cache separately.

## Performance Monitoring

Track RAG performance to understand cache impact:

```python
import time
import logging

logger = logging.getLogger(__name__)

def rag_with_metrics(query: str, collection: str):
    """RAG with performance tracking"""
    start = time.perf_counter()

    # Check cache
    result = cache.lookup(query, {"collection": collection})

    if result.is_hit:
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "rag_cache_hit",
            query_preview=query[:50],
            latency_ms=round(latency_ms, 2),
            similarity=round(result.similarity, 3),
            age_seconds=round(result.age_seconds, 1)
        )

        return result.result

    # Cache miss - full pipeline
    retrieval_start = time.perf_counter()
    # ... retrieval logic
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    generation_start = time.perf_counter()
    # ... generation logic
    generation_ms = (time.perf_counter() - generation_start) * 1000

    total_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "rag_cache_miss",
        query_preview=query[:50],
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_ms=round(total_ms, 2)
    )

    return answer

# Monitor performance
result = rag_with_metrics("What is machine learning?", "ml_docs")
```

**Key metrics to track**:
- Hit rate: `cache.get_stats()['hit_rate']`
- Average similarity on hits: Helps tune threshold
- Latency reduction: Cache hit vs miss latency

## Best Practices

1. **Cache at the right level**: Full pipeline caching for stable collections, layered caching for dynamic data
2. **Use context wisely**: Include parameters that affect results (collection, filters, model), exclude high-cardinality values
3. **Store DataFrames directly**: Let Reminiscence handle Arrow serialization—don't convert to JSON manually
4. **Monitor similarity scores**: Low-similarity hits might indicate threshold tuning needed
5. **Consider conversation state**: Hash conversation history for conversational RAG
6. **Filter carefully**: Metadata filters should be in context to ensure correct cache isolation

## Next Steps

- [Multi-Agent Systems](/examples/multi-agent/) - Caching agent workflows
- [LLM Applications](/examples/llm-apps/) - General LLM caching patterns
- [Configuration](/guides/configuration/) - Optimize RAG performance
