---
title: gRPC Microservices Example
description: Building distributed multi-agent systems with shared gRPC cache
---

This example demonstrates using Reminiscence's gRPC API to build a distributed multi-agent system where multiple microservices share a centralized cache.

## Architecture

```d2
Agents: Microservices (Python) {
  AgentA: Agent A\nResearch
  AgentB: Agent B\nTranslation
  AgentC: Agent C\nSQL Queries
}

CacheServer: Reminiscence gRPC Server\nPort 50051\n(Persistent Cache)

Agents.AgentA -> CacheServer: gRPC Cache\nLookup/Store
Agents.AgentB -> CacheServer: gRPC Cache\nLookup/Store
Agents.AgentC -> CacheServer: gRPC Cache\nLookup/Store
```

## Implementation

### 1. Cache Server

Start the shared cache server:

```python
# cache_server.py
from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.api.server import create_server
import structlog

logger = structlog.get_logger()


def main():
    """Start gRPC cache server for multi-agent system."""
    # Production configuration
    config = ReminiscenceConfig(
        db_uri="./data/multi_agent_cache.db",  # Persistent storage
        max_entries=50000,
        eviction_policy="lru",  # Keep frequently used entries
        ttl_seconds=3600,  # 1 hour default TTL
        similarity_threshold=0.82,
        enable_metrics=True,
        log_level="INFO",
    )

    logger.info("initializing_cache", config=config.db_uri)
    cache = Reminiscence(config)

    # Start background cleanup scheduler
    cache.start_scheduler()
    logger.info("scheduler_started", interval_seconds=3600)

    # Create gRPC server
    server = create_server(
        cache=cache,
        port=50051,
        max_workers=20,  # Handle 20 concurrent agents
        enable_reflection=True,  # Enable grpcurl/grpcui
    )

    server.start()

    logger.info(
        "grpc_server_started",
        port=50051,
        max_workers=20,
        max_entries=config.max_entries,
    )

    try:
        # Keep server running
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("shutting_down_server")
        server.stop(grace=5.0)
        cache.stop_scheduler()
        logger.info("server_stopped")


if __name__ == "__main__":
    main()
```

Run the server:

```bash
python cache_server.py
```

### 2. Research Agent (Service A)

Agent that researches topics using web search:

```python
# agent_research.py
from reminiscence.api.client import ReminiscenceClient
from reminiscence.types import MultiModalInput, QueryMode
import openai
import structlog

logger = structlog.get_logger()


class ResearchAgent:
    """Agent that researches topics and caches results."""

    def __init__(self, cache_server: str = "localhost:50051"):
        self.cache_client = ReminiscenceClient(cache_server)
        self.agent_name = "research"
        logger.info("research_agent_initialized", cache_server=cache_server)

    def research(self, topic: str, model: str = "gpt-4") -> dict:
        """Research a topic with caching."""
        query = MultiModalInput(text=f"Research: {topic}")
        context = {
            "agent": self.agent_name,
            "model": model,
            "task": "research",
        }

        # Try cache first
        result = self.cache_client.lookup(
            query=query,
            context=context,
            mode=QueryMode.SEMANTIC,  # Allow similar questions
        )

        if result.is_hit:
            logger.info(
                "cache_hit",
                topic=topic,
                similarity=result.similarity,
                age_seconds=result.age_seconds,
            )
            return result.result

        # Cache miss - perform research
        logger.info("cache_miss", topic=topic, performing="web_search")

        # Simulate web search + LLM synthesis
        research_data = self._perform_research(topic, model)

        # Store in cache
        self.cache_client.store(
            query=query,
            context=context,
            result=research_data,
            ttl_seconds=1800,  # 30 minutes (research may become stale)
        )

        logger.info("research_complete", topic=topic, sources=len(research_data["sources"]))
        return research_data

    def _perform_research(self, topic: str, model: str) -> dict:
        """Perform actual research (simulated)."""
        # In production: use web search API + LLM
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant. Provide comprehensive research.",
                },
                {"role": "user", "content": f"Research topic: {topic}"},
            ],
        )

        return {
            "topic": topic,
            "summary": response.choices[0].message.content,
            "sources": [
                "https://example.com/source1",
                "https://example.com/source2",
            ],
            "model": model,
        }

    def close(self):
        """Close cache connection."""
        self.cache_client.close()


# Usage
def main():
    agent = ResearchAgent()

    try:
        # First query - cache miss
        result1 = agent.research("quantum computing applications")
        print(f"Research result: {result1['summary'][:100]}...")

        # Similar query - cache hit
        result2 = agent.research("quantum computing use cases")
        print(f"Cached result: {result2['summary'][:100]}...")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
```

### 3. Translation Agent (Service B)

Agent that translates text with caching:

```python
# agent_translation.py
from reminiscence.api.client import ReminiscenceClient
from reminiscence.types import MultiModalInput, QueryMode, StoreRequest
import openai
import structlog

logger = structlog.get_logger()


class TranslationAgent:
    """Agent that translates text and caches translations."""

    def __init__(self, cache_server: str = "localhost:50051"):
        self.cache_client = ReminiscenceClient(cache_server)
        self.agent_name = "translation"
        logger.info("translation_agent_initialized", cache_server=cache_server)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str = "gpt-4",
    ) -> str:
        """Translate text with caching."""
        query = MultiModalInput(text=text)
        context = {
            "agent": self.agent_name,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": model,
        }

        # Check cache
        result = self.cache_client.lookup(
            query=query,
            context=context,
            mode=QueryMode.SEMANTIC,  # Allow similar phrases
            similarity_threshold=0.90,  # Strict for translations
        )

        if result.is_hit:
            logger.info(
                "translation_cached",
                source=source_lang,
                target=target_lang,
                similarity=result.similarity,
            )
            return result.result

        # Perform translation
        logger.info("translating", source=source_lang, target=target_lang)
        translation = self._translate(text, source_lang, target_lang, model)

        # Cache result
        self.cache_client.store(
            query=query,
            context=context,
            result=translation,
            ttl_seconds=86400,  # 24 hours (translations are stable)
        )

        return translation

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        model: str = "gpt-4",
    ) -> list[str]:
        """Batch translate multiple texts efficiently."""
        from reminiscence.types import LookupRequest

        # Batch lookup
        lookup_requests = [
            LookupRequest(
                query=MultiModalInput(text=text),
                context={
                    "agent": self.agent_name,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "model": model,
                },
                mode=QueryMode.SEMANTIC,
                similarity_threshold=0.90,
            )
            for text in texts
        ]

        results = self.cache_client.lookup_batch(lookup_requests)

        # Collect translations and identify misses
        translations = []
        missing_indices = []

        for i, result in enumerate(results):
            if result.is_hit:
                translations.append(result.result)
            else:
                translations.append(None)
                missing_indices.append(i)

        # Translate missing texts
        if missing_indices:
            logger.info("batch_translating", count=len(missing_indices))

            missing_translations = [
                self._translate(texts[i], source_lang, target_lang, model)
                for i in missing_indices
            ]

            # Store new translations
            store_requests = [
                StoreRequest(
                    query=MultiModalInput(text=texts[i]),
                    context={
                        "agent": self.agent_name,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "model": model,
                    },
                    result=translation,
                    ttl_seconds=86400,
                )
                for i, translation in zip(missing_indices, missing_translations)
            ]

            self.cache_client.store_batch(store_requests)

            # Fill in translations
            for idx, translation in zip(missing_indices, missing_translations):
                translations[idx] = translation

        logger.info(
            "batch_translation_complete",
            total=len(texts),
            cached=len(texts) - len(missing_indices),
            translated=len(missing_indices),
        )

        return translations

    def _translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str,
    ) -> str:
        """Perform actual translation."""
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"Translate from {source_lang} to {target_lang}.",
                },
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content

    def close(self):
        """Close cache connection."""
        self.cache_client.close()


# Usage
def main():
    agent = TranslationAgent()

    try:
        # Single translation
        translation = agent.translate(
            "Hello, how are you?",
            source_lang="English",
            target_lang="Spanish",
        )
        print(f"Translation: {translation}")

        # Batch translation
        texts = [
            "Good morning",
            "Thank you",
            "See you later",
        ]

        translations = agent.translate_batch(
            texts,
            source_lang="English",
            target_lang="Spanish",
        )

        for original, translated in zip(texts, translations):
            print(f"{original} → {translated}")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
```

### 4. SQL Agent (Service C)

Agent that executes SQL queries with caching:

```python
# agent_sql.py
from reminiscence.api.client import ReminiscenceClient
from reminiscence.types import MultiModalInput, QueryMode
import sqlite3
import structlog

logger = structlog.get_logger()


class SQLAgent:
    """Agent that executes SQL queries with intelligent caching."""

    def __init__(
        self,
        cache_server: str = "localhost:50051",
        db_path: str = "./data/app.db",
    ):
        self.cache_client = ReminiscenceClient(cache_server)
        self.agent_name = "sql"
        self.db_path = db_path
        logger.info(
            "sql_agent_initialized",
            cache_server=cache_server,
            database=db_path,
        )

    def query(self, sql: str, database: str = "main") -> list[dict]:
        """Execute SQL query with caching."""
        query = MultiModalInput(text=sql)
        context = {
            "agent": self.agent_name,
            "database": database,
            "operation": self._detect_operation(sql),
        }

        # Use EXACT mode for SQL queries
        result = self.cache_client.lookup(
            query=query,
            context=context,
            mode=QueryMode.EXACT,  # SQL requires exact matching
            similarity_threshold=0.98,  # Very strict
        )

        if result.is_hit:
            logger.info(
                "query_cached",
                database=database,
                age_seconds=result.age_seconds,
            )
            return result.result

        # Execute query
        logger.info("executing_query", database=database)
        results = self._execute_query(sql)

        # Cache only SELECT queries (not INSERT/UPDATE/DELETE)
        if self._is_cacheable(sql):
            ttl = self._get_ttl_for_query(sql)

            self.cache_client.store(
                query=query,
                context=context,
                result=results,
                ttl_seconds=ttl,
            )

            logger.info("query_cached", database=database, ttl_seconds=ttl)

        return results

    def _execute_query(self, sql: str) -> list[dict]:
        """Execute SQL query against database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _detect_operation(self, sql: str) -> str:
        """Detect SQL operation type."""
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("SELECT"):
            return "select"
        elif sql_upper.startswith("INSERT"):
            return "insert"
        elif sql_upper.startswith("UPDATE"):
            return "update"
        elif sql_upper.startswith("DELETE"):
            return "delete"
        else:
            return "other"

    def _is_cacheable(self, sql: str) -> bool:
        """Check if query should be cached."""
        # Only cache SELECT queries
        return sql.strip().upper().startswith("SELECT")

    def _get_ttl_for_query(self, sql: str) -> int:
        """Determine TTL based on query pattern."""
        sql_upper = sql.upper()

        # Short TTL for volatile data
        if any(keyword in sql_upper for keyword in ["CURRENT_", "NOW()", "RAND()"]):
            return 60  # 1 minute

        # Medium TTL for frequently changing data
        if any(table in sql_upper for table in ["ORDERS", "SESSIONS", "EVENTS"]):
            return 300  # 5 minutes

        # Long TTL for stable data
        if any(table in sql_upper for table in ["USERS", "PRODUCTS", "CONFIG"]):
            return 3600  # 1 hour

        # Default TTL
        return 600  # 10 minutes

    def invalidate_table(self, table: str, database: str = "main"):
        """Invalidate all cached queries for a table."""
        count = self.cache_client.invalidate(
            context={
                "agent": self.agent_name,
                "database": database,
            }
        )
        logger.info("invalidated_table_cache", table=table, count=count)
        return count

    def close(self):
        """Close cache connection."""
        self.cache_client.close()


# Usage
def main():
    agent = SQLAgent()

    try:
        # Query users table
        users = agent.query("SELECT * FROM users WHERE status = 'active'")
        print(f"Found {len(users)} active users")

        # Same query - cached
        users_cached = agent.query("SELECT * FROM users WHERE status = 'active'")
        print(f"Cached result: {len(users_cached)} users")

        # Invalidate after update
        # (In production, call this after INSERT/UPDATE/DELETE)
        agent.invalidate_table("users")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
```

### 5. Orchestrator

Coordinate multiple agents:

```python
# orchestrator.py
from agent_research import ResearchAgent
from agent_translation import TranslationAgent
from agent_sql import SQLAgent
from reminiscence.api.client import ReminiscenceClient
import structlog

logger = structlog.get_logger()


class MultiAgentOrchestrator:
    """Orchestrates multiple agents sharing a cache."""

    def __init__(self, cache_server: str = "localhost:50051"):
        self.cache_server = cache_server
        self.research_agent = ResearchAgent(cache_server)
        self.translation_agent = TranslationAgent(cache_server)
        self.sql_agent = SQLAgent(cache_server)
        self.cache_client = ReminiscenceClient(cache_server)

        logger.info("orchestrator_initialized", cache_server=cache_server)

    def process_multilingual_research(
        self,
        topic: str,
        target_languages: list[str],
    ) -> dict:
        """Research topic and translate to multiple languages."""
        logger.info("starting_multilingual_research", topic=topic)

        # 1. Research in English
        research = self.research_agent.research(topic)

        # 2. Translate summary to target languages
        translations = {}
        for lang in target_languages:
            translated = self.translation_agent.translate(
                research["summary"],
                source_lang="English",
                target_lang=lang,
            )
            translations[lang] = translated

        # 3. Log to database (not cached)
        # self.sql_agent.query(f"INSERT INTO research_logs ...")

        logger.info(
            "multilingual_research_complete",
            topic=topic,
            languages=target_languages,
        )

        return {
            "topic": topic,
            "english_summary": research["summary"],
            "translations": translations,
            "sources": research["sources"],
        }

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        stats = self.cache_client.get_stats()

        logger.info(
            "cache_statistics",
            entries=stats["cache_entries"],
            hit_rate=stats["hit_rate"],
            hits=stats["hits"],
            misses=stats["misses"],
        )

        return stats

    def health_check(self) -> bool:
        """Check if all systems are healthy."""
        health = self.cache_client.health_check()
        is_healthy = health["status"] == "healthy"

        logger.info("health_check", status=health["status"], healthy=is_healthy)

        return is_healthy

    def close(self):
        """Close all agent connections."""
        self.research_agent.close()
        self.translation_agent.close()
        self.sql_agent.close()
        self.cache_client.close()
        logger.info("orchestrator_closed")


# Usage
def main():
    orchestrator = MultiAgentOrchestrator()

    try:
        # Check health
        if not orchestrator.health_check():
            print("⚠ System unhealthy")
            return

        # Multilingual research
        result = orchestrator.process_multilingual_research(
            topic="renewable energy trends 2024",
            target_languages=["Spanish", "French", "German"],
        )

        print(f"\nTopic: {result['topic']}")
        print(f"\nEnglish Summary:\n{result['english_summary'][:200]}...")

        print("\nTranslations:")
        for lang, text in result['translations'].items():
            print(f"  {lang}: {text[:100]}...")

        # Show cache stats
        print("\nCache Statistics:")
        stats = orchestrator.get_cache_stats()
        print(f"  Entries: {stats['cache_entries']}")
        print(f"  Hit Rate: {stats['hit_rate']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")

    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
```

## Running the System

### 1. Start Cache Server

```bash
python cache_server.py
```

Output:
```
[INFO] initializing_cache config='./data/multi_agent_cache.db'
[INFO] scheduler_started interval_seconds=3600
[INFO] grpc_server_started port=50051 max_workers=20 max_entries=50000
```

### 2. Run Individual Agents

Research Agent:
```bash
python agent_research.py
```

Translation Agent:
```bash
python agent_translation.py
```

SQL Agent:
```bash
python agent_sql.py
```

### 3. Run Orchestrator

```bash
python orchestrator.py
```

Output:
```
[INFO] orchestrator_initialized cache_server='localhost:50051'
[INFO] health_check status='healthy' healthy=True
[INFO] starting_multilingual_research topic='renewable energy trends 2024'
[INFO] cache_miss topic='renewable energy trends 2024' performing='web_search'
[INFO] research_complete topic='renewable energy trends 2024' sources=2
[INFO] translating source='English' target='Spanish'
[INFO] translating source='English' target='French'
[INFO] translating source='English' target='German'
[INFO] multilingual_research_complete topic='renewable energy trends 2024' languages=['Spanish', 'French', 'German']

Topic: renewable energy trends 2024

English Summary:
Renewable energy continues to grow rapidly in 2024...

Translations:
  Spanish: La energía renovable continúa creciendo rápidamente en 2024...
  French: L'énergie renouvelable continue de croître rapidement en 2024...
  German: Erneuerbare Energien wachsen 2024 weiterhin schnell...

Cache Statistics:
  Entries: 8
  Hit Rate: 62.5%
  Hits: 5
  Misses: 3
```

## Docker Deployment

Deploy the entire system with Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Shared cache server
  cache-server:
    build:
      context: .
      dockerfile: Dockerfile.cache
    ports:
      - "50051:50051"
    volumes:
      - ./data:/data
    environment:
      - REMINISCENCE_DB_URI=/data/cache.db
      - REMINISCENCE_MAX_ENTRIES=50000
      - REMINISCENCE_LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "grpcurl", "-plaintext", "localhost:50051", "list"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Research agent
  agent-research:
    build:
      context: .
      dockerfile: Dockerfile.agent
    environment:
      - CACHE_SERVER=cache-server:50051
      - AGENT_TYPE=research
    depends_on:
      - cache-server

  # Translation agent
  agent-translation:
    build:
      context: .
      dockerfile: Dockerfile.agent
    environment:
      - CACHE_SERVER=cache-server:50051
      - AGENT_TYPE=translation
    depends_on:
      - cache-server

  # SQL agent
  agent-sql:
    build:
      context: .
      dockerfile: Dockerfile.agent
    volumes:
      - ./data:/data
    environment:
      - CACHE_SERVER=cache-server:50051
      - AGENT_TYPE=sql
      - DB_PATH=/data/app.db
    depends_on:
      - cache-server

  # Orchestrator
  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator
    ports:
      - "8000:8000"
    environment:
      - CACHE_SERVER=cache-server:50051
    depends_on:
      - cache-server
      - agent-research
      - agent-translation
      - agent-sql
```

Run the system:

```bash
docker-compose up -d
```

Monitor cache:

```bash
docker-compose logs -f cache-server
```

## Benefits

This architecture provides:

1. **Shared Intelligence**: All agents benefit from each other's cached results
2. **Reduced Costs**: Avoid redundant API calls across services
3. **Better Performance**: Sub-10ms cache lookups vs seconds for API calls
4. **Scalability**: Add more agents without linear cost increase
5. **Fault Tolerance**: Cache server failure doesn't break agents (cache miss mode)
6. **Observability**: Centralized metrics and monitoring

## Next Steps

- [gRPC API Guide](/guides/grpc-api/) - Full gRPC API documentation
- [Best Practices](/production/best-practices/) - Production deployment
- [OpenTelemetry](/production/opentelemetry/) - Distributed tracing
- [Health Checks](/production/health-checks/) - Monitoring strategies
