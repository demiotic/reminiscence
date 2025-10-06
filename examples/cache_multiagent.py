"""Multi-agent system with Memora caching."""

import time
from memora import Memora, CacheConfig


class Agent:
    """Base agent with caching."""

    def __init__(self, name: str, cache: Memora):
        self.name = name
        self.cache = cache

    def ask(self, query: str) -> str:
        """Ask agent with automatic caching."""
        context = {
            "agent": self.name,
            "version": "1.0",
        }

        # Check cache
        result = self.cache.lookup(query, context)

        if result.is_hit:
            print(f"  [{self.name}] ✅ Cache HIT (similarity: {result.similarity:.3f})")
            return result.result

        # Cache miss - do expensive work
        print(f"  [{self.name}] ❌ Cache MISS - processing...")
        response = self._process(query)

        # Store in cache
        self.cache.store(query, context, response)
        return response

    def _process(self, query: str) -> str:
        """Override in subclass."""
        time.sleep(1)
        return f"Response from {self.name}"


class SQLAgent(Agent):
    """Agent that generates SQL queries."""

    def _process(self, query: str) -> str:
        time.sleep(1.5)  # Simulate LLM call
        return "SELECT * FROM sales WHERE date > '2024-01-01' ORDER BY amount DESC;"


class PythonAgent(Agent):
    """Agent that writes Python code."""

    def _process(self, query: str) -> str:
        time.sleep(1.5)  # Simulate LLM call
        return "def analyze_data(df):\n    return df.groupby('category').sum()"


class ResearchAgent(Agent):
    """Agent that searches documentation."""

    def _process(self, query: str) -> str:
        time.sleep(1)  # Simulate search
        return "Found docs: https://pandas.pydata.org/docs/user_guide/groupby.html"


def main():
    print("=" * 60)
    print("Multi-Agent System with Shared Cache")
    print("=" * 60)

    # Shared cache for all agents - fresh start each run
    cache = Memora(
        CacheConfig(
            db_uri="memory://",
            ttl_seconds=3600,
            enable_metrics=True,
            log_level="ERROR",
            similarity_threshold=0.70,
        )
    )

    # Create agents
    sql_agent = SQLAgent("SQL Agent", cache)
    python_agent = PythonAgent("Python Agent", cache)
    research_agent = ResearchAgent("Research Agent", cache)

    # Scenario: Multiple agents processing queries
    print("\n📝 Task: Build a sales dashboard\n")

    # SQL agent - Query 1
    print("1. SQL Agent - Get sales data:")
    start = time.time()
    sql_result = sql_agent.ask("Get all sales data for 2024")
    print(f"   SQL: {sql_result[:60]}...")
    print(f"   ⏱️  {time.time() - start:.2f}s\n")

    # SQL agent - Query 2 (similar, should hit cache)
    print("2. SQL Agent - Similar query:")
    start = time.time()
    sql_result2 = sql_agent.ask("Retrieve 2024 sales records")
    print(f"   SQL: {sql_result2[:60]}...")
    print(f"   ⏱️  {time.time() - start:.2f}s\n")

    # Python agent
    print("3. Python Agent - Analysis code:")
    start = time.time()
    py_result = python_agent.ask("Write code to analyze sales by category")
    print(f"   Code: {py_result[:60]}...")
    print(f"   ⏱️  {time.time() - start:.2f}s\n")

    # Research agent
    print("4. Research Agent - Find docs:")
    start = time.time()
    research_result = research_agent.ask("Find pandas groupby documentation")
    print(f"   Docs: {research_result}")
    print(f"   ⏱️  {time.time() - start:.2f}s\n")

    # Same research query - should hit cache
    print("5. Research Agent - Same query again:")
    start = time.time()
    research_result2 = research_agent.ask("Find pandas groupby documentation")
    print(f"   Docs: {research_result2}")
    print(f"   ⏱️  {time.time() - start:.2f}s\n")

    # Show statistics
    print("=" * 60)
    print("Cache Statistics")
    print("=" * 60)

    metrics = cache.metrics
    if metrics:
        total = metrics.hits + metrics.misses
        hit_rate = metrics.hits / total if total > 0 else 0
        time_saved = metrics.hits * 1.5  # Avg 1.5s per agent call

        print(f"Total queries: {total}")
        print(f"Cache hits: {metrics.hits}")
        print(f"Hit rate: {hit_rate:.1%}")
        print(f"Time saved: ~{time_saved:.1f}s")
        print(f"Cache entries: {cache.table.count_rows()}")

    # Show health
    print("\n" + "=" * 60)
    print("Cache Health")
    print("=" * 60)
    health = cache.health_check()
    print(f"Status: {health['status']}")
    print(f"Total entries: {health['metrics']['total_entries']}")


if __name__ == "__main__":
    main()
