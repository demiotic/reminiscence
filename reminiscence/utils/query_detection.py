"""Query type detection for auto mode."""

import re


def should_use_exact_mode(query: str) -> bool:
    """
    Detect if query should use exact matching or semantic.

    Exact mode is used for:
    - SQL queries
    - API endpoints/URLs
    - Code snippets
    - Structured data (JSON, etc)
    - Hashes/IDs

    Semantic mode is used for:
    - Natural language questions
    - LLM prompts
    - Conversational text

    Args:
        query: Query text to analyze

    Returns:
        True if exact mode should be used, False for semantic
    """
    if not query or len(query.strip()) == 0:
        return False

    query_stripped = query.strip()

    # 1. SQL queries (case insensitive)
    sql_pattern = r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|WITH)\b"
    if re.match(sql_pattern, query_stripped, re.IGNORECASE):
        return True

    # 2. URLs and API endpoints
    if query_stripped.startswith(("http://", "https://", "www.")):
        return True

    # 3. API paths
    if query_stripped.startswith("/") and "/" in query_stripped[1:]:
        return True

    # 4. Code snippets (common keywords)
    code_keywords = [
        "def ",
        "class ",
        "function ",
        "import ",
        "from ",
        "const ",
        "let ",
        "var ",
        "async ",
        "await ",
        "public ",
        "private ",
        "protected ",
        "void ",
    ]
    if any(keyword in query_stripped for keyword in code_keywords):
        return True

    # 5. Structured data (JSON, XML-like)
    if query_stripped.startswith(("{", "[", "<")):
        return True

    # 6. Hash-like or ID-like (high density of non-alphabetic chars)
    if len(query_stripped) > 0:
        alpha_chars = sum(c.isalpha() for c in query_stripped)
        total_chars = len(query_stripped)
        alpha_ratio = alpha_chars / total_chars

        # If less than 50% alphabetic, likely a hash/ID/structured
        if alpha_ratio < 0.5:
            return True

    # 7. Command-line like
    if query_stripped.startswith(("$", ">", "#", "cmd", "bash", "sh")):
        return True

    # Default: use semantic for natural language
    return False
