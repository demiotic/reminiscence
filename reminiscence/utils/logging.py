"""Structured logging configuration for Reminiscence."""

import logging
import structlog
from datetime import datetime


# ANSI color codes
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "GRAY": "\033[90m",  # Gray (for timestamp)
    "RESET": "\033[0m",  # Reset
}


def _clean_text_renderer(logger, name, event_dict):
    """
    Custom renderer for clean, colorized text logs.

    Format: [LEVEL] HH:MM:SS - event_name | key=value key=value
    """
    # Extract core fields
    level = event_dict.pop("level", "info").upper()
    event = event_dict.pop("event", "")
    timestamp_iso = event_dict.pop("timestamp", None)

    # Format timestamp as HH:MM:SS
    if timestamp_iso:
        dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        timestamp = dt.strftime("%H:%M:%S")
    else:
        timestamp = datetime.now().strftime("%H:%M:%S")

    # Remove logger name
    event_dict.pop("logger", None)

    # Colorize level and timestamp
    level_color = COLORS.get(level, "")
    gray = COLORS["GRAY"]
    reset = COLORS["RESET"]

    # Build base: [LEVEL] HH:MM:SS - event |
    base = f"{level_color}[{level}]{reset} {gray}{timestamp}{reset} - {event}"

    # Add key=value pairs if they exist
    if event_dict:
        kvs = []
        for key, value in event_dict.items():
            # Skip redundant fields
            if key in {"query_preview", "matched_query_preview"}:
                # Only show matched_query if different from query
                if key == "matched_query_preview":
                    kvs.append(
                        f"matched={value[:50]}..."
                        if len(value) > 50
                        else f"matched={value}"
                    )
                continue

            # Format floats nicely
            if isinstance(value, float):
                if key.endswith("_ms"):
                    kvs.append(f"{key.replace('_ms', '')}={value:.1f}ms")
                elif key == "similarity":
                    kvs.append(f"sim={value:.3f}")
                elif key.endswith("_seconds"):
                    kvs.append(f"{key.replace('_seconds', '')}={value:.1f}s")
                else:
                    kvs.append(f"{key}={value:.2f}")
            else:
                # Truncate long strings
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                kvs.append(f"{key}={value}")

        return f"{base} | {' '.join(kvs)}" if kvs else base
    else:
        return base


def configure_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure structured logging for Reminiscence.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, output JSON format. If False, use clean colorized text.

    Note:
        This function can be called multiple times safely. It will reset
        the configuration each time to ensure consistent behavior.
    """
    # Reset structlog to ensure clean state
    structlog.reset_defaults()

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        force=True,
    )

    # Shared processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        # Production: JSON output
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: Clean colorized text format
        processors.append(_clean_text_renderer)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
