import pytest
import pyarrow as pa
from memora.utils import (
    content_hash,
    short_hash,
)


class TestContentHashing:
    """Tests for utils/hashing.py."""

    def test_content_hash_basic(self):
        """Basic content hash should generate valid hash."""
        hash_val = content_hash(
            agent_id="test_agent",
            agent_version="v1.0.0",
            config={"model": "gpt-4"},
            input_data="test input",
        )

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_content_hash_deterministic(self):
        """Same parameters should generate same hash."""
        params = {
            "agent_id": "test",
            "agent_version": "v1",
            "config": {"param": "value"},
            "input_data": "input",
        }

        hash1 = content_hash(**params)
        hash2 = content_hash(**params)

        assert hash1 == hash2

    def test_content_hash_different_version(self):
        """Different version should change hash."""
        base_params = {
            "agent_id": "test",
            "config": {"param": "value"},
            "input_data": "input",
        }

        hash1 = content_hash(**base_params, agent_version="v1")
        hash2 = content_hash(**base_params, agent_version="v2")

        assert hash1 != hash2

    def test_content_hash_different_input(self):
        """Different input should change hash."""
        base_params = {"agent_id": "test", "agent_version": "v1", "config": {}}

        hash1 = content_hash(**base_params, input_data="input1")
        hash2 = content_hash(**base_params, input_data="input2")

        assert hash1 != hash2

    def test_content_hash_with_dependencies(self):
        """Hash with dependencies should include them."""
        hash1 = content_hash(
            agent_id="test", agent_version="v1", config={}, input_data="input"
        )

        hash2 = content_hash(
            agent_id="test",
            agent_version="v1",
            config={},
            input_data="input",
            dependencies=["dep1", "dep2"],
        )

        assert hash1 != hash2

    def test_content_hash_dependencies_order(self):
        """Order of dependencies should not affect hash."""
        base_params = {
            "agent_id": "test",
            "agent_version": "v1",
            "config": {},
            "input_data": "input",
        }

        hash1 = content_hash(**base_params, dependencies=["dep1", "dep2"])
        hash2 = content_hash(**base_params, dependencies=["dep2", "dep1"])

        assert hash1 == hash2  # Sorted internally

    def test_short_hash(self):
        """Short hash should return truncated version."""
        full_hash = "a" * 64

        short = short_hash(full_hash, length=8)

        assert len(short) == 8
        assert short == "aaaaaaaa"

    def test_short_hash_default(self):
        """Short hash with default length should be 12 chars."""
        full_hash = "b" * 64

        short = short_hash(full_hash)

        assert len(short) == 12
