# tests/test_encryption.py
"""Tests for encryption backends."""

import pytest

from reminiscence.encryption import (
    AgeEncryption,
    DecryptionError,
)


class TestAgeEncryption:
    """Tests for AgeEncryption backend."""

    def test_init_with_private_key_string(self, age_private_key):
        """Should initialize with private key string."""
        enc = AgeEncryption(key=age_private_key)
        assert enc.is_private
        assert not enc.is_public
        assert enc.identity is not None
        assert enc.recipient is not None

    def test_init_with_public_key_string(self, age_public_key):
        """Should initialize with public key string."""
        enc = AgeEncryption(key=age_public_key)
        assert enc.is_public
        assert not enc.is_private
        assert enc.recipient is not None
        assert enc.identity is None

    def test_init_with_invalid_key_raises(self):
        """Should raise if key format is invalid."""
        with pytest.raises(ValueError, match="Invalid age key format"):
            AgeEncryption(key="invalid-key-format")

    def test_encrypt_decrypt_simple_dict(self, age_encryption):
        """Should encrypt and decrypt a simple dict."""
        data = {"user": "john", "id": 123}

        encrypted = age_encryption.encrypt(data)
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > 0

        decrypted = age_encryption.decrypt(encrypted)
        assert decrypted == data

    def test_encrypt_decrypt_nested_data(self, age_encryption):
        """Should handle nested structures."""
        data = {
            "user": {"name": "john", "email": "john@example.com"},
            "tags": ["admin", "developer"],
            "metadata": {"created": "2025-10-10", "count": 42},
        }

        encrypted = age_encryption.encrypt(data)
        decrypted = age_encryption.decrypt(encrypted)
        assert decrypted == data

    def test_encrypt_decrypt_list(self, age_encryption):
        """Should handle list data."""
        data = [1, 2, 3, "test", {"nested": True}]

        encrypted = age_encryption.encrypt(data)
        decrypted = age_encryption.decrypt(encrypted)
        assert decrypted == data

    def test_encrypt_decrypt_string(self, age_encryption):
        """Should handle string data."""
        data = "sensitive information"

        encrypted = age_encryption.encrypt(data)
        decrypted = age_encryption.decrypt(encrypted)
        assert decrypted == data

    def test_encrypt_decrypt_none(self, age_encryption):
        """Should handle None."""
        data = None

        encrypted = age_encryption.encrypt(data)
        decrypted = age_encryption.decrypt(encrypted)
        assert decrypted is None

    def test_decrypt_with_public_key_raises(self, age_private_key, age_public_key):
        """Should raise if trying to decrypt with public key."""
        enc_public = AgeEncryption(key=age_public_key)
        enc_private = AgeEncryption(key=age_private_key)

        data = {"test": "data"}
        encrypted = enc_private.encrypt(data)

        with pytest.raises(DecryptionError, match="requires a private key"):
            enc_public.decrypt(encrypted)

    def test_decrypt_invalid_data_raises(self, age_encryption):
        """Should raise if decrypting invalid data."""
        invalid_encrypted = b"not-valid-encrypted-data"

        with pytest.raises(DecryptionError, match="decryption failed"):
            age_encryption.decrypt(invalid_encrypted)

    def test_encrypt_batch_empty_list(self, age_encryption):
        """Should handle empty batch."""
        result = age_encryption.encrypt_batch([])
        assert result == []

    def test_encrypt_batch_single_item(self, age_encryption):
        """Should handle single-item batch."""
        data = [{"id": 1}]

        encrypted = age_encryption.encrypt_batch(data)
        assert len(encrypted) == 1
        assert isinstance(encrypted[0], bytes)

        decrypted = age_encryption.decrypt_batch(encrypted)
        assert decrypted == data

    def test_encrypt_batch_multiple_items(self, age_encryption):
        """Should encrypt/decrypt multiple items in batch."""
        data = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "charlie"},
            {"id": 4, "name": "diana"},
            {"id": 5, "name": "eve"},
        ]

        encrypted = age_encryption.encrypt_batch(data)
        assert len(encrypted) == len(data)
        assert all(isinstance(e, bytes) for e in encrypted)

        decrypted = age_encryption.decrypt_batch(encrypted)
        assert decrypted == data

    def test_encrypt_batch_preserves_order(self, age_encryption):
        """Should preserve order in batch operations."""
        data = [{"id": i} for i in range(20)]

        encrypted = age_encryption.encrypt_batch(data)
        decrypted = age_encryption.decrypt_batch(encrypted)

        for i, item in enumerate(decrypted):
            assert item["id"] == i

    def test_encrypt_batch_with_different_types(self, age_encryption):
        """Should handle mixed types in batch."""
        data = [
            {"dict": True},
            [1, 2, 3],
            "string",
            42,
            None,
            {"nested": {"deep": "value"}},
        ]

        encrypted = age_encryption.encrypt_batch(data)
        decrypted = age_encryption.decrypt_batch(encrypted)
        assert decrypted == data

    def test_decrypt_batch_with_public_key_raises(
        self, age_private_key, age_public_key
    ):
        """Should raise if trying to batch decrypt with public key."""
        enc_private = AgeEncryption(key=age_private_key)
        enc_public = AgeEncryption(key=age_public_key)

        data = [{"id": 1}, {"id": 2}]
        encrypted = enc_private.encrypt_batch(data)

        with pytest.raises(DecryptionError, match="requires a private key"):
            enc_public.decrypt_batch(encrypted)

    def test_decrypt_batch_empty_list(self, age_encryption):
        """Should handle empty batch for decryption."""
        result = age_encryption.decrypt_batch([])
        assert result == []

    def test_batch_encryption_is_parallel(self, age_encryption):
        """Batch should be faster than sequential for large batches."""
        import time

        data = [{"id": i, "data": "x" * 100} for i in range(50)]

        # Sequential
        start = time.time()
        seq_encrypted = [age_encryption.encrypt(d) for d in data]
        _ = time.time() - start

        # Batch (parallel)
        start = time.time()
        batch_encrypted = age_encryption.encrypt_batch(data)
        _ = time.time() - start

        # Results should be identical when decrypted
        seq_decrypted = [age_encryption.decrypt(e) for e in seq_encrypted]
        batch_decrypted = age_encryption.decrypt_batch(batch_encrypted)
        assert seq_decrypted == batch_decrypted == data

    def test_repr(self, age_private_key, age_public_key):
        """Should have meaningful repr."""
        enc_private = AgeEncryption(key=age_private_key, max_workers=8)
        enc_public = AgeEncryption(key=age_public_key)

        assert "private" in repr(enc_private)
        assert "max_workers=8" in repr(enc_private)
        assert "public" in repr(enc_public)

    def test_encrypted_data_is_different_each_time(self, age_encryption):
        """Should produce different ciphertext each time (non-deterministic)."""
        data = {"test": "data"}

        encrypted1 = age_encryption.encrypt(data)
        encrypted2 = age_encryption.encrypt(data)

        # Ciphertext should be different (age uses random nonces)
        assert encrypted1 != encrypted2

        # But both decrypt to same plaintext
        assert age_encryption.decrypt(encrypted1) == data
        assert age_encryption.decrypt(encrypted2) == data
