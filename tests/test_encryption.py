"""Tests for encryption backends."""

import json
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
        data_bytes = json.dumps(data).encode("utf-8")

        encrypted = age_encryption.encrypt(data_bytes)
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > 0

        decrypted_bytes = age_encryption.decrypt(encrypted)
        assert isinstance(decrypted_bytes, bytes)

        decrypted = json.loads(decrypted_bytes.decode("utf-8"))
        assert decrypted == data

    def test_encrypt_decrypt_nested_data(self, age_encryption):
        """Should handle nested structures."""
        data = {
            "user": {"name": "john", "email": "john@example.com"},
            "tags": ["admin", "developer"],
            "metadata": {"created": "2025-10-10", "count": 42},
        }
        data_bytes = json.dumps(data).encode("utf-8")

        encrypted = age_encryption.encrypt(data_bytes)
        decrypted_bytes = age_encryption.decrypt(encrypted)
        decrypted = json.loads(decrypted_bytes.decode("utf-8"))

        assert decrypted == data

    def test_encrypt_decrypt_list(self, age_encryption):
        """Should handle list data."""
        data = [1, 2, 3, "test", {"nested": True}]
        data_bytes = json.dumps(data).encode("utf-8")

        encrypted = age_encryption.encrypt(data_bytes)
        decrypted_bytes = age_encryption.decrypt(encrypted)
        decrypted = json.loads(decrypted_bytes.decode("utf-8"))

        assert decrypted == data

    def test_encrypt_decrypt_string(self, age_encryption):
        """Should handle string data."""
        data = "sensitive information"
        data_bytes = data.encode("utf-8")

        encrypted = age_encryption.encrypt(data_bytes)
        decrypted_bytes = age_encryption.decrypt(encrypted)

        assert decrypted_bytes.decode("utf-8") == data

    def test_encrypt_decrypt_none(self, age_encryption):
        """Should handle None."""
        data = None
        data_bytes = json.dumps(data).encode("utf-8")

        encrypted = age_encryption.encrypt(data_bytes)
        decrypted_bytes = age_encryption.decrypt(encrypted)
        decrypted = json.loads(decrypted_bytes.decode("utf-8"))

        assert decrypted is None

    def test_encrypt_decrypt_raw_bytes(self, age_encryption):
        """Should handle raw binary data."""
        data_bytes = b"\x00\x01\x02\x03\xff\xfe\xfd"

        encrypted = age_encryption.encrypt(data_bytes)
        decrypted = age_encryption.decrypt(encrypted)

        assert decrypted == data_bytes

    def test_decrypt_with_public_key_raises(self, age_private_key, age_public_key):
        """Should raise if trying to decrypt with public key."""
        enc_public = AgeEncryption(key=age_public_key)
        enc_private = AgeEncryption(key=age_private_key)

        data = {"test": "data"}
        data_bytes = json.dumps(data).encode("utf-8")
        encrypted = enc_private.encrypt(data_bytes)

        with pytest.raises(DecryptionError, match="requires a private key"):
            enc_public.decrypt(encrypted)

    def test_decrypt_invalid_data_raises(self, age_encryption):
        """Should raise if decrypting invalid data."""
        invalid_encrypted = b"not-valid-encrypted-data"

        with pytest.raises(DecryptionError, match="decryption failed"):
            age_encryption.decrypt(invalid_encrypted)

    def test_encrypt_non_bytes_raises(self, age_encryption):
        """Should raise TypeError if encrypting non-bytes."""
        with pytest.raises(TypeError, match="Expected bytes"):
            age_encryption.encrypt({"dict": "data"})

        with pytest.raises(TypeError, match="Expected bytes"):
            age_encryption.encrypt("string")

        with pytest.raises(TypeError, match="Expected bytes"):
            age_encryption.encrypt(123)

    def test_decrypt_non_bytes_raises(self, age_encryption):
        """Should raise TypeError if decrypting non-bytes."""
        with pytest.raises(TypeError, match="Expected bytes"):
            age_encryption.decrypt("not-bytes")

    def test_encrypt_batch_empty_list(self, age_encryption):
        """Should handle empty batch."""
        result = age_encryption.encrypt_batch([])
        assert result == []

    def test_encrypt_batch_single_item(self, age_encryption):
        """Should handle single-item batch."""
        data = [{"id": 1}]
        data_bytes_list = [json.dumps(d).encode("utf-8") for d in data]

        encrypted = age_encryption.encrypt_batch(data_bytes_list)
        assert len(encrypted) == 1
        assert isinstance(encrypted[0], bytes)

        decrypted_bytes = age_encryption.decrypt_batch(encrypted)
        decrypted = [json.loads(d.decode("utf-8")) for d in decrypted_bytes]

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
        data_bytes_list = [json.dumps(d).encode("utf-8") for d in data]

        encrypted = age_encryption.encrypt_batch(data_bytes_list)
        assert len(encrypted) == len(data)
        assert all(isinstance(e, bytes) for e in encrypted)

        decrypted_bytes = age_encryption.decrypt_batch(encrypted)
        decrypted = [json.loads(d.decode("utf-8")) for d in decrypted_bytes]

        assert decrypted == data

    def test_encrypt_batch_preserves_order(self, age_encryption):
        """Should preserve order in batch operations."""
        data = [{"id": i} for i in range(20)]
        data_bytes_list = [json.dumps(d).encode("utf-8") for d in data]

        encrypted = age_encryption.encrypt_batch(data_bytes_list)
        decrypted_bytes = age_encryption.decrypt_batch(encrypted)
        decrypted = [json.loads(d.decode("utf-8")) for d in decrypted_bytes]

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
        data_bytes_list = [json.dumps(d).encode("utf-8") for d in data]

        encrypted = age_encryption.encrypt_batch(data_bytes_list)
        decrypted_bytes = age_encryption.decrypt_batch(encrypted)
        decrypted = [json.loads(d.decode("utf-8")) for d in decrypted_bytes]

        assert decrypted == data

    def test_encrypt_batch_with_raw_bytes(self, age_encryption):
        """Should handle raw binary data in batches."""
        data_bytes_list = [
            b"\x00\x01\x02",
            b"\xff\xfe\xfd",
            b"plain text",
            b"\x80\x90\xa0",
        ]

        encrypted = age_encryption.encrypt_batch(data_bytes_list)
        decrypted = age_encryption.decrypt_batch(encrypted)

        assert decrypted == data_bytes_list

    def test_decrypt_batch_with_public_key_raises(
        self, age_private_key, age_public_key
    ):
        """Should raise if trying to batch decrypt with public key."""
        enc_private = AgeEncryption(key=age_private_key)
        enc_public = AgeEncryption(key=age_public_key)

        data = [{"id": 1}, {"id": 2}]
        data_bytes_list = [json.dumps(d).encode("utf-8") for d in data]
        encrypted = enc_private.encrypt_batch(data_bytes_list)

        with pytest.raises(DecryptionError, match="requires a private key"):
            enc_public.decrypt_batch(encrypted)

    def test_decrypt_batch_empty_list(self, age_encryption):
        """Should handle empty batch for decryption."""
        result = age_encryption.decrypt_batch([])
        assert result == []

    def test_batch_encryption_correctness(self, age_encryption):
        """Batch encryption should work correctly and preserve order."""
        data = [{"id": i, "data": "x" * 100} for i in range(50)]
        data_bytes_list = [json.dumps(d).encode("utf-8") for d in data]

        # Batch encryption
        encrypted = age_encryption.encrypt_batch(data_bytes_list)

        # Verify all encrypted
        assert len(encrypted) == len(data)
        assert all(isinstance(e, bytes) for e in encrypted)
        assert all(len(e) > 0 for e in encrypted)

        # Batch decryption
        decrypted_bytes = age_encryption.decrypt_batch(encrypted)
        decrypted = [json.loads(d.decode("utf-8")) for d in decrypted_bytes]

        # Verify correctness and order
        assert decrypted == data
        for i, item in enumerate(decrypted):
            assert item["id"] == i

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
        data_bytes = json.dumps(data).encode("utf-8")

        encrypted1 = age_encryption.encrypt(data_bytes)
        encrypted2 = age_encryption.encrypt(data_bytes)

        # Ciphertext should be different (age uses random nonces)
        assert encrypted1 != encrypted2

        # But both decrypt to same plaintext
        decrypted1 = age_encryption.decrypt(encrypted1)
        decrypted2 = age_encryption.decrypt(encrypted2)
        assert decrypted1 == decrypted2 == data_bytes

    def test_large_data_encryption(self, age_encryption):
        """Should handle large data efficiently."""
        large_data = b"x" * 1_000_000  # 1MB

        encrypted = age_encryption.encrypt(large_data)
        decrypted = age_encryption.decrypt(encrypted)

        assert decrypted == large_data
        assert len(encrypted) > len(large_data)  # Encrypted is larger

    def test_empty_bytes_encryption(self, age_encryption):
        """Should handle empty bytes."""
        data_bytes = b""

        encrypted = age_encryption.encrypt(data_bytes)
        decrypted = age_encryption.decrypt(encrypted)

        assert decrypted == data_bytes
