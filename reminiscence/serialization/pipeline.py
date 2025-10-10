"""Transformation pipeline for compression and encryption."""

import time
from typing import List, Tuple, Callable, Dict
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Mapping of reverse operations to their forward counterparts
REVERSE_STAGE_MAP: Dict[str, str] = {
    "decrypt": "encrypt",
    "decompress": "compress",
}


class TransformationPipeline:
    """
    Composable pipeline for data transformations.

    Pipeline order: compress → encrypt
    Reverse order: decrypt → decompress

    All transformations work with bytes to maintain consistency.

    The pipeline maintains a clear mapping between forward and reverse operations:
    - compress ↔ decompress
    - encrypt ↔ decrypt
    """

    def __init__(self, compressor=None, encryptor=None):
        """
        Initialize pipeline with optional transformers.

        Args:
            compressor: Compression backend (must have compress/decompress methods)
            encryptor: Encryption backend (must have encrypt/decrypt methods)
        """
        self.compressor = compressor
        self.encryptor = encryptor

        self.forward_stages: List[Tuple[str, Callable[[bytes], bytes]]] = []
        self.reverse_stages: List[Tuple[str, Callable[[bytes], bytes]]] = []

        if compressor:
            self.forward_stages.append(("compress", compressor.compress))
            self.reverse_stages.insert(0, ("decompress", compressor.decompress))

        if encryptor:
            self.forward_stages.append(("encrypt", encryptor.encrypt))
            self.reverse_stages.insert(0, ("decrypt", encryptor.decrypt))

        logger.debug(
            "pipeline_initialized",
            stages=len(self.forward_stages),
            has_compressor=compressor is not None,
            has_encryptor=encryptor is not None,
        )

    def transform(self, data: bytes) -> Tuple[bytes, List[str]]:
        """
        Apply all transformations in forward order.

        Args:
            data: Raw bytes to transform

        Returns:
            Tuple of (transformed_bytes, list_of_applied_stage_names)

        Raises:
            TypeError: If data is not bytes
            Exception: If any transformation fails
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")

        current = data
        applied = []
        original_size = len(data)

        for stage_name, transform_fn in self.forward_stages:
            start = time.perf_counter()
            input_size = len(current)

            try:
                current = transform_fn(current)
                output_size = len(current)
                elapsed_ms = (time.perf_counter() - start) * 1000

                applied.append(stage_name)

                ratio = output_size / input_size if input_size > 0 else 0
                cumulative_ratio = (
                    output_size / original_size if original_size > 0 else 0
                )

                logger.debug(
                    f"{stage_name}_applied",
                    input_bytes=input_size,
                    output_bytes=output_size,
                    ratio=round(ratio, 3),
                    cumulative_ratio=round(cumulative_ratio, 3),
                    latency_ms=round(elapsed_ms, 2),
                )
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{stage_name}_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    input_bytes=input_size,
                    latency_ms=round(elapsed_ms, 2),
                )
                raise

        if applied:
            total_ratio = len(current) / original_size if original_size > 0 else 0
            logger.debug(
                "pipeline_transform_complete",
                applied_stages=applied,
                original_bytes=original_size,
                final_bytes=len(current),
                total_ratio=round(total_ratio, 3),
            )

        return current, applied

    def reverse(self, data: bytes, applied_transformations: List[str]) -> bytes:
        """
        Reverse all transformations in reverse order.

        Args:
            data: Transformed bytes
            applied_transformations: List of transformation names that were applied during forward pass

        Returns:
            Original bytes

        Raises:
            TypeError: If data is not bytes
            ValueError: If applied_transformations is invalid
            Exception: If any reverse transformation fails
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")

        if not isinstance(applied_transformations, list):
            raise ValueError(
                f"applied_transformations must be a list, got {type(applied_transformations).__name__}"
            )

        current = data
        original_size = len(data)
        reversed_count = 0

        for stage_name, reverse_fn in self.reverse_stages:
            # Map reverse stage to forward stage for checking
            forward_stage = REVERSE_STAGE_MAP.get(stage_name, stage_name)

            if forward_stage not in applied_transformations:
                logger.debug(
                    f"{stage_name}_skipped",
                    reason=f"forward stage '{forward_stage}' was not applied",
                    applied_transformations=applied_transformations,
                )
                continue

            start = time.perf_counter()
            input_size = len(current)

            try:
                current = reverse_fn(current)
                output_size = len(current)
                elapsed_ms = (time.perf_counter() - start) * 1000
                reversed_count += 1

                logger.debug(
                    f"{stage_name}_applied",
                    input_bytes=input_size,
                    output_bytes=output_size,
                    expansion_ratio=round(output_size / input_size, 3)
                    if input_size > 0
                    else 0,
                    latency_ms=round(elapsed_ms, 2),
                )
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{stage_name}_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    input_bytes=input_size,
                    latency_ms=round(elapsed_ms, 2),
                    applied_transformations=applied_transformations,
                )
                raise

        if reversed_count > 0:
            logger.debug(
                "pipeline_reverse_complete",
                reversed_stages=reversed_count,
                original_bytes=original_size,
                final_bytes=len(current),
            )

        return current

    def __repr__(self) -> str:
        """String representation of pipeline."""
        stages = []
        if self.compressor:
            stages.append(f"compress({self.compressor.algorithm})")
        if self.encryptor:
            stages.append("encrypt")

        if stages:
            return f"TransformationPipeline({' → '.join(stages)})"
        else:
            return "TransformationPipeline(empty)"
