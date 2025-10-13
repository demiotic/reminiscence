#!/usr/bin/env python3
"""
Benchmark: Compression Performance (Low-level)

Direct serializer benchmark without embeddings/cache.
Run with: python benchmarks/compression_direct.py
"""

import json
import statistics
import time

import pandas as pd

from reminiscence.compression import create_compressor
from reminiscence.serialization import ResultSerializer


class CompressionBenchmark:
    """Benchmark compression at serializer level."""

    def __init__(self):
        self.results = []

    def create_test_data(self, data_type: str, size: str) -> list:
        """Create test data of different types and sizes."""
        if data_type == "dict":
            if size == "small":
                return [
                    {"id": i, "name": f"user_{i}", "value": i * 10} for i in range(100)
                ]
            elif size == "medium":
                return [
                    {"id": i, "data": "x" * 1000, "nested": {"a": i, "b": i * 2}}
                    for i in range(100)
                ]
            else:  # large
                return [
                    {"id": i, "data": "x" * 10000, "list": list(range(100))}
                    for i in range(100)
                ]

        elif data_type == "dataframe":
            if size == "small":
                return pd.DataFrame({"col" + str(i): range(100) for i in range(5)})
            elif size == "medium":
                return pd.DataFrame({"col" + str(i): range(1000) for i in range(10)})
            else:  # large
                return pd.DataFrame({"col" + str(i): range(10000) for i in range(20)})

        elif data_type == "text":
            text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
            if size == "small":
                return text[:1000]
            elif size == "medium":
                return text * 10
            else:  # large
                return text * 100

    def run_benchmark(
        self, data_type: str, size: str, compression: str, iterations: int = 10
    ) -> dict:
        """Run benchmark for specific configuration."""
        print(f"\n{'=' * 70}")
        benchmark_label = (
            f"Benchmarking: {data_type.upper():12} | "
            f"{size.upper():6} | {compression.upper():4}"
        )
        print(benchmark_label)
        print(f"{'=' * 70}")

        # Create serializer with compression
        if compression == "none":
            compressor = None
        elif compression == "zstd":
            compressor = create_compressor("zstd", level=3)
        else:  # gzip
            compressor = create_compressor("gzip", level=6)

        serializer = ResultSerializer(compressor=compressor)

        # Create test data
        data = self.create_test_data(data_type, size)

        # Calculate original size
        if isinstance(data, pd.DataFrame):
            original_bytes = len(data.to_json().encode("utf-8"))
        elif isinstance(data, list):
            original_bytes = len(json.dumps(data).encode("utf-8"))
        else:
            original_bytes = len(data.encode("utf-8"))

        serialize_times = []
        deserialize_times = []
        compressed_sizes = []

        # Warmup
        serialized, type_desc = serializer.serialize(data)
        _ = serializer.deserialize(serialized, type_desc)

        for iteration in range(iterations):
            # Serialize
            start = time.perf_counter()
            serialized, type_desc = serializer.serialize(data)
            serialize_time = (time.perf_counter() - start) * 1000
            serialize_times.append(serialize_time)

            # Track compressed size
            import base64

            compressed_sizes.append(len(base64.b64decode(serialized.encode("ascii"))))

            # Deserialize
            start = time.perf_counter()
            _ = serializer.deserialize(serialized, type_desc)
            deserialize_time = (time.perf_counter() - start) * 1000
            deserialize_times.append(deserialize_time)

        # Calculate stats
        avg_serialize = statistics.mean(serialize_times)
        avg_deserialize = statistics.mean(deserialize_times)
        std_serialize = (
            statistics.stdev(serialize_times) if len(serialize_times) > 1 else 0
        )
        std_deserialize = (
            statistics.stdev(deserialize_times) if len(deserialize_times) > 1 else 0
        )
        avg_compressed_size = statistics.mean(compressed_sizes)
        compression_ratio = (
            avg_compressed_size / original_bytes if original_bytes > 0 else 1.0
        )

        result = {
            "data_type": data_type,
            "size": size,
            "compression": compression,
            "serialize_ms": avg_serialize,
            "serialize_std": std_serialize,
            "deserialize_ms": avg_deserialize,
            "deserialize_std": std_deserialize,
            "original_kb": original_bytes / 1024,
            "compressed_kb": avg_compressed_size / 1024,
            "compression_ratio": compression_ratio,
            "space_savings_pct": (1 - compression_ratio) * 100,
            "throughput_mb_s": (original_bytes / 1024 / 1024) / (avg_serialize / 1000)
            if avg_serialize > 0
            else 0,
        }

        serialize_stats = (
            f"  Serialize:    {result['serialize_ms']:.2f} ± "
            f"{result['serialize_std']:.2f} ms"
        )
        print(serialize_stats)
        deserialize_stats = (
            f"  Deserialize:  {result['deserialize_ms']:.2f} ± "
            f"{result['deserialize_std']:.2f} ms"
        )
        print(deserialize_stats)
        print(f"  Original:     {result['original_kb']:.2f} KB")
        print(f"  Compressed:   {result['compressed_kb']:.2f} KB")
        ratio_stats = (
            f"  Ratio:        {result['compression_ratio']:.3f} "
            f"({result['space_savings_pct']:.1f}% savings)"
        )
        print(ratio_stats)
        print(f"  Throughput:   {result['throughput_mb_s']:.1f} MB/s")

        self.results.append(result)
        return result

    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite."""
        configs = [
            ("dict", "small"),
            ("dict", "medium"),
            ("dict", "large"),
            ("dataframe", "small"),
            ("dataframe", "medium"),
            ("dataframe", "large"),
            ("text", "small"),
            ("text", "medium"),
            ("text", "large"),
        ]

        compressions = ["none", "zstd", "gzip"]

        print("\n" + "=" * 70)
        print(" COMPRESSION BENCHMARK - DIRECT SERIALIZER TEST")
        print("=" * 70)

        for data_type, size in configs:
            for compression in compressions:
                try:
                    self.run_benchmark(data_type, size, compression, iterations=10)
                except Exception as e:
                    print(f"  ERROR: {e}")

    def print_summary(self):
        """Print summary tables."""
        if not self.results:
            print("No results to display")
            return

        df = pd.DataFrame(self.results)

        print("\n\n" + "=" * 80)
        print(" SERIALIZE TIME (ms) - Lower is better")
        print("=" * 80)
        pivot = df.pivot_table(
            values="serialize_ms",
            index=["data_type", "size"],
            columns="compression",
            aggfunc="mean",
        )
        print(pivot.round(2))

        print("\n\n" + "=" * 80)
        print(" DESERIALIZE TIME (ms) - Lower is better")
        print("=" * 80)
        pivot = df.pivot_table(
            values="deserialize_ms",
            index=["data_type", "size"],
            columns="compression",
            aggfunc="mean",
        )
        print(pivot.round(2))

        print("\n\n" + "=" * 80)
        print(" COMPRESSION RATIO - Lower is better (more compression)")
        print("=" * 80)
        pivot = df.pivot_table(
            values="compression_ratio",
            index=["data_type", "size"],
            columns="compression",
            aggfunc="mean",
        )
        print(pivot.round(3))

        print("\n\n" + "=" * 80)
        print(" SPACE SAVINGS (%) - Higher is better")
        print("=" * 80)
        pivot = df.pivot_table(
            values="space_savings_pct",
            index=["data_type", "size"],
            columns="compression",
            aggfunc="mean",
        )
        print(pivot.round(1))

        print("\n\n" + "=" * 80)
        print(" THROUGHPUT (MB/s) - Higher is better")
        print("=" * 80)
        pivot = df.pivot_table(
            values="throughput_mb_s",
            index=["data_type", "size"],
            columns="compression",
            aggfunc="mean",
        )
        print(pivot.round(1))

        # Recommendations
        print("\n\n" + "=" * 80)
        print(" RECOMMENDATIONS")
        print("=" * 80)

        best_compression = df.groupby(["data_type", "size"]).apply(
            lambda x: x.loc[x["space_savings_pct"].idxmax()]
        )

        best_speed = df.groupby(["data_type", "size"]).apply(
            lambda x: x.loc[x["serialize_ms"].idxmin()]
        )

        print("\n  Best Compression (space savings):")
        for idx, row in best_compression.iterrows():
            if isinstance(idx, tuple):
                print(
                    f"    {idx[0]:12} {idx[1]:6} → {row['compression']:4} "
                    f"({row['space_savings_pct']:.1f}% savings)"
                )

        print("\n  Best Speed (serialize):")
        for idx, row in best_speed.iterrows():
            if isinstance(idx, tuple):
                print(
                    f"    {idx[0]:12} {idx[1]:6} → {row['compression']:4} "
                    f"({row['serialize_ms']:.2f}ms)"
                )

        # Save results
        df.to_csv("benchmark_compression_direct.csv", index=False)
        print("\n✓ Results saved to benchmark_compression_direct.csv")


def main():
    """Run benchmark."""
    benchmark = CompressionBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.print_summary()


if __name__ == "__main__":
    main()
