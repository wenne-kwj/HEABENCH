#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from property_prediction.data import BenchmarkIndex
from property_prediction.graphs import periodic_neighbor_slots
from property_prediction.spec import load_benchmark_spec
from property_prediction.storage import validate_heavy_artifact_path


DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed 6 A structure cache.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(
        spec=DEFAULT_SPEC, cutoff=6.0, max_neighbors=32, limit=None, execute=True
    )
    args = parser.parse_args()
    spec = load_benchmark_spec(args.spec)
    output_dir = validate_heavy_artifact_path(args.output_dir, spec)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty cache directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = BenchmarkIndex(spec)
    count = spec.population if args.limit is None else min(args.limit, spec.population)
    neighbor_index = np.lib.format.open_memmap(
        output_dir / "neighbor_index.npy", mode="w+", dtype=np.uint16,
        shape=(count, 100, args.max_neighbors),
    )
    neighbor_distance = np.lib.format.open_memmap(
        output_dir / "neighbor_distance.npy", mode="w+", dtype=np.float32,
        shape=(count, 100, args.max_neighbors),
    )
    neighbor_count = np.lib.format.open_memmap(
        output_dir / "neighbor_count.npy", mode="w+", dtype=np.uint8,
        shape=(count, 100),
    )
    started = time.time()
    minimum_count = args.max_neighbors
    for position in range(count):
        record = index.record(position)
        with np.load(record.structure_path, allow_pickle=False) as payload:
            if payload["z"].shape != (100,):
                raise ValueError(f"non-100-atom structure: {record.dataset_entry_id}")
            nbr_idx, nbr_dist, nbr_count = periodic_neighbor_slots(
                payload["frac"], payload["cell"], cutoff=args.cutoff,
                max_neighbors=args.max_neighbors,
            )
        neighbor_index[position] = nbr_idx
        neighbor_distance[position] = nbr_dist
        neighbor_count[position] = nbr_count
        minimum_count = min(minimum_count, int(nbr_count.min()))
        if position == 0 or (position + 1) % 250 == 0:
            print(json.dumps({"built": position + 1, "count": count,
                              "elapsed_sec": time.time() - started}), flush=True)
    neighbor_index.flush(); neighbor_distance.flush(); neighbor_count.flush()
    metadata = {
        "status": "complete",
        "dataset": str(spec.benchmark_root),
        "population": count,
        "full_population": spec.population,
        "cutoff_A": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "minimum_observed_neighbor_count": minimum_count,
        "ordering": "manifest_and_labels_npz_order",
        "seed": spec.seed,
        "elapsed_sec": time.time() - started,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
