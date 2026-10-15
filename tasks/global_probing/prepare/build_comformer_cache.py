#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from property_prediction.comformer_data import periodic_neighbor_vector_slots
from property_prediction.data import BenchmarkIndex
from property_prediction.spec import load_benchmark_spec
from property_prediction.storage import validate_heavy_artifact_path


TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[1]
DEFAULT_SPEC = TASK_ROOT / "config.json"
REFERENCE_CACHE = Path(
    os.environ.get(
        "HEABENCH_T1_ARTIFACT_ROOT",
        REPOSITORY_ROOT / "artifacts/t1_property_prediction",
    )
) / "cache/structure_cgcnn_r6_n32_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed eComFormer vector cache.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(
        spec=DEFAULT_SPEC, reference_cache=REFERENCE_CACHE, cutoff=6.0,
        max_neighbors=32, limit=None, execute=True,
    )
    args = parser.parse_args()
    spec = load_benchmark_spec(args.spec)
    output = validate_heavy_artifact_path(args.output_dir, spec)
    reference = validate_heavy_artifact_path(args.reference_cache, spec)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    index = BenchmarkIndex(spec)
    count = spec.population if args.limit is None else min(args.limit, spec.population)
    reference_index = np.load(reference / "neighbor_index.npy", mmap_mode="r")
    reference_distance = np.load(reference / "neighbor_distance.npy", mmap_mode="r")
    arrays = {
        "neighbor_index": np.lib.format.open_memmap(
            output / "neighbor_index.npy", mode="w+", dtype=np.uint16,
            shape=(count, 100, args.max_neighbors),
        ),
        "neighbor_vector": np.lib.format.open_memmap(
            output / "neighbor_vector.npy", mode="w+", dtype=np.float32,
            shape=(count, 100, args.max_neighbors, 3),
        ),
        "neighbor_count": np.lib.format.open_memmap(
            output / "neighbor_count.npy", mode="w+", dtype=np.uint8,
            shape=(count, 100),
        ),
    }
    started = time.time(); maximum_distance_delta = 0.0
    for position in range(count):
        record = index.record(position)
        with np.load(record.structure_path, allow_pickle=False) as payload:
            nbr_index, nbr_distance, nbr_vector, nbr_count = periodic_neighbor_vector_slots(
                payload["frac"], payload["cell"], cutoff=args.cutoff,
                max_neighbors=args.max_neighbors,
            )
        if not np.array_equal(nbr_index, reference_index[position]):
            raise ValueError(f"neighbor index differs from CGCNN cache: {record.dataset_entry_id}")
        distance_delta = float(np.max(np.abs(nbr_distance - reference_distance[position])))
        maximum_distance_delta = max(maximum_distance_delta, distance_delta)
        if distance_delta > 1e-5:
            raise ValueError(f"neighbor distance differs from CGCNN cache: {record.dataset_entry_id}")
        arrays["neighbor_index"][position] = nbr_index
        arrays["neighbor_vector"][position] = nbr_vector
        arrays["neighbor_count"][position] = nbr_count
        if position == 0 or (position + 1) % 250 == 0:
            print(json.dumps({"built": position + 1, "count": count,
                              "elapsed_sec": time.time() - started}), flush=True)
    for array in arrays.values():
        array.flush()
    metadata = {
        "status": "complete",
        "population": count,
        "full_population": spec.population,
        "dataset": str(spec.benchmark_root),
        "cutoff_A": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "edge_vector_definition": "source atom to selected periodic image of neighbor",
        "selection_matches_cache": str(reference),
        "maximum_distance_delta_A": maximum_distance_delta,
        "ordering": "manifest_and_labels_npz_order",
        "elapsed_sec": time.time() - started,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
