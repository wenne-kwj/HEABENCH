# HEABench

HEABench is a multimodal dataset and benchmark for studying learning across
composition, atomic structure, and electronic representations in chemically
disordered alloys.

## Dataset

HEABench contains 3,109 quinary 100-atom configurations spanning 60 elements
and 1,033 five-element systems. It links:

- composition;
- relaxed atomic structures;
- three-dimensional charge-density fields;
- formation-energy and electronic-structure targets;
- a spin-polarized subset with magnetic labels.

## Benchmarks

### Global Multimodal Probing

Evaluates the predictive utility of composition, structure, charge density,
and their combinations for energetic and electronic targets.

### Local Structure-Charge Matching

Evaluates local correspondence between atomic environments and charge-density
patches, including controlled retrieval settings.

### Low-Label Magnetic Transfer

Evaluates whether paired structure-charge pretraining supports magnetic
prediction with limited labels and structure-only inference.

## Repository Status

This repository accompanies an anonymous conference submission. It contains
compact reference loaders, training entry points, benchmark configurations,
and frozen metadata.

## Data

Processed arrays are distributed separately. Bundle names, schemas, and split
files are documented in [`data/registry.json`](data/registry.json) and
[`data/README.md`](data/README.md).

| Dataset | Train | Validation | Test | Split unit |
|---|---:|---:|---:|---|
| Global probing | 2,487 | 311 | 311 | configuration |
| Local matching | 1,957 | 245 | 244 | parent structure |
| Magnetic pretraining | 1,595 | 199 | 199 | parent structure |
| Magnetic transfer | 620 | 133 | 133 | parent structure |

All views and sites from one parent structure remain in the same split. Each
task uses its own frozen split.

## Reproducibility

The task implementations are reference code for using the released data. Core
comparison logic is kept explicit:

- Global probing aligns predictions by sample ID and common masks. A paired
  gain is `error(reference) - error(comparison)`, so positive values indicate
  lower error. Auxiliary NLL is calibrated on validation data and fixed for
  test evaluation.
- Local matching uses symmetric contrastive training and bidirectional
  retrieval metrics under fixed candidate sets.
- Magnetic transfer compares random, DFT-pretrained, and Gaussian-pretrained
  structure encoders under frozen label budgets.

Task-specific commands and settings are provided in
[`tasks/global_probing`](tasks/global_probing),
[`tasks/local_matching`](tasks/local_matching), and
[`tasks/magnetic_transfer`](tasks/magnetic_transfer).
