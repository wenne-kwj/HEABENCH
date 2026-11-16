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

This repository accompanies an anonymous conference submission. The current
code corresponds to the experiments in the original manuscript and includes
reference loaders, training entry points, benchmark configurations, and frozen
metadata. Additional experiments conducted for the rebuttal have not yet been
integrated into this code release; the task documentation and configurations
therefore describe the original version.

During anonymous review, we provide this reference code, split metadata,
anonymized raw-data examples, and visualizations. Public download links for
the complete task-specific datasets, the full preprocessing workflow, and
environment installation instructions will be made available after anonymous
review has concluded.

## Data

The complete processed arrays are distributed separately, with public
download links deferred until anonymous review has concluded. Bundle names,
schemas, and split files are documented in [`data/registry.json`](data/registry.json)
and [`data/README.md`](data/README.md).

| Dataset | Train | Validation | Test | Split unit |
|---|---:|---:|---:|---|
| Global probing | 2,487 | 311 | 311 | configuration |
| Local matching | 1,957 | 245 | 244 | parent structure |
| Magnetic pretraining | 1,595 | 199 | 199 | parent structure |
| Magnetic transfer | 620 | 133 | 133 | parent structure |

All views and sites from one parent structure remain in the same split. Each
task uses its own frozen split.

## Examples and visualization

Complete sample archives and visualizations are collected in [`examples/`](examples):

- [Structure gallery](examples/structures.html): 3,109 relaxed structures; open the self-contained HTML locally.
- [NSP/SP comparison](examples/charge_density_comparison.png): download the [calculation package](examples/nsp_sp_example.tar.gz) for SP, fixed-structure NSP, and relaxed NSP calculations.
- Anonymized raw VASP examples: [0001](examples/samples/0001.tar.gz), [0002](examples/samples/0002.tar.gz), and [0003](examples/samples/0003.tar.gz), including native-grid charge density and available relaxation, static, and band outputs.

The PNG shows aligned relaxed-NSP charge density minus SP, including structural
relaxation effects; the fixed-structure NSP calculation is supplied separately.
Raw-example identifiers are independent of the shuffled gallery identifiers.

## Reproducibility

The task implementations describe the original manuscript experiments using
prepared task inputs. The full preprocessing workflow and environment
installation instructions will accompany the post-review public release.
Core comparison logic is kept explicit:

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

## Licenses and downloads

Original HEABench code is licensed under [MIT](LICENSE). Scientific data,
metadata and figures are licensed under [CC BY 4.0](LICENSE-DATA.txt).
See [LICENSING.md](LICENSING.md) for scope and [third-party notices](third_party/README.md)
for upstream terms, including ComFormer/AIRS GPLv3 and ALIGNN NIST terms.

The original complete sample archives are stored in `examples/` using Git LFS.
Use **Download raw file** on each GitHub file page to obtain the full archive,
or clone with Git LFS installed. The existing Release downloads are also available.
The [example index](examples/manifest.csv) lists sample paths and downloads.
The offline gallery and raw examples are preview materials; they do not replace
the full task-specific numerical bundles described in `data/registry.json`.
