# HEABench reference data layouts

The numerical dataset is released separately from this small code package.
This directory documents the prepared layouts consumed by the reference
loaders. `registry.json` records dataset versions, split sizes, and expected
directories, while `metadata/` contains sanitized schemas and indexes without
workstation or raw-calculation paths.

Expected layout after download:

```text
data/
  t1_property_prediction/
    manifest.csv
    labels.npz
    structures/shards/*.npz
    fields/...
  t2_retrieval/
    parent_manifest.csv
    patch_manifest.csv
    structures/*.npz
    charges/*.npz
  t3_magnetic_transfer/
    nsp_manifest.csv
    sp_manifest.csv
    nsp/...
    sp/...
    gaussian_view/...
```

The example loaders do not consume raw VASP trees directly. They illustrate
model use after labels, splits, spatial samples, and local patches have been
prepared. The primary dataset remains useful independently of these exact
views, and users may construct alternative preprocessing pipelines for their
own research questions.

The anonymous review package deliberately leaves external URLs unset in
`registry.json`. No account name or public dataset URL is embedded in this
repository.

## Reference loader inputs

The supplied examples expect the following model inputs and portable metadata:

| Paper task | Include | Exclude |
|---|---|---|
| Global Multimodal Probing | sanitized manifest, labels, structure shards, MCDC/PointVector fields and the frozen Gaussian field | raw VASP paths, checkpoints, predictions and analysis exports |
| Local Structure-Charge Matching | sanitized parent, patch, anchor and candidate manifests plus structure and charge shards | raw DFT paths, training logs and paper-analysis candidate sets |
| Low-Label Magnetic Transfer | sanitized NSP/SP manifests, site arrays, labels, schedules, splits, budgets and `gaussian_view` | raw source manifests, logs, checkpoints and temporary subsets |

`t3_magnetic_transfer/gaussian_view` is intentionally lightweight and reuses
`t1_property_prediction/fields/gaussian_zval_56x40x40_sigma0p35_v1`. Both
bundles must be installed as siblings to run the S-Gauss pretraining control.

When prepared archives are used, each archive should unpack to the directory
named by its `directory` entry in `registry.json`. The optional local installer
supports `.tar.gz` archives.

Install one locally supplied bundle without a network request:

```bash
python scripts/download_data.py --task t1_property_prediction --archive /path/to/bundle.tar.gz
```

The compact source release intentionally omits long internal raw-to-view
builders. The code is provided as a concise usage reference; creating new task
views from raw calculations is a separate dataset-processing workflow.
