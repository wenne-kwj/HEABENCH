# Low-Label Magnetic Transfer

This task asks whether paired structure-field pretraining improves magnetic
moment prediction when fine-tuning labels are scarce. Downstream training and
inference consume atomic structure only.

The NSP pretraining split contains 1,595/199/199 parent structures. The
spin-polarized downstream split contains 620/133/133 parents, with frozen nested
training budgets of 50, 300, and 620. The compared conditions are:

- `s0`: randomly initialized structure encoder;
- `s_dft`: structure encoder initialized from structure-DFT pretraining;
- `s_gauss`: structure encoder initialized from structure-Gaussian pretraining
  (evaluated only at the fixed 50-label budget).

Pretrain and fine-tune with:

```bash
python -m hea_magnetic_transfer.pretrain --output-dir PRETRAIN_DFT
python -m hea_magnetic_transfer.sgauss_train \
  data/t3_magnetic_transfer/gaussian_view --output-dir PRETRAIN_GAUSS
python -m hea_magnetic_transfer.magnetic_train s_dft 50 \
  --checkpoint PRETRAIN_DFT/best.pt --output-dir TRANSFER_RUN
```

The fixed optimizer and budget settings are summarized in `config.json`.
S-Gauss uses the released periodic ZVAL Gaussian view (`sigma = 0.35 A`) and
the same EquiformerV2 structure and PointVector field branches as S-DFT.
Differences are attributable to the complete pretraining procedures under this
control, not to the matching objective alone.

The S-Gauss view is a lightweight mapping onto the frozen Gaussian field in the
Global Multimodal Probing bundle. Install the T1 and T3 bundles as sibling
directories under `data/`; the portable view contract resolves the shared field
with a relative path and does not contain workstation paths.
