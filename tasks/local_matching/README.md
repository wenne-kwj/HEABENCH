# Local Structure-Charge Matching

This task trains paired structure and charge encoders with symmetric in-batch
InfoNCE. Validation and held-out evaluation retrieve the corresponding local
view in both structure-to-charge and charge-to-structure directions.

All patches from one parent structure remain in one split. The parent split is
1,957/245/244 for train/validation/test. Candidate sets are evaluation-only and
never enter the training objective.

Train any of the 4x4 encoder combinations with:

```bash
python -m hea_retrieval.train STRUCTURE CHARGE --output-dir OUTPUT
```

Structure choices are `geoformer`, `equiformer`, `mace`, and `chgnet`; charge
choices are `pointvector`, `x3d`, `ptv3`, and `lshce`. The small public protocol
is in `config.json`. `lshce_stats.json` contains the train-split physical
normalization required by LSHCE.

```bash
python -m hea_retrieval.evaluate OUTPUT/best.pt --output-dir EVALUATION
```

High retrieval under a controlled candidate set demonstrates discrimination
within that set; it does not by itself identify which geometric or electronic
cue the encoder used.
