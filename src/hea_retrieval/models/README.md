# Retrieval encoder organization

```text
models/
  common.py             dual-encoder interface, pooling helpers, projection
  factory.py            one constructor for the complete 4x4 matrix
  structure/
    equiformer.py       upstream adapter
    geoformer.py        upstream adapter
    chgnet.py           upstream adapter
    mace.py             upstream adapter
  charge/
    pointvector.py      upstream-derived adapter
    x3d.py              upstream-derived adapter
    ptv3_hea.py         task-specific local implementation
    lshce.py            task-specific local implementation
```

Adapters expose a pooled patch representation and an independent normalized
projection. Upstream repositories remain pinned and unmodified. Lazy imports
in `models/__init__.py` prevent incompatible optional dependency stacks from
being imported together unless a selected experiment requires them.
