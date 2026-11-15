# Third-party sources and licenses

Upstream repositories are fetched separately at the commits below and are
ignored by Git. The original HEABench license does not replace upstream terms.
License texts were checked against the indicated upstream sources.

| Target | Upstream repository | Pinned commit | License text |
|---|---|---|---|
| CGCNN | [txie-93/cgcnn](https://github.com/txie-93/cgcnn) | `f42ab233c4ee0c416879d6bc2d22a264418413ad` | [MIT](licenses/CGCNN-LICENSE) |
| ALIGNN | [usnistgov/alignn](https://github.com/usnistgov/alignn) | `f2366daa3413d28a825b46e34d001b5549b05a40` | [NIST Terms of Use](licenses/ALIGNN-LICENSE) |
| ComFormer | [divelab/AIRS](https://github.com/divelab/AIRS) | `4a16c68a7da707c521019067dec51c227c10de45` | [GPLv3](licenses/ComFormer-AIRS-LICENSE) |
| EDBench | [HongxinXiang/EDBench](https://github.com/HongxinXiang/EDBench) | `89183fee107501589d2c2ab3c8be8ab54096e0f6` | [MIT](licenses/EDBench-LICENSE) |
| GeoFormer | [microsoft/AI2BMD](https://github.com/microsoft/AI2BMD) | `20ba4c1fe3045de862f03ddbb84c10811884f5f8` | [MIT](licenses/GeoFormer-LICENSE) |
| EquiformerV2 | [atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2) | `d5ad4be729b56f74012ebb7f097f77c5b00a1004` | [MIT](licenses/EquiformerV2-LICENSE) |
| PointVector/OpenPoints | [guochengqian/openpoints](https://github.com/guochengqian/openpoints) | `69fd158bc4951c0c99e976d1e210cba8935208de` | [MIT](licenses/OpenPoints-PointVector-LICENSE) |
| X-3D | [sunshuofeng/X-3D](https://github.com/sunshuofeng/X-3D) | `d1648995179391be17502bb6f5a9c5320ef318a3` | [MIT](licenses/X3D-LICENSE) |
| Pointcept/PTv3 | [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept) | `2b247ed555e96eae11710dfde79a3fc16c8c74b3` | [MIT](licenses/Pointcept-PTv3-LICENSE) |

Run `bash scripts/fetch_third_party.sh` to fetch these sources. Existing
directories are never reset: a checkout at a different commit stops the script.
Native extensions still need to be built for the active PyTorch/CUDA version.

## ComFormer / AIRS

Task 1 imports `third_party/ComFormer/OpenMat/ComFormer`. AIRS supplies a
GPLv3 license at the pinned repository root. The ComFormer `setup.py` lists
an MIT classifier but supplies no component-specific MIT license text at this
commit. This release therefore retains the actual repository GPLv3 text and
does not represent ComFormer as an MIT-only dependency. Distribution of a
combined GPL-covered program must comply with GPLv3; fetching upstream sources
separately does not remove those obligations.

## Architecture reference: MCDC

The charge encoder is implemented in HEABench's
[`mcdc_charge.py`](../src/property_prediction/models/mcdc_charge.py) using
PyTorch, periodic padding and adaptive pooling. It does not import the original
Keras training code. Its architecture reference is
[3D-CNN_Elastic_Constants](https://github.com/clemson-mcdc/3D-CNN_Elastic_Constants/tree/d946578abca6732f116ec2d1778995626d158ac1).
No license file is present in that upstream snapshot. The upstream training
source is not included or fetched by this release, and this notice grants no
permission to reuse it. The local adapter's implementation and the architecture
reference must not be confused with redistribution of that upstream source.

## Other integration notes

The Equiformer adapter uses the OCP compatibility tree bundled in EDBench.
Preserve notices supplied with nested upstream sources. PointVector and X-3D
require OpenPoints native operators. PTv3-HEA is a task-specific PyTorch
implementation inspired by Pointcept PTv3; the checkout is retained as a reference.

MACE and CHGNet are Python dependencies, not vendored code:

- [MACE upstream notice](licenses/MACE-LICENSE), copied from
  [ACEsuit/mace](https://github.com/ACEsuit/mace/blob/main/LICENSE.md).
  The upstream file is titled MIT License but contains only the copyright,
  permission and retention clauses; it does not include the standard MIT
  warranty paragraph. It is preserved as published, without adding terms on
  the upstream authors' behalf.
- [CHGNet license](licenses/CHGNet-LICENSE), copied from
  [CederGroupHub/chgnet](https://github.com/CederGroupHub/chgnet/blob/main/LICENSE),
  is BSD-3-Clause-LBNL and includes the upstream copyright and disclaimer.

Additional installed dependencies and downloaded weights retain their own
licenses. A software license must not be assumed to license model weights or
their training datasets.
