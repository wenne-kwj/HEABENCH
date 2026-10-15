# Third-party sources

HEABench adapters keep upstream model repositories unmodified. The repositories
are fetched at pinned commits and ignored by Git.

| Target directory | Upstream | Commit |
|---|---|---|
| `third_party/CGCNN` | https://github.com/txie-93/cgcnn | `f42ab233c4ee0c416879d6bc2d22a264418413ad` |
| `third_party/ALIGNN` | https://github.com/usnistgov/alignn | `f2366daa3413d28a825b46e34d001b5549b05a40` |
| `third_party/ComFormer` | https://github.com/divelab/AIRS | `4a16c68940dd97a51f1dc32725ed36eed38322dd` |
| `third_party/ElasticConstants3DCNN` | https://github.com/clemson-mcdc/3D-CNN_Elastic_Constants | `d94657861aeb6fcdd7aee0398068c871658edd4b` |
| `third_party/EDBench` | https://github.com/HongxinXiang/EDBench | `89183fee107501589d2c2ab3c8be8ab54096e0f6` |
| `third_party/geoformer` | https://github.com/microsoft/AI2BMD | `20ba4c1fe3045de862f03ddbb84c10811884f5f8` |
| `third_party/equiformer_v2` | https://github.com/atomicarchitects/equiformer_v2 | `d5ad4be729b56f74012ebb7f097f77c5b00a1004` |
| `third_party/openpoints_pointvector` | https://github.com/guochengqian/openpoints | `69fd158bc4951c0c99e976d1e210cba8935208de` |
| `third_party/x3d` | https://github.com/sunshuofeng/X-3D | `d1648995179391be17502bb6f5a9c5320ef318a3` |
| `third_party/pointcept_ptv3` | https://github.com/Pointcept/Pointcept | `2b247ed555e96eae11710dfde79a3fc16c8c74b3` |

Run `bash scripts/fetch_third_party.sh` to materialize them. Existing
directories are never reset: a checkout at the wrong commit causes the script
to stop.

Task 1 imports ComFormer from `third_party/ComFormer/OpenMat/ComFormer` and the
MCDC charge encoder from `third_party/ElasticConstants3DCNN`. The Equiformer
adapter also uses the OCP compatibility tree bundled by the pinned EDBench
checkout. PointVector and X-3D require their OpenPoints native
point-neighbour operators to be built for the active PyTorch/CUDA ABI. PTv3-HEA
is a pure-PyTorch, task-specific implementation inspired by Pointcept PTv3; the
Pointcept checkout is retained for provenance and comparison.

MACE and CHGNet are normal Python dependencies and are not vendored. Upstream
license texts must be reviewed and copied under `third_party/licenses/` before
the public release; `LICENSE_PENDING.md` entries mark sources whose license
copy is not yet present. These upstream licenses do not license the original
HEABench code.
