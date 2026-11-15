# NSP/SP example

This example contains three calculations for Mn19Fe21Co22Cu21Zn17 (100 atoms), plus the supplied 10 Å × 10 Å charge-density comparison figure.

| Archive | Setting | Geometry and contents |
|---|---|---|
| `sample/sp.tar.gz` | SP, ISPIN=2 | Static calculation on the reference relaxed structure. |
| `sample/nsp_fixed.tar.gz` | NSP, ISPIN=1 | Static calculation at exactly the same input cell and positions as the SP reference. |
| `sample/nsp_relaxed.tar.gz` | NSP, ISPIN=1 | Relaxation from that input structure, followed by a static calculation; the final static files are in `nsp_relaxed/relax/energy/`. |

Each archive expands to its named directory and includes a regenerated file manifest. Scientific files include structures, INCAR/KPOINTS, CHGCAR, DOSCAR, EIGENVAL, OUTCAR, OSZICAR, and available XML/trajectory/projection files. Original empty intermediate outputs, where present, remain empty. All three variants use ENCUT=550 eV and a 1×1×1 k-point mesh.

## Figure

[PNG: charge-density comparison](png/charge_density_comparison.png)

The supplied figure shows (a) SP charge density, (b) charge density after NSP relaxation, rigidly aligned for display, and (c) the difference Δn = n_NSP,relaxed − n_SP. It is a 10 Å × 10 Å local comparison. The figure is associated with the relaxed NSP calculation, while the fixed-structure NSP calculation is provided separately. The figure therefore includes the effect of structural relaxation; it is not the fixed-geometry NSP−SP comparison.

The PNG image pixels, labels, color scales, and resolution are unchanged. Only ancillary metadata was removed.

## Anonymous release processing

Original source identifiers, machine/build timestamps, execution metadata, archive owner names, and source directory names have been removed or normalized. SYSTEM labels have been replaced by `HEABench`; other calculation settings and scientific values are preserved. Scheduler stdout/stderr and POTCAR files are excluded from the public package, consistent with the existing raw-data example release. Numerical VASP outputs remain unchanged apart from the specified descriptive metadata in INCAR, OUTCAR, and vasprun.xml.

The sample is independent of the numbered raw-data examples and the shuffled structure-gallery identifiers. `manifest.csv` lists the calculation settings, archive sizes and file counts. The single-download Release asset is `nsp_sp_example.tar.gz`.

## Distribution and license

Obtain `nsp_sp_example.tar.gz` from [GitHub Releases](https://github.com/wenne-kwj/HEABENCH/releases).
It expands to `nsp_sp/`, including the three sample archives, PNG and documentation. These calculation data and the figure are licensed under
CC BY 4.0; see `LICENSE-DATA.txt` and `ATTRIBUTION.txt`.
