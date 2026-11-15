# Anonymous raw-data examples

Three complete sample archives are provided: `0001.tar.gz`, `0002.tar.gz`, and `0003.tar.gz`. Each extracts to a numbered directory with `metadata.json`, `relax/`, `static/`, and `band/`.

These archives include the available VASP output files, including the native static CHGCAR, structure files, energies, DOS, and band data. Optional files may differ between samples; see each metadata file. POTCAR files are not included.

Source identifiers, source paths, execution/build metadata, and archive ownership/timestamps have been removed or normalized. Numerical structure, charge-density, energy, and other scientific output values have been retained. `manifest.csv` lists archive sizes, file counts and spin settings.

The identifiers in these raw-data examples are separate from the shuffled `S0001`-style identifiers used in the structure gallery; they must not be joined by numeric suffix.

Extract an example with `tar -xzf 0001.tar.gz` and open the structure or charge-density files in a compatible viewer such as VESTA.

## Distribution and license

The archives are distributed as assets on [GitHub Releases](https://github.com/wenne-kwj/HEABENCH/releases).
They are excluded from the Git checkout. `manifest.csv`
identifies the prepared archives. Each archive contains `LICENSE-DATA.txt` and
`ATTRIBUTION.txt`; the data license is CC BY 4.0.
