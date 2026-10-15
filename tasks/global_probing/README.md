# Global Multimodal Probing

This task measures the predictive utility available to the tested models from
composition (`C`), atomic structure (`S`), and charge-density fields (`rho`).
Every non-composition condition retains the composition branch.

The frozen split contains 2,487/311/311 train/validation/test configurations.
Four target families are trained separately with train-set target
standardization and masked Smooth L1 loss: formation energy, electronic
scalars, DOS curves, and near-Fermi band energies. Model selection uses only
the validation split.

The frozen targets are:

| Target | Definition and unit |
|---|---|
| `E_form` | internal formation-energy proxy, eV/atom |
| `DOS_EF` | total DOS at the static Fermi energy, states/(eV atom) |
| `N_cross` | number of spin-resolved bands with a segment-local Fermi crossing, count |
| `N_event` | deduplicated adjacent-point crossing events within path segments, count |
| `cross_slope` | median absolute slope over deduplicated crossings, eV Angstrom |
| `DOS_curve` | 1,001 DOS values on the `E-E_F = [-10,10]` eV grid, states/(eV atom) |
| `Band_EF` | 16 near-Fermi bands at 80 resampled path positions, eV |

Crossing energies use the static Fermi reference. The band path is processed
segment by segment, so repeated high-symmetry endpoints do not create an
artificial crossing. The DOS and band masks retain only finite stored
components.

Train through the unified entry point:

```bash
python -m property_prediction.train CONDITION FAMILY --output-dir OUTPUT
```

`CONDITION` is one of `c`, `c_s_cgcnn`, `c_s_alignn`, `c_s_ecomformer`,
`c_rho_mcdc`, `c_rho_pointvector`, `c_s_rho`, or `c_s_gaussian`. `FAMILY` is
`formation`, `electronic_scalars`, `dos_curve`, or `band_structure`.

The Gaussian condition replaces the DFT field with a periodic atom-derived
Gaussian field while keeping the ALIGNN/MCDC fusion path and training protocol
fixed. It is a spatial-field control, not a complete isolation of electronic
information. Fixed data and model definitions are in `config.json`; the
condition-specific engines under `train/` implement the published runs.

## Paper comparison logic

The paper's offline comparisons use frozen predictions from the same split and
training seed. They do not require another training entry point. A compact
implementation should follow this logic:

```text
for each target and condition:
    align prediction, target, and mask by dataset_entry_id
    absolute_error[i] = mean(abs(prediction[i] - target[i]) over valid components)

percent_gain(reference, comparison) =
    100 * (metric(reference) - metric(comparison)) / metric(reference)

DFT_over_structure[i] = error[C+S][i] - error[C+S+rho][i]
Gaussian_over_structure[i] = error[C+S][i] - error[C+S+G][i]
DFT_over_Gaussian[i] = error[C+S+G][i] - error[C+S+rho][i]
```

Positive gain means that the comparison condition has lower error. The main
absolute MAE and RMSE pool all valid target components. The training loss uses
a different, deliberate reduction: DOS and band losses are averaged within
each configuration and then across configurations; the four electronic scalar
losses are averaged within each target and then across targets. This prevents
configurations with more finite vector components from receiving greater
training weight.

The environment analysis gives configurations equal weight by first averaging
absolute error over the valid components of each configuration. Consequently,
its overall DOS MAE need not equal the component-micro MAE in the main table
(for the published `C+S` result these are 0.09950 and 0.09809,
respectively). Every paired comparison uses one common validity mask, so the
different aggregation does not mix conditions within an effect estimate.

The auxiliary Gaussian NLL is calibrated once on the complete validation set.
For each target component, first divide residuals by that component's training
label standard deviation, then fit

```text
variance[j] = max(mean(validation_residual[:, j]^2), 1e-4)
NLL[i, j] = 0.5 * (log(2*pi) + log(variance[j])
                   + test_residual[i, j]^2 / variance[j])
NLL[i] = mean(NLL[i, valid components])
```

For the environment-conditioned comparison, compute the two train-set tertile
cut points of the frozen `delta_D12` table, apply those cut points to test
configurations, and use one common finite mask across compared conditions.
Subtract errors within each configuration before resampling. A paired percentile
bootstrap repeatedly samples configuration indices and reports the mean gain;
the high-minus-low contrast independently resamples the predefined high and low
groups and subtracts their mean gains. Validation-fitted NLL variances remain
fixed during this bootstrap. These intervals do not include training-seed or
calibration-parameter uncertainty.
