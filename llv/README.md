# Lattice Lotka--Volterra application

The selected source preserves all 18 LLV Python modules and the frozen
specification unchanged. Cluster-specific Slurm files and pilot launchers
are excluded.

## Replot the initial and continuation diagnostics

The three saved inputs under `diag12k_llv/` are included. From this directory:

```bash
python llv_diag_figures.py --diag-dir diag12k_llv --out preview_figures
```

This draws the initial/continuation correlation and density figures using
12,000 trajectories per ensemble. It reads existing arrays, does not load a
model, and does not apply another pair-count conversion. NumPy and Matplotlib
suffice. `llv_figures.py` and `llv_spec.py` are required imports.

## Replot the setup, family ratios and stability table

All 20 saved stage files are included under their original runtime
directories (`final_llv/`, `calib_llv/`, `select_formal/`, `stability_llv/`,
`setup_llv/`, `shortlag_llv/`, `curves_llv/`, and `density_llv/`). From this
directory:

```bash
python llv_figures.py --out preview_stage_figures
```

The full builder checks all eight JSON sidecars, four NPZ hashes and
cross-stage identities before plotting. This command was tested successfully
with the selected source and transferred inputs. It regenerates no model
output, changes no classification and reselects no checkpoint.

Use `fig_llv_setup`, `fig_llv_family_ratios`, and `tableS_stability` from
that output. The current manuscript's segmentwise correlation/density
figures come from the separate 12,000-trajectory command above. The archived
full builder also emits older short-/long-lag, sliding-window and density
figures; those are not additional current manuscript figures. Their required
exports explain why `llv_shortlag.py`, `llv_curves.py`, and `llv_density.py`
remain in the source selection.

The auxiliary `tableS_run_level` is not included in current Supplementary S3.
Without optional `class_formal/classification_run_200.json` through
`classification_run_219.json`, its validation-ratio and worst-family columns
contain `nan` and `?`. Do not distribute that unused auxiliary table as a
complete paper result. No new classification is needed.

## Experiment stages

| Stage | Entry point |
| --- | --- |
| Archive checks and manifest | `llv_gates.py` |
| Training | `llv_train.py` |
| Independent post-selection classification | `llv_classify.py` |
| Validation-side calibration | `llv_calibrate.py` |
| Freeze representative and population | `llv_select.py` |
| Final evaluation | `llv_final.py` |
| Long-run stress test | `llv_stability.py` |
| Representative 12,000-trajectory diagnostics | `llv_diag_12k.py` |
| Setup data | `llv_setup_data.py` |

The formal run population is 200--219, defined by `FORMAL_RUNS` in
`llv_train.py`. Its historical gated `RUN_SEEDS` entry 0--19 is not the formal
population. Preserve the original code rather than editing this discrepancy
away. Training/evaluation need the external KMC archive, original stage
products/checkpoints, and a valid manifest.

`llv_spec.py`, `llv_streams.py`, `llv_gates.py`, and the specification are
checksum-bound. Do not format or edit them merely for presentation. Existing
manifests can record absolute specification paths, so copying this source
does not automatically relocate an old manifest. No token or checksum bypass
is supplied.

## Time and sample sizes

Raw slices `[400:800)` and `[800:1200)` correspond to retained steps 1--400
and 401--800, after discarding 400 steps. Do not shift the slices again.
Quantitative evaluation uses 400 generated trajectories per model; boxes
summarize the 18 validation-successful models among 20 retained models.
The representative is run 201, checkpoint iteration 7000.

The million-step stability test applies only to that representative model:
50 trajectories, each with 400 warm-up steps and 1,000,000 retained steps,
without state resets. It is not a test of all 20 models. Family losses keep
`NT` normalization; the displayed diagnostic correlations use `N(T-k)`.

## Upstream data

The archive is `llv_theta_T1200_B20000.npz`, conventionally supplied through
`ARCHIVE`; the specification is supplied through `SPEC`. The archive is not
included. Selected upstream helpers are in `data_generation/`, but that is
not a complete SPPARKS build or a confirmed production-launch recipe.
