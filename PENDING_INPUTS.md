# Completed saved-input inventory

The requested saved-result collection is complete. The inventory below
records the included inputs, not new requests to retrain or regenerate
model ensembles. Original filenames and relative directories are preserved.

This file retains its historical `PENDING_INPUTS.md` name for existing links.
The public release includes the curated inputs needed by the documented
paper figure/table workflows. The lists below are completed collection
records, not outstanding release tasks. They do not inventory everything
needed for full retraining or simulation from scratch; see
[reproducibility scope](README.md#reproducibility-scope) and the experiment
READMEs for those separate requirements.

## Section 4 Duffing

Items 1--5 below have been collected and checked. All six selected plots,
including the final contour figure, run successfully. Item 6 is exported
locally from item 1 by
`duffing/table_balancing.py`; no cluster search for an old table is needed.
The list is retained as the input inventory:

1. Final L2H10 balancing histories and retained-run records under `runs/grid/`:
   `hist_nf3_lb{on,off}_din2_L2H10_lam*_s*.npz` and their matching
   `run_nf3_lb{on,off}_din2_L2H10_lam*_s*.json`. These cover the six couplings,
   with 20 runs at lambda=4 and 10 otherwise: 140 runs total.
2. The 24 `reports_val/report_nf{0,1,2,3}_lbon_din2_L2H10_lam*.json`
   reports for those six couplings.
3. Official `din_eval.json` and its 18 `din_eval_cells/` JSON shards.
4. Saved `data/reference_lam*.npz`, `data/reference_lam*.json`,
   and `data/target_stats_lam*.npz` for those six couplings.
5. `timing_exclusive.json`, the saved standalone GPU timing result. Its
   `slurm_exclusive` field is null; allocation exclusivity is not established
   by the filename or the number of visible devices.
6. The final S1 late-training-loss/success table is now available under
   `duffing/tables/table_loss_balancing.{csv,tex}`, with a per-run audit CSV.
   All 12 cells reproduce the supplied values; totals are 66/70 per setting.
   Do not substitute the old `appc2.py` output.

The contour inputs are included: `data/traj_lam4.npy`, plus four saved
`gen_<display_run>.npz` ensembles and four matching report JSONs under
`reports_val_disp/`. No ready-made density grids have been found. The
existing renderer uses these files to reproduce the lambda=4 contour
figure; the 24 main validation reports alone would be insufficient.

The target archive hash and reference constants pass the original loader;
the four display runs and report fields match the main validation reports.
The rebuilt figure prints the same displayed/cell JOINT ratios as the
recorded figure-generation output. No model was retrained or resampled.
There are no remaining Duffing transfer requests for this plotting workflow.

The included validation-anchored reports must not be replaced by older local
evaluation-anchored reports. Replotting these saved results needs no new jobs.

## Section 5.2 LLV

The four 12,000-trajectory diagnostic figures are already covered locally.
All 20 requested stage files were received and verified on 2026-10-31,
then integrated under `llv/` in their original relative directories.
There are no outstanding required files for the current saved-input LLV
figure/table workflow. The completed list is retained below for reference:

```text
final_llv/final_evaluation.json
final_llv/final_evaluation.sha256
calib_llv/calibration.json
calib_llv/calibration.sha256
select_formal/representative.json
select_formal/representative.sha256
stability_llv/stability.json
stability_llv/stability.sha256
setup_llv/setup_data.json
setup_llv/setup_data.sha256
setup_llv/setup_data.npz
shortlag_llv/shortlag_curves.json
shortlag_llv/shortlag_curves.sha256
shortlag_llv/shortlag_curves.npz
curves_llv/longlag_curves.json
curves_llv/longlag_curves.sha256
curves_llv/longlag_curves.npz
density_llv/joint_density.json
density_llv/joint_density.sha256
density_llv/joint_density.npz
```

The short-/long-lag and older density exports are required by the unmodified
builder, not requested as additional manuscript figures. No checkpoints,
raw archive, newly generated ensembles, or new classification is needed.

All eight JSON sidecar checks, four binary artifact hashes and cross-stage
identities passed. The full archived builder completed successfully. The
collection records and earlier transfer snapshot are preserved internally
under `paper_code/llv_saved_inputs_20261031/`; they are not duplicated in
the release. No checksum sidecar was manufactured or changed.

### Optional run-level table

The current `section_5_2_llv_supplement.tex` includes setup, transient,
paired-change and stability tables, but not `tableS_run_level`. Therefore
`class_formal/classification_run_200.json` through
`classification_run_219.json` are not required for the current paper figures
and tables, and are not added to `WANTED_FILES.txt`.

If the extra run-level table is restored, retrieve those 20 existing
classification JSONs as well. Otherwise the unchanged builder can emit
that unused table with `nan` validation ratios and `?` worst-family fields.
Such an incomplete auxiliary table must not be distributed as a complete
paper result. This does not require new classification or changing any mask.

## Outside the completed saved-results scope

FPU plot inputs and numerical supplementary inputs are included; no further
collection is required for those documented outputs. Full raw archives, all
trained models, historical job logs, and the complete SPPARKS build tree are
outside this saved-results selection.

The runnable Duffing example uses the included lambda=4 data and has successful
CPU quick-mode and GPU full-mode run records, as documented in the
[example README](examples/duffing/README.md). It is a one-seed demonstration;
the executed full-mode notebook is a results preview, not a release of all
paper checkpoints or a reproduction of the across-run experiments.

Full retraining and evaluation still depend on external raw data and original
checkpoints/stage products described in the experiment READMEs. LLV simulation
from scratch additionally needs the exact SPPARKS fork/version, build details
and confirmed production inputs. The SPPARKS binary remains excluded;
third-party redistribution permissions require separate verification as
described in the [licensing note](README.md#license-and-third-party-components).
These limitations do not reopen the completed plotting-input collection.
