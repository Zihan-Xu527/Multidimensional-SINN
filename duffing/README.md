# Coupled Duffing benchmark

The selected files cover data preparation, training, retained-checkpoint
evaluation, and the current Section 4/S1 analyses. The 140 saved run/history
pairs, 24 validation reports, input-dimension evaluation, fixed references,
timing record and the lambda=4 contour inputs are included. All six selected
analysis/diagnostic figure entry points have been tested successfully.

## Data and model workflow

The preparation order is `make_target.py`, `make_reference.py`, then
`make_target_stats.py`. These create new data and are not needed just to
inspect existing results. Keep the original references when loading a saved
model: they fix standardization, density grids, scales and bandwidths.

`train_sinn.py` contains the generator and single-run training entry point;
`sinn_loss.py` contains the differentiable objective; `evaluate.py` contains
evaluation statistics. `report.py` creates retained-model reports, while
`eval_din.py` performs the noise-dimension final evaluation.

The paper configuration explicitly sets `--layers 2 --hidden 10
--iters 10000 --warmup 200 --val-every 200 --batch 400 --n-gen 400`.
Several archived defaults differ. File parameter `--nf 3` means four total
frames in the manuscript, including the coordinate frame.

The legacy trainer/report `--tau` uses an unsquared error ratio, whereas the
paper's threshold 10 is on the squared loss ratio. The corresponding legacy
CLI value is `3.1622776601683795`, not 10. The current reporting scripts
square stored per-run ratios once. At lambda=0, CCF is retained as an
absolute training loss but excluded from the validation worst-family score.

## Final analysis entry points

Run these from this directory using the included saved inputs:

| Result | Entry point |
| --- | --- |
| Adaptive weights and run-level spread | `fig4_pair_runwise.py --rundir runs/grid` |
| Threshold attainment | `fig_attain.py --rundir runs/grid` |
| Joint-density error | `fig_joint_loss.py --repdir reports_val --key JOINT --quantity error --yscale log` |
| Held-out projection error | `fig_joint_loss.py --repdir reports_val --key PROJ_HOLDOUT --quantity error --yscale log` |
| Joint-density contours at lambda=4 | `fig_joint.py --repdir reports_val_disp --lam 4` |
| Noise-dimension figure/table | `din_analytic.py --json din_eval.json --cachedir din_eval_cells` |
| Family-level validation ratios | `ratios10.py` |
| Updated within-run spread column | `spread_runwise.py` |
| Late-training loss and validation-success table | `table_balancing.py --rundir runs/grid --outdir tables` |
| Saved per-iteration timing | `timing_exclusive.json`; `timing.py` is the benchmark implementation |

`fig4b_runwise.py` and `fig_valloss.py` are required imports for the final
weight/spread plot. Their older standalone aggregation is not the current
spread definition. Likewise, replace the old spread emitted by `ratios10.py`
with the values from `spread_runwise.py`.

The joint/held-out diagnostic default is the unsquared error even when a
filename contains `loss`; do not silently square it. The final input-dimension
figure uses `total_analytic_densities`, not `total`. Names containing
`candidate` are preserved because they are the actual selected renderers.

Use only the final L2H10 histories in `runs/grid/`; some analysis scripts do
not filter other capacities. Across-run medians include all runs; the
first-attainment iteration summaries necessarily include only runs reaching
the corresponding threshold. Timing is the synchronized standalone benchmark,
not packed-job wall time divided by iteration count.

The saved timing record gives 0.0105416095 s/iteration without balancing and
0.0112924966 with balancing on an A100. Its `slurm_exclusive` field is null:
the JSON alone does not establish node exclusivity. Check the accounting
record for job 340735 before making that allocation claim. No new timing
benchmark is required to replot or summarize the saved paper results.

The joint-contour renderer also reads `data/traj_lam4.npy` and four generated
display ensembles plus four report JSONs in `reports_val_disp/`. It is not
yet a small-array-only renderer. These inputs are now included: nf0s03,
nf1s16, nf2s09 and nf3s13. Their reports match the main validation reports,
and the target/reference/provenance hashes pass the original loaders.
The main `reports_val/` directory is not a substitute for the display
directory. `fetch_saved_inputs.sh` is retained only as a transfer helper,
not a prerequisite for using the completed bundle.

The old `appc2.py` is deliberately excluded: it does not produce the current
late-training loss table. The added `table_balancing.py` is a focused
saved-record exporter, not a new trainer or evaluator. It takes each run's
median final-five `train_unweighted` values, then the median across runs.
Success uses the maximum squared stored family ratio at the retained
checkpoint, with threshold 10; it does not read the old `best.success`
boolean. The generated six rows match the supplied paper values and both
success totals are 66/70. Summary CSV, per-run CSV and LaTeX are included
under `tables/`.
