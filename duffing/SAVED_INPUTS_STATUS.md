# Section 4 Duffing — saved-input collection status

Collection attempt: 2026-10-31. Request: `../PENDING_INPUTS.md`, section
"Section 4 Duffing". Copy only; no training, no re-evaluation, no recomputed
statistics.

## Result

Items 1-5 were fetched with `fetch_saved_inputs.sh` and are in place.
Item 6 is now exported locally from the collected histories/run records by
`table_balancing.py`; all 12 published cells reproduce. The contour-renderer
inputs (item 7) have also arrived and the figure has been rebuilt successfully.
The saved-result collection is complete.

| # | Requested | Expected | Collected |
| --- | --- | --- | --- |
| 1 | `runs/grid/hist_*` + `run_*` | 140 runs = 280 files | 280 |
| 2 | `reports_val/report_nf{0,1,2,3}_lbon_din2_L2H10_lam*.json` | 24 | 24 |
| 3 | `din_eval.json` + `din_eval_cells/` | 1 + 18 | 1 + 18 |
| 4 | `data/reference_lam*.{npz,json}`, `data/target_stats_lam*.npz` | 6+6+6 | 18 |
| 5 | `timing_exclusive.json` | 1 | 1 |
| 6 | final S1 late-training-loss/success table | summary/per-run CSV and TeX | **exported and checked** |
| 7 | `data/traj_lam4.npy` + four display ensembles and matching reports | 1 + 4 + 4 | **collected and checked** |

Source: `zxu64@login.rc.ucmerced.edu:/home/zxu64/GIT/Multidimensional-SINN/MDSINN/Duffing_dimer`.

## What was checked after the transfer

Reads only. Nothing was retrained, re-evaluated or recomputed.

* **Cell census.** 140 runs, split 20/20 at `lambda = 4` and 10/10 at each of
  0, 0.5, 1, 2, 8. Every `run_*.json` has its matching `hist_*.npz`.
* **Uniform training configuration.** All 140 carry
  `families = (ACF, CCF, ACF2, PDF, PROJ)`, `iters = 10000`, and
  `gpu_name = NVIDIA A100 80GB PCIe`.
* **Archive binding.** Six distinct `constants_sha256`, one per coupling, and
  each equals the `constants_sha256` of the matching `data/reference_lam*.json`.
  So every retained checkpoint is bound to the reference archive it was trained
  against, and the archives that came down are the right six.
* **Report anchor.** All 24 reports carry `anchor = "validation"`,
  `anchor_segment = [4000, 8000]`, and the same six `constants_sha256`. This is
  the validation-anchored set, not the superseded evaluation-anchored
  `reports/` still sitting in `~/Downloads/toolbox/Duffing_dimer/`. The two were
  not mixed.
* **`din_eval`.** 18 cached cells, matching `d_in` in {1,2,3} x six couplings.
* **Timing.** `s_per_iter = 0.01054161` (balancing off) and `0.01129250`
  (balancing on), IQR ~1.7e-5 and ~1.6e-5, on `gnode009.cluster`, one A100
  80GB PCIe. The 0.01054 figure quoted in Section 4 comes from this file.

  One caveat, stated rather than resolved: the file records
  `slurm_exclusive = null`, so **it does not itself record that the allocation
  was exclusive**. `CUDA_VISIBLE_DEVICES = "0,1"` with two GPUs visible on a job
  that requested `gpu:a100:1` is consistent with the whole node having been
  allocated, which is what `--exclusive` does, but that is an inference, not a
  record. If exclusivity has to be asserted in print, check the job's SLURM
  accounting entry (job 340735) rather than this file.

## Where the contour-figure display ensembles actually live

`PENDING_INPUTS.md` says "four saved `gen_<display_run>.npz` ensembles". These
are **not** in `reports_val/`. `reports_val.sh` passes `--save-display` on the
default cell only, so `reports_val/` holds `gen_nf3_*` and nothing for
`nf = 0, 1, 2`. The four panels come from `reports_val_disp/`, written by
`gen_joint.sh` (the four `n_f` cells at `lambda = 4`, `--save-display` on each).
`fig_joint.py --repdir reports_val_disp` is the documented invocation.

No saved target/SINN density grids or contour levels for the four displayed
panels were found. The original renderer therefore uses the now-collected
`data/traj_lam4.npy` and four `gen_*.npz` ensembles with their matching report
JSONs. Its four selected runs are nf0s03, nf1s16, nf2s09 and nf3s13, all on
the validation anchor. The target archive SHA, reference constants and
generated-array provenance pass the original loaders. Relevant report fields
are identical to the corresponding `reports_val/` records. The figure builds
successfully and reproduces the recorded displayed/cell ratios:
8.60/8.58, 4.13/4.12, 4.13/3.59 and 1.45/1.84. No trajectory was generated.

## Item 6 resolved from the saved records

`table_balancing.py` reads all 140 saved run/history pairs. It computes each
run's median last-five `train_unweighted` values, then the median across
runs of each cell. Validation successes use the retained checkpoint's
maximum squared family ratio <=10, not the old boolean `best.success`.
The outputs are `tables/table_loss_balancing.csv`,
`tables/table_loss_balancing_runs.csv` and `tables/table_loss_balancing.tex`.
All 12 loss/success entries match the supplied table; totals are 66/70 for
both settings. No model was loaded or re-evaluated.

## Historical table file not used

`../../section_4_benchmark_supplement.tex` (device path
`/Users/zihan/Documents/ChatGPT/Multidimensional-SINN/section_4_benchmark_supplement.tex`)
contains a table `tab:supp_4lb` with a part (b) "Unweighted loss". It was **not**
collected as item 6, for two reasons stated by the file itself:

* its caption defines the spread as "the ratio between the largest and smallest
  family-level median validation ratios at the retained checkpoints" — the
  across-run-median definition, not the run-level column produced by
  `spread_runwise.py`;
* it has no success column.

So it is a superseded version of that table, not the final input item 6 asks
for. It is left where it is. `appc2.py` was likewise not used (excluded by the
request and by `README.md`).

## Duffing result files that ARE on this machine, and are NOT substitutes

Under `/Users/zihan/Downloads/toolbox/Duffing_dimer/` (the same folder is also
mounted as `/Users/zihan/Downloads/toolbox`, identical inode):

* `reports/` — 70 files, mtime 2026-09-15 21:34. Written by `reports.sh` /
  `reports2.sh`, which score against the **evaluation** segment. Item 2 asks for
  `reports_val/`, written by `reports_val.sh` with `--against validation`. The
  two anchors must not be mixed; `reports_val.sh` records that `fig_din.py` and
  `fig_joint.py` refuse a directory holding both. Not collected.
* `reports/gen_*.npz` — 10 display ensembles on the same superseded anchor. Not
  collected.
* `figs/` — 22 rendered PDFs, mtimes 2026-09-15 to 2026-10-03, all superseded by
  the current figure set.
* `section4_full.tex` (2026-09-15), `report_340054.out` (job log).
* `_superseded/` — 20 earlier script versions.

## Where it was searched

`/Users/zihan/Documents/ChatGPT/Multidimensional-SINN` (whole tree, including
`paper_code/`, `outputs/`, `output/`, `scripts/`, `tmp/`),
`/Users/zihan/Downloads`, `/Users/zihan/Downloads/toolbox`,
`/Users/zihan/Downloads/Multi-Dimensional SINN (Mach. Learn.: Sci. Technol.)`,
and the listing of `paper_release_20261031.tar.gz` (110 entries; its
`duffing/` entries are the 19 source files only — the FPU saved inputs under
`fpu/plotting/*/data/` have no Duffing counterpart).

Patterns searched: `hist_nf*`, `run_nf*`, `report_nf*`, `din_eval*`,
`reference_lam*`, `target_stats_lam*`, `timing*.json`, `traj_lam*`, `gen_*.npz`,
and directories named `grid`, `reports_val*`, `din_eval_cells`, `*duffing*`.
