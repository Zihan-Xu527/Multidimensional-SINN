# FPU application

The selected source keeps the archived FPU convention: absolute-MSE family
losses and directly optimized positive scales initialized at 2 and clipped
below at 1e-4. It is not replaced by the Duffing/LLV relative-loss convention.

## Replot the saved results

These commands need only NumPy and Matplotlib. Each block starts from the
stated directory relative to this FPU directory:

```bash
cd plotting/fig09_family_ratios
python fig_family_ratios_vs.py --root .
```

```bash
cd plotting/fig10_correlations
python render_commonylim.py
```

```bash
cd plotting/fig11_joint_densities
python -c "import render_s4style as r; r.figure11()"
```

The family boxes use the saved validation-successful masks (16/20 and
18/20), with unsuccessful models shown separately. The script's auxiliary
caption still has older accepted/rejected wording; use the revised paper
caption. The empty output directories are intentional.

The correlation input already uses pair-count normalization `N(T-k)`.
Do not apply the conversion again. The renderer uses common vertical
limits; its second output is an alternative tick-spacing version. The
local manuscript still had the older per-panel-axis version at collection.

The density renderer uses preserved contour paths, not recovered density
grids. It can reproduce the saved paths, but does not establish regeneration
from the original trajectories. The September batch's checkpoint hash was
not recorded. Do not execute `render_s4style.py` directly: its main block
first calls a superseded correlation renderer.

The supplementary capacity and final-evaluation tables are included under
`section51_tau10_refresh/tables/`. To rebuild their numerical summaries from
the four saved evaluation files, run from this FPU directory:

```bash
python section51_tau10_refresh/scripts/build_section51_tau10.py --src section51_tau10_refresh/source_data --out section51_tau10_refresh/rebuilt
```

This archived transform also emits historical audit metadata. Its missing
checkpoint/figure audit inputs are not needed for the numerical tables;
its hard-coded historical provenance text is not a record of a new local
execution. Use the separate final renderers above for the figures.

## Training and evaluation

`generate_data.py` and `fpu_chain_worker.py` generate fitting data;
`gen_newT.py` generates the independently seeded target archive.
`experiments.py` trains models using `toolbox/`; all five toolbox modules
are retained because the package imports them eagerly. `refblocks.py`,
`widtheval.py`, and `finaleval.py` implement reference blocks, independent
validation classification, and final evaluation.

The paper uses three LSTM layers, 20,000 iterations, and width
30/50/75/110 with 10/10/20/10 runs per target. Set these explicitly:
the trainer defaults to five seeds, and the fitting-data generator defaults
to 4,000 trajectories rather than the paper's 20,000. File `--n-frames 3`
corresponds to four total frames including coordinates.

Training uses 400 generated trajectories per step. Monitoring reuses the
archived fixed validation noise; independent classification is a later
stage. Quantitative final evaluation uses 400 generated trajectories per
run, not the trainer's legacy `--eval-traj 4000` default. The diagnostic
curves/contours use 12,000 per ensemble.

Full retraining/evaluation requires the large fitting and target archives,
reference-block files and model checkpoints, which are not included in
this small saved-results release. Do not point new runs at archived output
locations or reuse a generation cache for different parameters.
