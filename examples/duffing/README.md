# Minimal Duffing SINN example

Train and evaluate one SINN model at coupling `lambda=4`.
The notebook calls the existing scripts in `duffing/`; it does not introduce
a second implementation of the model, losses, or estimators.

For a preview without running anything, open the
[completed 10,000-iteration notebook](demo_full_executed.ipynb). It includes
the training history, evaluation table, marginal densities, ACFs, and
joint-density contours. Use [demo.ipynb](demo.ipynb) for a fresh run.

The default uses the target archive and statistics already included in this
repository. Choose the budget with one setting in the notebook:

```python
MODE = "quick"  # "quick": 60 iterations; "full": 10,000 iterations.
```

Quick mode checks the workflow, with validation every 20 iterations. Full mode
uses the original 10,000-iteration budget and validation every 200 iterations
to demonstrate learning. Both train one seed, retain the checkpoint with the
smallest validation aggregate, and evaluate a new generated ensemble. All
other numerical settings are identical. Quick mode need not pass the accuracy
threshold, and neither mode reproduces the paper's across-run study.

## Environment

Use a Python environment with NumPy, Matplotlib, and a PyTorch installation
appropriate for the available CPU or GPU. Install
missing packages in that environment, not a different notebook kernel.
For Jupyter support:

```bash
python -m pip install notebook nbconvert ipykernel
python -m ipykernel install --user --name sinn-demo --display-name "SINN demo"
```

On a cluster, run the notebook kernel on an allocated compute node. Do not
run the training cells on a login node. CUDA is selected when available;
otherwise the default is CPU. Runtime depends on the machine and environment.

The quick-mode training and evaluation path completed on CPU with four threads using
Python 3.8.18, NumPy 1.24.4, PyTorch 2.0.1+cu118, and Matplotlib 3.7.2.
Training took 16.6 seconds and evaluation 7.9 seconds, producing 400 finite
trajectories of 200 scored steps each.

The full-mode preview completed on an NVIDIA A100 80GB PCIe GPU using
Python 3.9.7, NumPy 1.25.1, PyTorch 2.0.1+cu118, and Matplotlib 3.7.2.
Training and evaluation together took 110.7 seconds. Both checks reuse the
supplied target data; optional target regeneration is not covered.

## Run interactively

From the repository root:

```bash
python -m notebook examples/duffing/demo.ipynb
```

Select the **SINN demo** kernel, then **Restart Kernel and Run All**.
The notebook covers configuration, target-data inspection, training,
checkpoint evaluation, marginal-density/ACF comparisons, and joint-density
contours. Change `MODE` to `"full"` for the longer training demonstration.
After changing the mode or other settings, restart and run all cells to create fresh
outputs instead of rerunning only the training cell.

## Run without a browser

From the repository root, in the same environment and compute allocation:

```bash
mkdir -p examples/duffing/outputs
python -m nbconvert --to notebook --execute examples/duffing/demo.ipynb \
  --output demo.executed.ipynb --output-dir examples/duffing/outputs \
  --ExecutePreprocessor.kernel_name=sinn-demo \
  --ExecutePreprocessor.timeout=3600
```

Inspect the executed notebook and the new `outputs/demo-quick-*/` or
`outputs/demo-full-*/` directory.
Each execution creates a fresh directory containing a checkpoint, history,
evaluation report, generated ensemble, four PNG figures, a joint-density PDF
and density arrays, stage logs, configuration, and `demo_summary.json`.
New run outputs are ignored by Git and never
overwrite `duffing/data/`, `duffing/runs/`, or the saved paper reports.

## Plot a completed run without retraining

The joint-density helper also accepts an existing demo directory, including
outputs from the earlier notebook without the mode switch. Replace the path
below with your completed run's directory:

```bash
python examples/duffing/plot_joint.py \
  --run-dir examples/duffing/outputs/demo-full-XXXXXXXX
```

This reads the saved evaluation ensemble and uses the original bivariate KDE.
It does not train, generate trajectories, rerun evaluation, or change validation
classification. The output files are `joint_density_contours.png`,
`joint_density_contours.pdf`, and `joint_density_contours.npz` in that same
directory. NumPy and Matplotlib suffice for this plotting-only command.

## Optional data generation

Set `REGENERATE_TARGET=True` in the configuration cell and rerun from the
top. This branch calls `make_target.py`, `make_reference.py`, and
`make_target_stats.py` for lambda 4 in the fresh output directory. It uses
20,000 trajectories and the original split and preprocessing definitions;
do not shrink that count to the generator's standalone 2,000-trajectory
smoke example, which is insufficient for the reference-building protocol.
Regeneration needs extra time and memory. It is not required for the default
demonstration and does not repeat the simulation-convergence study.

## Numerical conventions

The example keeps two LSTM layers of width 10, two active noise channels,
four total projection frames, loss balancing, 200 warm-up steps, and 200
scored steps. Training and evaluation each use 400 generated trajectories.
Quick mode shortens only the iteration budget and validation interval.
It deliberately avoids `--smoke`, which also changes the training batch size.

The original CLI uses unsquared error ratios, so `--tau` is `sqrt(10)`.
The notebook squares the evaluator's ratios once for the displayed
reference-normalized loss ratios. Density targets at evaluation are
kernel-smoothed analytic densities; correlations use the held-out target
ensemble and the original `NT` estimator. No pair-count conversion is made.

Joint-density contours reuse that same 400-trajectory generated ensemble.
The target is the kernel-smoothed analytic joint density, and both sides use
the original two-dimensional bandwidth. Black solid and red dashed lines
denote target and SINN. The contour thresholds are chosen once to enclose
10%, 30%, 50%, 70%, and 90% of target mass, then reused for SINN. Joint density
is a diagnostic and enters neither training nor checkpoint selection.

For the paper-scale settings, see [the Duffing README](../../duffing/README.md).
