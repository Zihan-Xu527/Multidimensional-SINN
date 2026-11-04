# Multidimensional SINN paper code

Curated code for Section 4 (Duffing), Section 5.1 (FPU), and Section 5.2
(lattice Lotka--Volterra). The three experimental implementations remain
separate. Existing source files and saved numerical inputs are copied
unchanged; `SELECTION.csv` records their origins.

This public research-code release was made available on 2026-11-01. It
includes the saved inputs for the documented figure/table workflows and a
runnable Duffing example. It does not provide complete end-to-end
reproduction of every paper experiment.

## Reproducibility scope

| Workflow | Available in this public release | Scope and limitations |
| --- | --- | --- |
| Replot saved paper results | Curated inputs and the figure/table entry points listed below | Rebuilds the documented outputs from archived results; does not retrain models or independently regenerate the underlying ensembles |
| Train and evaluate the Duffing example | Included lambda=4 target data, runnable notebook, and executed full-mode preview | CPU quick mode and GPU full mode have successful run records; each uses one seed and does not reproduce the paper's across-run study |
| Reproduce the complete paper experiments | Research source and per-study protocol notes | Full retraining and evaluation still require external raw datasets, original checkpoints/stage products, and study-specific setup; these materials are not all public here |

The saved-input collection is complete for the documented plotting scope.
[PENDING_INPUTS.md](PENDING_INPUTS.md) retains its historical filename and
records that completed inventory, not an outstanding task list. See the
experiment READMEs for the exact inputs and scientific limitations.

## Included workflows

| Directory | Contents | Saved-input rendering available now |
| --- | --- | --- |
| [duffing/](duffing/README.md) | Data preparation, training, evaluation, final analyses and timing | All six analysis/diagnostic figures and balancing tables |
| [fpu/](fpu/README.md) | Data generation, training, evaluation, final plots and supplementary tables | Figures 9--11 and supplementary FPU tables |
| [llv/](llv/README.md) | Frozen experiment stages, stability test and final diagnostic renderers | Setup, family-ratio boxes, stability table, and initial/continuation diagnostics |
| [llv/data_generation/](llv/data_generation/README.md) | Log parser, site generator and reference input decks | Not a complete SPPARKS simulation release |

Read the README inside each experiment directory. Figure numbers refer to
the manuscript version used during collection, so filenames and descriptions
are the more stable identifiers.

## Minimal runnable example

[The Duffing notebook](examples/duffing/demo.ipynb) offers a 60-iteration
quick check and a 10,000-iteration training demonstration using the included
lambda=4 data. Both show marginal densities, ACFs, and joint-density contours;
an optional branch regenerates the target data. This is a one-seed example,
not the paper's across-run study. See the
[example instructions](examples/duffing/README.md) for environment setup,
interactive execution, and a headless cluster command. New outputs remain
separate from the saved paper results.

For an immediate preview of the plots and results, open the
[completed 10,000-iteration notebook](examples/duffing/demo_full_executed.ipynb).
The successful CPU quick-mode and A100 GPU full-mode environments are recorded
in the example instructions. These runs reuse the included target data;
optional target regeneration is not covered by those checks. The executed
notebook embeds the results for inspection; its runtime checkpoint and
generated output directory are not included.

## Requirements

NumPy and Matplotlib suffice for the included saved-input plots and FPU
tables. Training and model evaluation additionally require PyTorch; FPU
also uses SciPy. `requirements.txt` lists dependencies, not a verified
cross-platform environment lock. Do not assume bitwise GPU reproduction
from an arbitrary software/backend combination.

Training uses the original per-study loss and optimizer conventions. Do not
merge the absolute-MSE FPU implementation with the relative-loss Duffing
and LLV implementations while claiming to reproduce the archived runs.

## Selection boundaries

Historical notebooks, cluster-specific launchers, pilot jobs, obsolete
standalone plots, duplicated outputs, debug logs, and the independent LLV C
cross-check are excluded. One Duffing lambda=4 target archive is included
because the contour renderer verifies and reads it. Other large trajectory
archives and model checkpoints need a separate access location. The
SPPARKS binary is not redistributed here; see the licensing note below.

Some retained modules contain older standalone functions but are still
imported by the selected final scripts. Use the entry points named in each
README rather than executing every file indiscriminately.

The original collection under `paper_code/` is an internal archive, outside
this public selection. Full experimental reproduction requires separate
access to the external materials described above and the original dependency
and backend settings. The public plotting inputs and example do not imply
that all raw datasets or trained models are available.

## License and third-party components

This repository uses the [BSD 3-Clause License](LICENSE). That license does
not establish redistribution permission for third-party SPPARKS components.
The excluded binary's specific license and fork/build provenance have not
been confirmed. Verify the applicable terms for the exact component before
adding or redistributing it. The included LLV helpers and reference decks
do not supply a complete SPPARKS build or confirmed production-launch recipe.

## Checks completed on this selection

The 57 preserved Python files and the added saved-record table exporter
pass syntax checks, and their local import
dependencies are present. The three FPU figures and four LLV diagnostic
figures were rendered successfully from the selected saved inputs in a
temporary copy. All five rebuilt FPU supplementary CSV/TeX files match the
supplied files byte-for-byte. These are saved-results checks, not a new
training run or an end-to-end GPU reproduction.

All 20 requested LLV stage files have also been integrated. Their eight
JSON checksum sidecars, four NPZ hashes, and cross-stage references agree.
The full archived LLV figure/table builder completes successfully, including
the setup, family-ratio and stability outputs. Its auxiliary run-level table
is not part of the current Supplementary and lacks optional classification
columns; do not distribute that auxiliary output as a complete paper table.

The Duffing inputs now cover 140 saved runs, 24 reports, 18 input-dimension
cells, frozen references and the timing record. Six analysis/diagnostic plots run
successfully; the added table exporter reproduces all 12 late-training
loss/success cells. The final lambda=4 contour figure was also rebuilt from
the matching target archive and four saved display ensembles/reports.
There are no further input-collection requests for this saved-results
plotting workflow. This does not claim that all raw
datasets, checkpoints or from-scratch simulation dependencies are released.
