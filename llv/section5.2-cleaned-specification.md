# Section 5.2 (LLV) — frozen implementation and reporting specification

**Status:** implementation source of truth, frozen 2026-10-06.  This document
supersedes `section5-restructure-plan.md` §4 and
`llv-section5.2-test-design.md` wherever they disagree.  The older documents
remain useful only for background and provenance.

No final-evaluation result may be generated until all pre-evaluation gates in
§12 pass and the configuration and seed manifests have been written.  Once the
final target segment is opened, changes to the estimators, run-selection rule,
figures, or reporting population require a new version of this specification
and a complete final-evaluation rerun.

## 1. Scientific question and claim boundary

The LLV application tests a different extension from §5.1.  The microscopic
state contains \(10^4\) lattice occupations, while the surrogate generates only
the two spatially averaged densities

\[
\boldsymbol X(t)=\bigl(\rho_A(t),\rho_B(t)\bigr)^{\mathsf T}.
\]

It therefore combines genuine spatial coarse-graining (\(10^4\to2\)) with a
temporal extrapolation test.  The generated trajectory is advanced beyond the
recurrent depth used in training, and its statistics are compared with a
strictly held-out KMC target ensemble.

The experiment may support only the following claims:

1. validation-successful SINNs reproduce the prescribed statistics within the
   training-scored window \(G_{\rm in}\), at the reported finite-sample scale;
2. the paired error-change distribution from \(G_{\rm in}\) to the contiguous
   \(G_{\rm out}\), which lies beyond the training-time unroll;
3. whether the extended rollout shows a detectable long-lag linear-correlation
   discrepancy above a validation-side finite-sample envelope at lags that
   never entered the loss; and
4. one validation-selected model remains finite and empirically stable for
   \(10^6\) post-warm-up recurrent steps.

The experiment does **not** establish pathwise prediction, mathematical
stationarity of the finite LLV system, exact long-time invariant-measure
accuracy, or learned long-range dependence.  The \(10^6\)-step calculation is
a stability diagnostic, not a long-time accuracy test.

The recurrent architecture is motivated by the information lost under spatial
coarse-graining (and hence the generally non-Markovian projected dynamics), not
by the oscillatory ACF alone.

## 2. Notation and terminology

- \(\rho_A,\rho_B\): physical prey and predator densities.  The symbol
  \(\boldsymbol\theta\) is reserved for network parameters.
- \(x_i=(\rho_i-\mu_i)/s_i\): standardized density, and
  \(\boldsymbol x=(x_A,x_B)^{\mathsf T}\).
- \(T=400\): one scored window; it is not the final data set.
- \(\mathcal F\): fitting trajectories; \(\mathcal R_j\): nine finite-sample
  reference blocks; \(\mathcal V\): validation target; and
  \(\mathcal T_{\rm ev}\): final target.
- `Target`: a KMC curve or ensemble, including \(\mathcal V\) and
  \(\mathcal T_{\rm ev}\).
- `Reference`: only a finite-sample baseline constructed from the nine
  \(\mathcal R_j\), or a threshold calibrated from them.  Figures must not call
  the KMC target curve “Reference.”
- \(\mathcal S=\{\mathrm{ACF},\mathrm{CCF},\mathrm{ACF2},
  \mathrm{PDF},\mathrm{PROJ}\}\).
- \(\mathcal L_s\): the **squared relative** family discrepancy defined
  explicitly in §4 below.  No square root is taken anywhere in model
  classification or evaluation.

## 3. Target archive, transient, and split

The processed analysis archive is `llv_theta_T1200_B20000.npz`, containing float32
`traj` with shape `(1200, 20000, 2)`.  Array index 0 corresponds to physical
time \(t=1\).  The archive contains 20,000 independent SPPARKS runs at

\[
k_1=0.5,\qquad k_2=0.3,\qquad k_3=0.1,
\]

on a periodic \(100\times100\) lattice, sampled at \(\Delta t=1\).  The raw
initialization assigns prey to 20% of sites and predators to 37.5% of the
remaining vacant sites.  The raw `res/surfcov*.txt` files are the immutable
provenance source and must not be deleted.

The KMC physical burn-in and generator warm-up are distinct quantities even
though both equal 400 here:

\[
W_K=400,\qquad W_G=400.
\]

Diagnostics support treating the recorded post-burn-in target process as
stationary at the resolution of the statistics considered here.  This is an
empirical quasi-stationary statement, not a proof about the invariant law of a
finite LLV lattice with absorbing states.

The existing direct block checks support the temporal families.  For PDF and
PROJ, the manuscript may state only that the two window-specific reference
errors are similar unless a separate paired, trajectory-level calibration is
performed; it must not upgrade similarity of denominators into a formal
“within reference variability” claim.

The trajectory split is fixed by trajectory index:

| set | trajectory indices | size | role |
|---|---:|---:|---|
| \(\mathcal F\) | `[0,400)` | 400 | fitting statistics and training target |
| \(\mathcal R_j\) | `[400+400(j-1), 400+400j)`, \(j=1,\ldots,9\) | 9 × 400 | finite-sample reference errors |
| \(\mathcal V\) | `[4000,8000)` | 4000 | checkpoint monitoring and independent success classification |
| \(\mathcal T_{\rm ev}\) | `[8000,20000)` | 12000 | final evaluation, opened once |

Within every KMC trajectory, physical \(t=1,\ldots,400\) is discarded.  The
two target windows are therefore the array slices

```text
target G_in  = traj[400:800,  ...]   # physical t = 401,...,800
target G_out = traj[800:1200, ...]   # physical t = 801,...,1200
```

All archive-wide NumPy reductions and all stored reporting statistics use
float64 accumulation.  The archive remains stored as float32.

### 3.1 Frozen burn-in/stationarity verification

This is an archive-health gate, not a model-selection endpoint.  The split and
random stream are fixed to the provenance-producing calculation in
`verify_and_calibrate_v2.py`: validation trajectories `[4000,6000)` form the
late-reference half, trajectories `[6000,8000)` form the disjoint tested half,
and the NumPy seed is `20260913`.  The late reference pools physical
\(t=901,\ldots,1200\) from the reference half.  The observed point is physical
\(t=400\), the last discarded step, from the tested half.

For the mean and variance limits, each of 200 replicates resamples complete
tested trajectories and evaluates the deviation from the fixed late reference
at every physical time \(t=901,\ldots,1200\).  The replicate statistic is the
maximum over all 300 late times and both components, so its 95th percentile is
a conservative simultaneous late-window envelope rather than a pointwise
standard error.  For the marginal-KS limit, each of 200 replicates draws one
tested late-window time and takes the maximum over the two components against
the pooled late reference.  This declared calculation reproduces:

| diagnostic | observed | frozen 95% comparison limit |
|---|---:|---:|
| maximum standardized mean deviation | 0.0407 | 0.1141 |
| maximum relative variance deviation | 0.0782 | 0.1692 |
| maximum marginal KS distance | 0.0234 | 0.0337 |

On the same tested half, compare physical \(t=401,\ldots,800\) directly with
\(t=801,\ldots,1200\).  This archive-health check uses the separately named,
**unsquared** window discrepancy

\[
d_s^{\rm win}
=
\frac{\left\|S_s(\mathcal V_{\rm in})-
                 S_s(\mathcal V_{\rm out})\right\|_2}
     {\left\|S_s(\mathcal V_{\rm out})\right\|_2}.
\]

It is not the squared family loss \(\mathcal L_s\), is not used in training or
model classification, and satisfies \((d_s^{\rm win})^2=\mathcal L_s\) for
this same ordered comparison.  The existing values and predeclared fixed
tolerance are therefore reported on the \(d_s^{\rm win}\) scale:

| family | observed \(d_s^{\rm win}\) | fixed limit |
|---|---:|---:|
| ACF | 0.0192 | 0.10 |
| CCF | 0.0199 | 0.10 |
| ACF2 | 0.0291 | 0.10 |

The mean and variance limits are trajectory-bootstrap simultaneous envelopes;
the KS limit is an empirical late-time cutoff, not a trajectory-bootstrap
maximum.  The three family limits are fixed operational tolerances and must
not be described as quantiles.  This gate reads no final-evaluation
trajectories.  PDF and PROJ have no stationarity pass/fail criterion; for them
apply only the restricted language stated above.  An equivalent implementation
may check the squared values against \(0.01\), but the manifest must store both
scales and label them explicitly.

## 4. Frozen preprocessing and statistical quantities

The standardization constants are estimated once in float64 from
\(\mathcal F\) over physical \(t=401,\ldots,800\), using population moments
(`ddof=0`), then frozen:

```text
mu_A = 0.121613     s_A = 0.022545
mu_B = 0.355175     s_B = 0.027240
```

These displayed values are rounded summaries, not constants to copy into code.
The full-precision float64 arrays are computed from the processed archive,
stored in the frozen configuration manifest, and covered by its checksum.
They are applied unchanged to \(\mathcal R_j\), \(\mathcal V\),
\(\mathcal T_{\rm ev}\), and every generated ensemble.  No later ensemble is
re-centered or re-scaled.

The coordinate marginals plus the following three non-coordinate orthonormal
frames are used:

\[
\mathbf W^{(1)}=\frac1{\sqrt2}
\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad
\mathbf W^{(2)}=\frac1{\sqrt5}
\begin{pmatrix}1&2\\2&-1\end{pmatrix},\quad
\mathbf W^{(3)}=\frac1{\sqrt5}
\begin{pmatrix}1&-2\\2&1\end{pmatrix}.
\]

Thus the paper reports **four frames in total** (one coordinate frame plus
three non-coordinate frames), while the code argument is
`n_noncoordinate_frames = 3`.  A metadata field named simply `n_frames` is
forbidden unless it also records which convention it uses.

For each unit row \(\boldsymbol w\), define and save, in float64,

\[
\varsigma_{\boldsymbol w}
=\operatorname{std}_{\mathcal F,\,{\tt ddof}=0}
\bigl(\boldsymbol w^{\mathsf T}\boldsymbol x\bigr),\qquad
u_{\boldsymbol w}
=\frac{\boldsymbol w^{\mathsf T}\boldsymbol x}
{\varsigma_{\boldsymbol w}}.
\]

The six full-precision scales are computed once from \(\mathcal F\), stored in
the frozen manifest, and then applied unchanged.  Directions and scales must be
passed separately.  In particular, do not encode the scale as
\(\boldsymbol w/\varsigma_{\boldsymbol w}\) and pass that vector to a helper
that normalizes directions again.  A fixture must verify
\(\operatorname{std}_{\mathcal F}(u_{\boldsymbol w})=1\) to float64 numerical
precision for every projection.  `PDF` contains only the two coordinate
marginals; `PROJ` contains only the six non-coordinate projected densities.

The KDE grid is 513 points on `[-5,5]`.  With \(TB=160000\), the frozen
**effective** bandwidths are

\[
h=(TB)^{-1/5}=0.091028210151304,\qquad
h_2=(TB)^{-1/6}=0.13572088082974534.
\]

The bandwidth never changes with generated ensemble size.  The current legacy
toolbox argument named `bw` is an exponent denominator: `bw=5` produces
\(n_{\rm current}^{-1/5}\), while passing `bw=0.091028...` would be
catastrophically wrong.  Passing `bw=5` is also wrong for
\(\mathcal V\), \(\mathcal T_{\rm ev}\), or \(B_{\rm vis}=12000\), because it
would recompute the bandwidth from their larger sample counts.  New code must
therefore expose an unambiguous `bandwidth_value` interface that uses the
two numerical values above unchanged for every ensemble.  The legacy helper
may be used only after it is wrapped or modified to bypass its exponent
semantics; neither the numerical \(h\) nor exponent 5/6 may be passed directly
to that old interface.  The bandwidth \(h_2\) is evaluation-only and never
enters training.

The canonical estimator for every training, checkpoint-monitoring,
classification, reference, and final-evaluation quantity is the literal
Gaussian-kernel sum in the manuscript.  It may be evaluated in sample chunks
and on a GPU; the training implementation must remain differentiable with
respect to the generated samples.  Histogram deposition followed by FFT
convolution is not mathematically identical for arbitrary sample locations
and must not replace the canonical estimator in the official pipeline.
Such an approximation may be retained only for profiling or exploratory
plots that do not enter any reported loss, ratio, classification, or table.

`ACF`, `CCF`, and
`ACF2` retain \(K=400\) lags, \(k=0,\ldots,399\).  `CCF` must return the two
ordered quantities

\[
C_{AB}(k)=\mathbb E[x_A(t)x_B(t+k)],\qquad
C_{BA}(k)=\mathbb E[x_B(t)x_A(t+k)],
\]

and tests must fail if either direction is dropped or duplicated.

For an ensemble \(A\), let \(S_s(A)\) be the complete family array.  Its order
is fixed as

```text
ACF   : ACF_A, ACF_B                         (2 x 400)
CCF   : CCF_AB, CCF_BA                       (2 x 400)
ACF2  : ACF2_A, ACF2_B                       (2 x 400)
PDF   : PDF_A, PDF_B                         (2 x 513)
PROJ  : the six directions in frame/row order (6 x 513)
```

Flatten and concatenate each family once in exactly this order.  For every
training, validation, reference, and evaluation comparison in this
specification,

\[
\mathcal L_s(A,B)
=
\frac{\left\|S_s(A)-S_s(B)\right\|_2^2}
{\left\|S_s(B)\right\|_2^2}.
\]

All five target norms must be finite and nonzero.  The shared loss constructor
must explicitly pass `loss_type=('mse_loss',)`; this specifies the
pointwise squared-difference primitive in the numerator, while division by the
target mean square produces the relative loss above (the common entry count
cancels).  Relying on the constructor default is forbidden because historical
and current copies of `_loss.py` have different defaults.  No L1 term,
per-curve normalization, square root, or second square may be introduced.

## 5. Frozen LLV training configuration

There is no recoverable LLV trainer or LLV checkpoint in the accessible
archive.  Consequently, no unverified historical setting is labeled as a code
fact.  The new experiment retains the configuration already declared in the
manuscript by “settings follow §4”; it does **not** inherit the capacity-tuned
FPU configuration from `experiments.py`.

| item | frozen LLV value |
|---|---|
| recurrent network | 2 stacked LSTM layers, 10 hidden units per layer, linear 2-output readout |
| network input width | 4; first 2 channels receive independent standard Gaussian noise and the remaining 2 are identically zero, matching §4 |
| active noise dimension | \(d_{\rm in}=2\) |
| training iterations | 10,000 |
| optimizer | Adam, learning rate \(10^{-3}\), betas `(0.9,0.999)` |
| training batch | 400 trajectories |
| raw training unroll | \(2T=800\) steps |
| scored training slice | raw `[400:800)` only |
| statistical families | all five families in \(\mathcal S\), all in relative-loss form |
| projection configuration | coordinate frame + all 3 non-coordinate frames |
| loss balancing | on |
| adaptive scales | \(\sigma_s=\exp(\alpha_s)\), all \(\alpha_s=0\) initially; separate Adam with the same learning rate and betas |
| validation cadence | iterations 200, 400, ..., 10000 exactly; 50 events and no iteration-0 event |
| independent training seeds | `0,...,19` (20 runs) |

Every training iteration draws a fresh length-800 noise batch.  The hidden and
cell states start at zero.  The first 400 generated outputs are warm-up and do
not enter any loss.

This table is deliberately different from the FPU application (3 layers × 75
hidden units, 20,000 iterations, direct \(\sigma_s\), validation every 50
iterations).  Copying those FPU defaults would silently change the LLV
experiment.  Any decision to use the FPU architecture instead requires a new
specification version before training begins.

## 6. Random streams and checkpoint selection

The implementation uses independent deterministic generators.  For run
\(m\in\{0,\ldots,19\}\), the initial seeds are:

| purpose | seed |
|---|---:|
| parameter initialization | `m` |
| training noise | `1000 + m` |
| checkpoint-monitoring noise | `2000 + m` |
| independent validation-classification noise | `3000 + m` |
| final finite-horizon generation | `4000 + m` |
| smooth-density visualization (representative run only) | `5000 + m` |
| \(10^6\)-step stability generation (representative run only) | `6000 + m` |
| long-lag reference bootstrap | `7000` |
| burn-in/stationarity calibration | `20260913` |

The checkpoint-monitoring generator is initialized once per run and advanced
continuously.  At **every** validation event it draws a new \(B=400\),
length-800 noise tensor.  It must not reuse a fixed `val_input`, and it must not
be re-seeded at each event.

Create explicit PyTorch generators on the recorded compute device for every
model-side stream.  Save the PyTorch and CUDA versions, device model, dtype,
deterministic-algorithm settings, and generator initial states in metadata.
Reproducibility is asserted on the same recorded backend; CPU and CUDA streams
are not expected to generate the same ensemble.  Calibration uses its own
NumPy generators with the seeds above, so adding one calibration cannot advance
another calibration’s stream.

For each event, compute on \(\mathcal V\) and generated \(G_{\rm in}\) the five
relative family losses defined in §4 above and select the checkpoint minimizing

\[
Q_{\rm val}^{(m)}=\sum_{s\in\mathcal S}\mathcal L_s^{\rm val,(m)}.
\]

Because each \(\mathcal L_s\) is already a squared discrepancy, the code is
exactly `Q_val = sum(L_val[s])`; it must not square these values again.  The
adaptive scales do not enter \(Q_{\rm val}\).  Save every event’s iteration,
five losses, \(Q_{\rm val}\), and validation-event index, plus the selected
iteration and complete model/optimizer configuration.

Checkpoint selection must never inspect \(G_{\rm out}\), lags
\(k\ge400\), admissibility, the joint density, the long-run calculation, or
\(\mathcal T_{\rm ev}\).

## 7. Independent validation classification

After the best checkpoint is frozen, classify it using a new \(B=400\),
length-800 generated ensemble from the independent classification stream;
discard raw `[0:400)` and use `[400:800)` as \(G_{\rm in}\).  Do not reuse the
ensemble that attained the minimum checkpoint score.

For each family,

\[
\mathcal L_{s,\rm ref}^{\rm val,in}
=\operatorname*{median}_{j=1,\ldots,9}
\mathcal L_s(\mathcal R_{j,\rm in},\mathcal V_{\rm in}),
\]

\[
r_s^{\rm val,(m)}
=
\frac{
\mathcal L_s(G_{\rm in}^{(m)},\mathcal V_{\rm in})
}{
\mathcal L_{s,\rm ref}^{\rm val,in}
},\qquad
r^{\rm val,(m)}=\max_{s\in\mathcal S}r_s^{\rm val,(m)}.
\]

Run \(m\) is validation-successful if and only if

\[
r^{\rm val,(m)}\le10.
\]

The inequality includes equality.  This is the current no-square-root loss
ratio; the obsolete criterion \(e/e_{\rm ref}\le3\) must not appear in code,
tables, figures, captions, or prose.  Record the accepted mask and report the
success count \(M/20\) before final results.

The nine reference blocks belong to this classifier/evaluator, not to the
trainer.  The trainer consumes only fitting statistics and \(\mathcal V\) for
checkpoint monitoring.  This separation resolves the missing-reference-block
problem in `experiments.py`.

## 8. Continuous final rollout and endpoint quantities

Final evaluation generates **all 20 retained checkpoints once**, irrespective
of validation success.  For each run, generate \(B_{\rm gen}=400\) continuous
trajectories of 1200 raw recurrent steps from zero hidden and cell states:

```text
raw [0:400)     generator warm-up, discarded
raw [400:800)   G_in, coincides with the training-scored window
raw [800:1200)  G_out, entirely beyond the training-time unroll
```

The recurrent state is not reset at raw step 800.  `lags` and retained rollout
length are independent parameters; no code may implement either window as
“the last `lags` steps.”  Window boundaries must be explicit slices as above.

For \(w\in\{\mathrm{in},\mathrm{out}\}\), recompute the final reference error
against the previously unopened target segment:

\[
\mathcal L_{s,\rm ref}^{\rm eval,w}
=\operatorname*{median}_{j=1,\ldots,9}
\mathcal L_s(\mathcal R_{j,w},\mathcal T_{{\rm ev},w}),
\]

and for every run compute

\[
r_{s,w}^{\rm eval,(m)}=
\frac{
\mathcal L_s(G_w^{(m)},\mathcal T_{{\rm ev},w})
}{
\mathcal L_{s,\rm ref}^{\rm eval,w}
},\qquad
r_w^{\rm eval,(m)}=\max_{s\in\mathcal S}r_{s,w}^{\rm eval,(m)}.
\]

The two endpoint ratios deliberately use their own window-specific finite-
sample denominators.  They describe accuracy within each window; they do not
classify runs.

A difference of these two ratios is **not** used as the degradation measure,
because it would subtract quantities with different denominators.  Instead use
one common symmetric scale per family:

\[
\overline{\mathcal L}_{s,\rm ref}^{\rm eval}
=\frac12\left(
\mathcal L_{s,\rm ref}^{\rm eval,in}
+\mathcal L_{s,\rm ref}^{\rm eval,out}
\right),
\]

\[
\Delta_s^{(m)}=
\frac{
\mathcal L_s(G_{\rm out}^{(m)},\mathcal T_{{\rm ev},\rm out})
-\mathcal L_s(G_{\rm in}^{(m)},\mathcal T_{{\rm ev},\rm in})
}{
\overline{\mathcal L}_{s,\rm ref}^{\rm eval}
}.
\]

This paired change is descriptive.  It is not a non-inferiority test and has no
pass/fail threshold.

For the Supplementary sliding-window diagnostic, use exactly 21 length-400
windows with zero-based starts `a = 400, 420, ..., 800` and Python slices
`[a:a+400]`.  Every generated trajectory, reference block, and final target
trajectory uses the same slice.  Adjacent windows overlap and are therefore
strongly dependent; the figure is descriptive and does not add 21 tests.  At
each start \(a\), define

\[
\mathcal L_{s,\rm ref}^{\rm eval,a}
=\operatorname*{median}_{j=1,\ldots,9}
\mathcal L_s(\mathcal R_{j,a},\mathcal T_{{\rm ev},a}),
\qquad
r_{s,a}^{(m)}
=\frac{\mathcal L_s(G_a^{(m)},\mathcal T_{{\rm ev},a})}
{\mathcal L_{s,\rm ref}^{\rm eval,a}}.
\]

Plot the median and IQR of \(r_{s,a}^{(m)}\) over the \(M\)
validation-successful runs.  The gray reference band is the IQR of the nine
block ratios
\(\mathcal L_s(\mathcal R_{j,a},\mathcal T_{{\rm ev},a})/
\mathcal L_{s,\rm ref}^{\rm eval,a}\), whose median is one.  Save all 20
run-level curves and accepted flags.

## 9. Reporting population and representative run

All run-level data are saved for all 20 checkpoints.  The main-text endpoint
table, long-lag summary, sliding-window summary, and admissibility summary are
conditioned on the \(M\) validation-successful runs, exactly as in §5.1.  The
Supplementary Material contains all 20 runs with unsuccessful runs marked.

For each family the main table reports

```text
median [Q1,Q3] r_eval,in
median [Q1,Q3] r_eval,out
median [Q1,Q3] Delta
n_+ = number of successful runs with Delta > 0, out of M
```

Raw model losses and the two reference denominators go to the Supplementary
Material.  The text may separately report the median and IQR of the run-wise
worst-family ratios \(r_{\rm in}^{\rm eval}\) and
\(r_{\rm out}^{\rm eval}\); these are not maxima of the displayed medians.

One representative run is selected before \(\mathcal T_{\rm ev}\) is opened.
Sort only the validation-successful runs by their stored checkpoint-monitoring
\(Q_{\rm val}\), ascending, and select zero-based element `M // 2`, i.e. the
one-based rank \(\lfloor M/2\rfloor+1\).  If \(M=0\), stop: no representative
figure or conditional main result is produced.

The **same** representative checkpoint is used for every single-run figure,
the smooth joint density, and the \(10^6\)-step stability diagnostic.  No seed
may be reselected after viewing final, out-of-window, density, long-lag, or
long-run results.

## 10. Long-lag diagnostic

The training loss continues to use the window-normalized correlation estimator
of §3.  Only the evaluation-only long-lag diagnostic uses pair-count
normalization over the combined retained interval
\(G_{\rm in}\cup G_{\rm out}\), of length 800.

For an ensemble of \(B\) standardized trajectories over this interval, set

\[
\bar x_i=\frac1{800B}\sum_{b=1}^{B}\sum_{t=0}^{799}x_i^{(b)}(t),
\qquad
\widehat\sigma_i^2
=\frac1{800B}\sum_{b=1}^{B}\sum_{t=0}^{799}
\bigl(x_i^{(b)}(t)-\bar x_i\bigr)^2,
\]

using float64 accumulation and population normalization.  The diagnostic
estimator is

\[
\widehat C_{ij}^{\rm pc}(k)
=
\frac{1}{\widehat\sigma_i\widehat\sigma_j}
\frac{1}{B(800-k)}
\sum_{b=1}^{B}\sum_{t=0}^{799-k}
\bigl(x_i^{(b)}(t)-\bar x_i\bigr)
\bigl(x_j^{(b)}(t+k)-\bar x_j\bigr).
\]

Means and standard deviations are computed separately for each ensemble whose
curve is measured; they are not borrowed from \(\mathcal F\).  The curve order
is fixed as `ACF_A, ACF_B, CCF_AB, CCF_BA`.  The window-normalized counterpart
would use divisor \(800B\) on this interval (and uses \(400B\) in the actual
training window); only this diagnostic uses \(B(800-k)\).

The diagnostic range is \(400\le k\le600\).  The upper endpoint is fixed by

\[
800-k\ge\frac{800-400}{2},
\]

so every reported lag retains at least 200 time pairs per trajectory.  Do not
plot or interpret \(k>600\).

Before opening \(\mathcal T_{\rm ev}\), draw 2000 trajectory-level bootstrap
blocks of size 400 from \(\mathcal R=\cup_j\mathcal R_j\), preserving each
trajectory’s complete 800-step dependence.  Compare each block with
\(\mathcal V\) over the same interval and compute

```text
M_ref = max over k=400,...,600 and over
        ACF_A, ACF_B, CCF_AB, CCF_BA
        of absolute(curve_block - curve_V)

eps_lag = empirical q_0.95(M_ref)
```

The bootstrap generator uses seed 7000.  No factor of three and no pointwise
band are added.  `ACF2` is not part of this maximum statistic.

For every retained checkpoint compute the analogous
\(M_{\rm SINN}^{(m)}\) from its \(B_{\rm gen}=400\) combined rollout against
\(\mathcal T_{\rm ev}\).  For the \(M\) successful runs report the median and
IQR of \(M_{\rm SINN}^{(m)}/\varepsilon_{\rm lag}\) and

\[
n_{\rm lag}=\#\{m:M_{\rm SINN}^{(m)}\le\varepsilon_{\rm lag}\}
\quad\text{out of }M.
\]

This is descriptive and does not alter the validation-success mask.  The main
correlation figure shows the representative run in four panels, labels the KMC
curve `Target`, ends at \(k=600\), marks \(k=400\), and uses the transferred
equal-width band \(C^{\rm target}(k)\pm\varepsilon_{\rm lag}\).  Its caption
states the run’s seed and \(M_{\rm SINN}/\varepsilon_{\rm lag}\), and that the
band was calibrated from \(\mathcal R\)-versus-\(\mathcal V\), not from final
data.

### 10.1 Supplementary joint-density visualization

Use the same representative checkpoint selected in §9.  Generate one separate
\(B_{\rm vis}=12000\) continuous 1200-step ensemble with the visualization
stream, discard raw `[0:400)`, and display only
\(G_{\rm out}={\tt [800:1200)}\).  Compare it with
\(\mathcal T_{{\rm ev},\rm out}\), also 12,000 trajectories.  Reference blocks
do not enter this figure.

Compute both bivariate KDEs in standardized \((x_A,x_B)\) coordinates with the
same frozen \(h_2\), and compute the two marginal KDEs with frozen \(h\).  For a
figure in physical \((\rho_A,\rho_B)\) coordinates, transform the axes using
\(\rho_i=s_i x_i+\mu_i\) and transform the bivariate density by the Jacobian
\(f_{\rho}=f_x/(s_A s_B)\); transform each marginal by \(1/s_i\).  The two
panels are labeled `Target` and `SINN`, never `Reference`.  This
visualization-only ensemble enters no numerical summary or decision.

## 11. Physical admissibility and \(10^6\)-step stability

After inverse standardization, do not clip, project, round to
\(N_{\rm site}^{-1}\), or otherwise repair generated densities.  For each
state define

\[
v=\max\{-\rho_A,-\rho_B,\rho_A-1,\rho_B-1,
\rho_A+\rho_B-1\}.
\]

A finite state violates the physical constraints when \(v>0\).  For a window
with \(B T\) state-time samples, save separately

```text
nonfinite_rate   = n_nonfinite / (B*T)
constraint_rate  = n_finite_with_v_gt_0 / (B*T)
invalid_rate     = (n_nonfinite + n_finite_with_v_gt_0) / (B*T)
max_exceedance   = max(max(v,0)) over finite states only
```

Thus non-finite states are counted as invalid and cannot disappear from the
denominator, while `max_exceedance` is never contaminated by NaN propagation.
If no finite state exists, store `max_exceedance = NaN` together with the
nonfinite count; this is an automatic failed diagnostic.  For each run compute
these quantities separately in \(G_{\rm in}\) and \(G_{\rm out}\).  Main
summaries are median and IQR over successful runs, the number of runs with at
least one invalid state, and the paired change in `invalid_rate`.

The stability calculation uses the representative run and 50 independently
generated trajectories.  Starting from zero recurrent states, discard 400
warm-up steps and then advance for \(10^6\) evaluated steps without reset.
At post-warm-up steps \(10^5,2\times10^5,\ldots,10^6\), retain the preceding
\(10^4\) steps for statistical summaries.  Accumulate finiteness, violation
count, maximum exceedance, and first violation/non-finite step **online over
all \(10^6\) evaluated steps**, not only the ten retained windows.

For each of the ten retained windows, save the float64 pooled mean and standard
deviation of \(x_A,x_B\), the finite physical-coordinate minima and maxima,
the three rates and maximum exceedance above, and the five relative family
losses against the fixed \(\mathcal T_{{\rm ev},\rm in}\) target statistics.
For these family statistics, split each continuously generated 10,000-step
window into 25 contiguous, non-overlapping 400-step blocks without resetting
the recurrent state.  Together with the 50 parallel trajectories this gives
1,250 analysis blocks of length \(T=400\), so the correlation estimator and
lag taper match the target endpoint protocol.  Use \(K=400\) and the frozen KDE
settings.  These ten losses describe drift of stationary statistics; they
have no reference-error denominator, no threshold, and no interpretation as
pathwise or \(10^6\)-step target accuracy.

The long-run calculation reports stability of this one validation-selected
checkpoint.  It supplies neither a success criterion nor evidence of
\(10^6\)-step target accuracy.

## 12. Pre-evaluation gates

Claude must implement and pass the following before any final generation or
access to \(\mathcal T_{\rm ev}\):

1. **Archive and split:** verify archive shape, finite values, checksum,
   disjoint trajectory indices, and the fixed standardization constants.
2. **Window fixture:** a labeled synthetic tensor must prove that raw slices
   `[400:800)` and `[800:1200)` are used and that recurrent state is not reset.
3. **CCF fixture:** asymmetric synthetic data must prove that both
   \(C_{AB}\) and \(C_{BA}\) are returned in the declared order.  Compare the
   FFT implementation against a direct brute-force sum; merely checking that
   two columns exist is insufficient because historical copies used opposite
   FFT conjugation conventions.
4. **Frame fixture:** verify orthonormality and the exact six non-coordinate
   directions; metadata must distinguish 3 non-coordinate from 4 total frames.
   Verify all six saved projection scales, unit fitting-block standard
   deviations after scaling, and that the KDE helper does not renormalize away
   those scales.
5. **KDE fixture:** verify the 513-point grid and the effective bandwidths
   \(h=0.091028210151304\) and
   \(h_2=0.13572088082974534\) against a direct Gaussian-KDE calculation.
   Repeating the same empirical samples to mimic different ensemble sizes must
   not change the bandwidth or density.  Fail if an effective bandwidth value
   or an exponent denominator is passed to the ambiguous legacy `bw` interface.
   Compare the differentiable chunked implementation used by the trainer with
   the direct NumPy formula at both bandwidths, including smooth, discrete,
   boundary, and collapsed fixtures.  A binned/FFT approximation does not
   satisfy this gate merely by agreeing on one large Gaussian sample.
6. **Loss-scale fixture:** verify numerically that stored \(\mathcal L_s\) is
   exactly the family-level squared relative discrepancy in §4,
   `Q_val == sum(L_val)`, and
   `r == L/L_ref`; fail if a square root, second square, or L1 term appears.
7. **Fresh-validation fixture:** two consecutive validation events must consume
   different tensors, while rerunning the same run and device reproduces the
   complete sequence.
8. **Independent-classification fixture:** assert that the classification
   generator and checkpoint-monitoring generator have different seeds and
   states.
9. **Reference fixture:** compute all five positive, finite
   \(\mathcal L_{s,\rm ref}^{\rm val,in}\) from the nine
   \(\mathcal R_j\)-versus-\(\mathcal V\) comparisons.  Record all five target
   family norms so weak statistics remain auditable, but do not introduce an
   additional magnitude cutoff such as \(\mathcal L_{s,\rm ref}>1\); no such
   cutoff is part of the classification rule or this gate.
10. **Burn-in/stationarity fixture:** reproduce every value and verdict in
    §3.1 with the declared halves, time indices, bootstrap seed, and distinction
    between bootstrap limits and fixed tolerances.
11. **Float precision:** on one real block compare reporting losses accumulated
   in float32 and float64; record the relative difference and require it below
   \(10^{-5}\).  All final stored statistics use float64 accumulation.
12. **Long-lag fixture:** verify the direct formula in §10, pair counts at
    \(k=400\) and \(k=600\), both
    CCF directions, trajectory-level resampling, 2000 bootstrap replicates, and
    the absence of any factor-three multiplier.
13. **Admissibility fixture:** hand-constructed valid, boundary, invalid, and
    non-finite states must yield the correct rates and maximum exceedance with
    no clipping.
14. **Manifest freeze:** write and checksum the configuration, projection,
    seed, checkpoint-selection, classification, reporting-population, and
    figure manifests.  Only after this gate may the final-target loader expose
    indices `[8000,20000)`.  Token existence alone is insufficient: the loader
    must verify that the token content matches the current manifest checksum,
    that the manifest archive checksum matches the archive being opened, and
    that the recorded specification and source checksums match the current
    frozen pipeline.  A token from a superseded gate implementation must not
    unlock the final target.

Any failed gate stops the pipeline with a nonzero exit status.  A blocked or
failed gate may not be converted to a warning.

## 13. Deliverables and file contracts

The implementation is separated into stages; one monolithic script is not
acceptable.

| stage | required responsibility |
|---|---|
| trainer | read \(\mathcal F\) and \(\mathcal V\); train 20 runs; save all validation histories and retained checkpoints; never expose final indices |
| classifier | independently regenerate \(G_{\rm in}\); construct nine validation reference errors; save five ratios, worst ratio, and accepted mask |
| calibration | validation-side stationarity summaries and 2000-replicate long-lag envelope; never read final indices |
| final evaluator | evaluate all 20 checkpoints once on one continuous 1200-step rollout; save both windows, all losses, ratios, common-scale changes, long-lag values, and admissibility |
| stability evaluator | run the one frozen representative checkpoint for \(10^6\) post-warm-up steps and accumulate online diagnostics |
| figure/table builder | read only saved evaluation products; never regenerate model output or reclassify runs |

Every result file stores at least: archive checksum, code revision, complete
configuration, all relevant seeds, run id, checkpoint iteration,
\(Q_{\rm val}\), accepted flag, \(B_{\rm gen}\), target size, exact time slices,
family order, frame convention, grid, bandwidths, estimator names, and source
file checksums.  Published CLI defaults must equal the frozen values above;
the user must not have to remember hidden overrides such as `--seeds 20` or
`--eval-traj 400`.

## 14. Main-text and Supplementary Material allocation

The complete Section 5 must remain within five typeset pages.  After §5.1 is
finally paginated, §5.2 receives the remaining space and is capped at 2.5
pages.  The main text is kept to the temporal-extension argument:

1. one compact setup paragraph and the microstate/coarse-density overview;
2. validation success count \(M/20\) and the continuous-window protocol;
3. one five-row endpoint table containing
   \(r_{\rm in}\), \(r_{\rm out}\), \(\Delta\), and \(n_+/M\);
4. one four-panel correlation figure to \(k=600\), with the transferred
   validation-side envelope;
5. a short admissibility result and one sentence reporting the \(10^6\)-step
   stability diagnostic; and
6. restrained conclusions within the claim boundary of §1.

Move the following to the Supplementary Material:

- full numerical setup and seed/configuration tables;
- burn-in and empirical-stationarity verification;
- all-20 run-level validation and final-evaluation tables;
- raw model losses and the two window-specific reference denominators;
- five-panel sliding-window error plot (21 overlapping windows);
- bivariate density and marginal plots using
  \(B_{\rm vis}=12000\) for the representative run only;
- detailed \(10^6\)-step checkpoint table and any optional stability figure;
- pair-count estimator and bootstrap details; and
- fixtures, QA summary, and provenance hashes.

The \(B_{\rm vis}=12000\) density ensemble is visualization-only.  It enters no
loss ratio, success classification, endpoint table, long-lag statistic,
representative-run selection, or admissibility summary.

The default long-run deliverable is one compact Supplementary table.  A
separate stability figure is optional and should be added only if it reveals a
time-dependent feature that cannot be communicated by the table.  Regardless
of outcome, the main text reports the diagnostic in at most one sentence and
does not interpret it as long-time target accuracy.

If the combined section still exceeds five pages, retain only the five-row
endpoint table and four-panel correlation figure in the §5.2 main text.  Move
the microstate/coarse-density overview to the Supplementary Material before
shortening the selection safeguards, endpoint definitions, or claim boundary.

## 15. Resolved FPU-to-LLV migration questions

1. The current §5.1 quantitative family-ratio evaluation used
   \(B_{\rm gen}=400\) against a 12,000-trajectory final target.  The
   `experiments.py --eval-traj=4000` default is a legacy path and did not
   produce the published boxplot or table.
2. The LLV quantitative endpoint also uses \(B_{\rm gen}=400\), matching every
   reference block.  There is no `--eval-traj` ambiguity in the new pipeline.
3. The published run count is encoded as the explicit list `0,...,19`, and the
   program asserts exactly 20 unique training seeds.
4. Reference blocks are handled by the classifier/calibration/final evaluator,
   not by the trainer.
5. Correlation lag count, raw rollout length, and scored slices are separate
   named fields and are never inferred from one another.
6. A “last recorded validation value” such as historical `x_floor` is neither
   the retained-checkpoint value nor a publishable diagnostic and is not
   implemented.
