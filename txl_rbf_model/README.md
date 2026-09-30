# TXL-ACAM v4 Model — Implementation Notes

This document describes the TXL-ACAM (RRAM-CMOS crossbar) model as implemented in
`txl_model_v4.py` and used by `HardwareTXLClassifier`
(as found in `artificial_txl_hardware_core.py`). It covers the hardware constants, the
threshold/encode/decode equations, the per-cell matching-window response, the
matchline (row-level) equations, the `TXLCell` / `TXLArray` classes, and the
classifier that wraps them.

---

## 1. Hardware constants

All physical parameters live in one dictionary, `HW`:

| Symbol | Key | Value | Meaning |
|---|---|---|---|
| $V_{DD}$ | `V_DD` | 3.3 V | Supply voltage |
| $V_{tn}$ | `V_tn` | 1.5 V | nMOS threshold magnitude |
| $V_{tp}$ | `V_tp` | 1.5 V | \|pMOS threshold\| magnitude |
| $\beta_n$ | `beta_n` | 1.00 | nMOS $\mu C_{ox} W/L$ (normalised) |
| $\beta_p$ | `beta_p` | 0.60 | pMOS $\mu C_{ox} W/L$ (normalised) |
| $I_s$ | `I_s` | $3\times10^{-5}$ A | Switching current |
| $R_B$ | `R_B` | 200 kΩ | Fixed nMOS source degeneration |
| $V_{ML}$ | `V_ML` | 3.3 V | Matchline drive voltage |
| $R_{lim}$ | `R_lim` | 50 kΩ | Matchline current limiter |
| $R_{min}$ | `R_min` | 1 kΩ | RRAM minimum resistance |
| $R_{max}$ | `R_max` | 1 MΩ | RRAM maximum resistance |
| $P_{IDO}$ | `P_IDO` | 0.90 | Reliable-zone inner confidence |
| $P_{OOD}$ | `P_OOD` | 0.99 | OOD-zone outer confidence |

Two derived quantities are computed once from `HW` and reused everywhere:

$$
k_r = \sqrt{\dfrac{\beta_p}{\beta_n}}
\qquad\qquad
V_{TH0} = \dfrac{V_{tn} + k_r\,(V_{DD}-V_{tp})}{1+k_r}
\qquad\qquad
A = \dfrac{1+k_r}{I_s\,k_r}
$$

$k_r$ is the pMOS/nMOS strength ratio, $V_{TH0}$ is the threshold voltage at zero
RRAM bias, and $A$ (units Ω/V) is the resistance-to-voltage sensitivity of the
cell — it sets how many ohms of RRAM state correspond to one volt of encoded
uncertainty.

---

## 2. Threshold / encode / decode equations

### 2.1 Threshold voltage from resistance

$$
V_{TH}(R_M) \;=\; V_{TH0} \;+\; \frac{I_s\,(R_B - k_r R_M)}{1+k_r}
$$

Increasing the pMOS RRAM resistance $R_M$ lowers $V_{TH}$; increasing the fixed
nMOS resistance $R_B$ raises it.

### 2.2 Encoding a prototype $(\mu,\sigma)$ into RRAM state

Each cell stores a matching window as a pair of resistances, $R_{M1}$ (lower
threshold) and $R_{M2}$ (upper threshold):

$$
R_{M1} = \operatorname{clip}\!\Big(\tfrac{R_B}{k_r} - A(\mu - \sigma - V_{TH0}),\; R_{min}, R_{max}\Big)
\qquad
R_{M2} = \operatorname{clip}\!\Big(\tfrac{R_B}{k_r} - A(\mu + \sigma - V_{TH0}),\; R_{min}, R_{max}\Big)
$$

The programmed gap $R_{M1}-R_{M2} = 2A\sigma$ is the hardware's physical encoding
of uncertainty: a wider window (larger $\sigma$) is a larger resistance gap
between the two RRAM cells.

### 2.3 Decoding RRAM state back to $(\mu,\sigma)$

$$
V_{lo} = V_{TH}(R_{M1}) \qquad V_{hi} = V_{TH}(R_{M2})
\qquad\qquad
\mu = \frac{V_{hi}+V_{lo}}{2} \qquad \sigma = \frac{V_{hi}-V_{lo}}{2}
$$

Encode and decode are exact inverses of each other everywhere the resistances
are not clipped against $R_{min}/R_{max}$.

---

## 3. Cell response — double-sigmoid matching window

Each TXL cell's response to an input voltage $x$ is a **plateau-shaped window**
built from the product of two logistic (sigmoid) transitions, one rising at the
window's lower edge and one falling at its upper edge. This is the analytical
shape used for every classification and reliability decision in the model.

Given a cell's decoded $(\mu,\sigma)$ and four shape parameters — edge
steepness $k_{lo}, k_{hi}$, a width scale $w_{scale}$, and a center shift
$\Delta_c$ — the effective window is:

$$
\sigma_{eff} = \max(w_{scale}\cdot\sigma,\ 10^{-9})
\qquad
\mu_{eff} = \mu + \Delta_c
$$

$$
V_{lo}^{eff} = \mu_{eff} - \sigma_{eff}
\qquad
V_{hi}^{eff} = \mu_{eff} + \sigma_{eff}
\qquad
k_{lo} = \frac{k_{edge,lo}}{\sigma_{eff}}
\qquad
k_{hi} = \frac{k_{edge,hi}}{\sigma_{eff}}
$$

$$
g(x) \;=\; \sigma\!\big(k_{lo}\,(x - V_{lo}^{eff})\big)\;\cdot\;\sigma\!\big(k_{hi}\,(V_{hi}^{eff} - x)\big)
\qquad\text{where } \sigma(z) = \frac{1}{1+e^{-z}}
$$

The first factor rises from 0 toward 1 as $x$ crosses $V_{lo}^{eff}$; the second
falls from 1 toward 0 as $x$ crosses $V_{hi}^{eff}$. Their product is a smooth
plateau: near 0 outside the window, near its peak value inside it, with two
shoulders whose steepness is set by $k_{edge,lo}$/$k_{edge,hi}$.

- $\mu,\sigma$ come from the cell's programmed RRAM state (Section 2.3) — they
  are never set independently of the resistance encoding.
- $k_{edge,lo}$, $k_{edge,hi}$, $w_{scale}$, $\Delta_c$ are configured per cell
  at construction time (uniformly, by whatever builds the array) and require no
  external data — they are analytical shape parameters, defaulted to
  $k_{edge,lo}=k_{edge,hi}=2.0$, $w_{scale}=1.0$, $\Delta_c=0.0$, and may be
  swept empirically against a dataset's reliable-coverage / OOD-rejection
  behaviour.
- At $x=\mu_{eff}$ (with $k_{lo}=k_{hi}$), $g(\mu_{eff}) = \sigma(k_{edge})^2$ —
  the window's peak value is set by the chosen edge steepness rather than
  fixed at 1.

---

## 4. Matchline equations

A **row** (`TXLArray`) is one class's matchline: $D$ cells in parallel, one per
feature.

### 4.1 KCL current accumulation

$$
I_m^{ML} \;=\; \sum_{i=1}^{D} \frac{V_{ML}}{R_{lim}}\, g_{m,i}(x_i)
$$

Each cell that matches its input contributes a fixed current increment
$V_{ML}/R_{lim}$ scaled by its response $g_{m,i}$; the row sums these currents
by Kirchhoff's current law.

### 4.2 Normalised match score

$$
\hat{\mathcal{N}}_m \;=\; \operatorname{clip}\!\Big(\frac{I_m^{ML}}{I_{max}},\ 0,\ 1\Big)
\qquad\text{where}\qquad
I_{max} = \frac{D\,V_{ML}}{R_{lim}}
$$

### 4.3 Unified similarity-to-distance mapping

A single distance-like quantity, derived purely from $\hat{\mathcal{N}}_m$,
drives both which row wins and how reliable that win is:

$$
\tilde{d}^2 \;=\; -2D\,\ln\hat{\mathcal{N}}_m
$$

Because this is a strictly monotonically decreasing function of
$\hat{\mathcal{N}}_m$, the row with the highest match score is always the row
with the lowest $\tilde{d}^2$:

$$
\arg\max_m \hat{\mathcal{N}}_m \;\equiv\; \arg\min_m \tilde{d}^2_m \qquad\text{(always, by construction)}
$$

so ranking (which class wins) and reliability zoning (how confident that win
is) are guaranteed to agree — they are two views of the same accumulated
matchline current, not two independently computed quantities.

### 4.4 Reliability thresholds

Each row carries two thresholds, $\tau_{IDO}$ and $\tau_{OOD}$, that partition
$\tilde{d}^2$ into three zones:

$$
\text{zone}(x) = \begin{cases}
\text{RELIABLE} & \tilde{d}^2 \le \tau_{IDO} \\
\text{UNRELIABLE} & \tau_{IDO} < \tilde{d}^2 \le \tau_{OOD} \\
\text{OOD} & \tilde{d}^2 > \tau_{OOD}
\end{cases}
$$

Thresholds can be set two ways:

- **Theoretical.** $\tau_{IDO} = \chi^2_{P_{IDO}}(D)$, $\tau_{OOD} =
  \chi^2_{P_{OOD}}(D)$ — the standard chi-squared quantiles for $D$ degrees of
  freedom, computed once when the row (`TXLArray`) is built.
- **Calibrated.** `TXLArray.calibrate_thresholds(X_calib)`
  replaces both thresholds with percentiles of that row's own $\tilde{d}^2$
  distribution, evaluated over a calibration sample:
  $$
  \tau_{IDO} = \operatorname{percentile}_{P_{IDO}\cdot 100}\big(\tilde{d}^2(X_{calib})\big)
  \qquad
  \tau_{OOD} = \operatorname{percentile}_{P_{OOD}\cdot 100}\big(\tilde{d}^2(X_{calib})\big)
  $$
  This is the default behaviour of the classifier (`calibrate_thresholds=True`)
  and lets each row's decision boundary reflect its own measured match-score
  distribution rather than a fixed theoretical reference.

---

## 5. `TXLCell`

One `TXLCell` object represents one physical crossbar cell: a single feature's
matching window, stored as a resistance pair.

**State**
- `R_M1`, `R_M2` — the two programmed resistances (Ω).
- `k_edge_lo`, `k_edge_hi`, `w_scale`, `center_shift` — the window's shape
  parameters (Section 3), stored per cell.
- `hw` — the hardware-constant dictionary this cell is encoded against.

**Derived properties** (computed on demand from `R_M1`/`R_M2`)
- `mu`, `sigma` — decoded prototype mean/width (Section 2.3).
- `V_lo`, `V_hi` — decoded threshold voltages.
- `gap` — $R_{M1}-R_{M2}$.

**Construction**
- `TXLCell.from_params(mu, sigma, hw, k_edge_lo, k_edge_hi, w_scale, center_shift, ...)`
  encodes a prototype statistic directly into a new cell's RRAM state.

**Behaviour**
- `response(x)` evaluates $g(x)$ (Section 3) at the cell's current programmed
  state.
- `adapt(x, eta)` performs an on-the-fly exponential-moving-average update:
  $$
  \mu_{new} = (1-\eta)\mu + \eta x \qquad
  \sigma_{new}^2 = (1-\eta)\sigma^2 + \eta(x-\mu_{new})^2
  $$
  then re-encodes $(\mu_{new},\sigma_{new})$ through the same `txl_encode`
  used at construction — only `R_M1`/`R_M2` move; the shape parameters are
  untouched.

---

## 6. `TXLArray`

One `TXLArray` is one class's matchline: an ordered list of `D` `TXLCell`
objects (one per feature), plus the row-level decision logic.

**State**
- `cells` — the `D` `TXLCell` objects.
- `D` — number of cells (= feature dimension for this classifier).
- `tau_IDO`, `tau_OOD` — reliability thresholds (Section 4.4), initialised from
  the theoretical $\chi^2$ quantiles at construction and optionally replaced by
  empirical calibration afterward.

**Methods**
- `cell_responses(x_vec)` — per-cell $g_{m,i}(x_i)$ for a query vector.
- `I_ML_value(x_vec)` — row current $I_m^{ML}$ (Section 4.1).
- `N_hat(x_vec)` — normalised match score $\hat{\mathcal{N}}_m$ (Section 4.2).
- `d2_tilde(x_vec)` — the unified distance $\tilde{d}^2$ (Section 4.3); this is
  the one quantity the zone decision uses.
- `d2(x_vec)` — alias for `d2_tilde`, kept as the row's public "distance"
  accessor.
- `calibrate_thresholds(X_calib)` — empirical threshold calibration
  (Section 4.4).
- `zone(x_vec)` — returns `(zone_label, N_hat, d2_tilde)` for a query, applying
  the three-way partition in Section 4.4.
- `sweep_alpha(x_far, x_exact, n)` — linearly interpolates a query from a far
  point to an exact match over `n` steps and returns `I_ML`, `N_hat`, and
  `d2_tilde` along the sweep, for characterisation plots.

---

## 7. `HardwareTXLClassifier`

The classifier wraps one `TXLArray` per class and exposes training, inference,
adaptation, and new-class allocation.

### 7.1 Structure

- One class label ↦ one `TXLArray` row ↦ `D` `TXLCell` objects (one per input
  feature).
- Uniform response-shape parameters (`k_edge_lo`, `k_edge_hi`, `w_scale`,
  `center_shift`) are set once at classifier construction and threaded through
  every `TXLCell` it builds.
- `calibrate_thresholds` (bool, default `True`) controls whether each row's
  `tau_IDO`/`tau_OOD` are the theoretical $\chi^2$ quantiles or the empirical
  percentiles described in Section 4.4.

### 7.2 Training a class row (`fit_class` / `fit`)

For a batch of samples belonging to one class:

1. Compute the feature-domain mean $\mu_x$ and sample standard deviation
   $\sigma_x$ per feature.
2. Map to voltage domain: $\mu_v = V_{min} + \mu_x\,(V_{max}-V_{min})$,
   $\sigma_v = \sigma_x\,(V_{max}-V_{min})$, then clip $\sigma_v$ to
   $[\sigma_{v,min}, \sigma_{v,max}]$.
3. Encode each feature's $(\mu_v,\sigma_v)$ into a `TXLCell` (`txl_encode`,
   Section 2.2) — this is the row's `TXLArray`.
4. If `calibrate_thresholds` is set, calibrate `tau_IDO`/`tau_OOD` on the
   voltage-mapped training samples for this class.

All of this — feature statistics, voltage-domain values, per-cell RRAM states,
decoded values, and clipping flags — is retained in a `PrototypeStats` record
per class for inspection and plotting.

### 7.3 Inference

- `inference_full(x)` — evaluates every class row against a voltage-mapped
  query and returns, per row: `zone`, `N_hat`, `I_ML`, `d2`, plus the
  classifier-wide winner (by highest `N_hat`, equivalently lowest `d2`, by
  Section 4.3), the runner-up, the winning margin, and the row's reliability
  zone.
- `predict(X)` — batched predicted labels.
- `decision_scores(X, score_type)` — scalar ID/OOD-style scores per sample
  (`best_N_hat`, `best_d2`, `best_I_ML`, or their sign-adjusted forms) for use
  in OOD-detection metrics such as AUROC.

### 7.4 Adapting an existing class (`adapt_class`)

Given new support samples for a class already trained:

1. Compute the support batch's mean/std.
2. Blend with the existing prototype's mean/std using `adapt_ratio`
   ($r\in[0,1]$), in one of three modes:
   - `pooled_ema` — mean blend plus an EMA-style pooled-variance blend.
   - `sigma_ema` — mean and standard deviation both blended linearly.
   - `replace` — the prototype is refit from the support samples only.
3. Re-encode the blended $(\mu,\sigma)$ into a new `TXLArray` (new `TXLCell`
   states — this is the RRAM re-programming step).
4. Carry the row's existing `tau_IDO`/`tau_OOD` forward rather than
   recalibrating, since adaptation is treated as a small update to an
   already-calibrated row.

### 7.5 Allocating a new class (`allocate_new_class`)

Trains a brand-new `TXLArray` row from few-shot support samples for a label
that does not yet exist in the classifier (refuses to silently overwrite an
existing label unless `allow_overwrite=True`). This is how an
out-of-distribution stream becomes a new recognised class: once enough support
samples are available, `allocate_new_class` runs the same encoding pipeline as
`fit_class` for the new label, expanding the classifier by one row (`D`
additional `TXLCell` objects).
