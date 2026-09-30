# -*- coding: utf-8 -*-
"""
txl_model_v4.py  --  TXL-ACAM Core Model Definition
===========================================================
"""

# -- Standard library --------------------------------------------------------
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

# -- Third-party --------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2

# ─────────────────────────────────────────────────────────────────────────────
# SEC 0  HARDWARE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HW: Dict = dict(
    V_DD   = 3.3,    # Supply voltage                    [V]
    V_tn   = 1.5,    # nMOS threshold magnitude          [V]
    V_tp   = 1.5,    # |pMOS threshold| magnitude        [V]
    beta_n = 1.00,   # nMOS uCox*W/L (normalised)
    beta_p = 0.60,   # pMOS uCox*W/L (normalised)
    I_s    = 3e-5,   # Switching current                 [A]
    R_B    = 200e3,  # Fixed nMOS source degeneration    [Ohm]
    V_ML   = 3.3,    # Matchline drive voltage           [V]
    R_lim  = 50e3,   # Matchline current limiter         [Ohm]
    R_min  = 1e3,    # RRAM minimum resistance           [Ohm]
    R_max  = 1e6,    # RRAM maximum resistance           [Ohm]
    P_IDO  = 0.90,   # Reliable inner confidence
    P_OOD  = 0.99,   # OOD outer confidence
)

def _derive(hw: Dict) -> Tuple[float, float, float]:
    """Return (k_r, V_TH0, A) for any HW parameter set."""
    k_r   = float(np.sqrt(hw['beta_p'] / hw['beta_n']))
    V_TH0 = float((hw['V_tn'] + k_r*(hw['V_DD'] - hw['V_tp'])) / (1.0 + k_r))
    A     = float((1.0 + k_r) / (hw['I_s'] * k_r))
    return k_r, V_TH0, A

K_R, V_TH0, A_HW = _derive(HW)

# -- Colour palette -----------------------------------------------------------
_C = dict(
    rm1="#2563EB", rm2="#DC2626", window="#7C3AED",
    v4="#059669", v3="#9CA3AF", ml="#0891B2", norm="#15803D",
    model="#374151", unrel="#D97706", ood="#BE123C",
    bg="#FFFFFF", neutral="#374151", cell="#6741D9",
)

# ─────────────────────────────────────────────────────────────────────────────
# SEC 1  CORE THRESHOLD / ENCODE / DECODE EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def txl_V_TH(R_M, R_B=None, hw=HW):
    """V_TH = V_TH0 + I_s*(R_B - k_r*R_M) / (1+k_r)"""
    if R_B is None: R_B = hw['R_B']
    k_r, V_TH0_, _ = _derive(hw)
    return V_TH0_ + hw['I_s'] * (R_B - k_r * np.asarray(R_M, float)) / (1.0 + k_r)

def txl_encode(mu, sigma, hw=HW):
    """(mu,sigma) -> (R_M1, R_M2), both clipped to [R_min, R_max].
    Gap R_M1-R_M2 = 2*A*sigma (uncertainty encoder) -- unaffected by the
    response-shape change in SEC 2 below."""
    k_r, V_TH0_, A_ = _derive(hw)
    base = hw['R_B'] / k_r
    R_M1 = np.clip(base - A_*(np.asarray(mu,float) - np.asarray(sigma,float) - V_TH0_),
                   hw['R_min'], hw['R_max'])
    R_M2 = np.clip(base - A_*(np.asarray(mu,float) + np.asarray(sigma,float) - V_TH0_),
                   hw['R_min'], hw['R_max'])
    return R_M1, R_M2

def txl_decode(R_M1, R_M2, hw=HW):
    """Inverse of txl_encode. mu = (V_hi+V_lo)/2, sigma = (V_hi-V_lo)/2."""
    V_lo = txl_V_TH(R_M1, hw['R_B'], hw)
    V_hi = txl_V_TH(R_M2, hw['R_B'], hw)
    return (V_hi + V_lo)/2.0, (V_hi - V_lo)/2.0

# ─────────────────────────────────────────────────────────────────────────────
# SEC 2  CELL RESPONSE  --  parametric double-sigmoid matching window
# ─────────────────────────────────────────────────────────────────────────────
#
#   sigma_eff = w_scale * sigma
#   V_lo_eff  = mu + center_shift - sigma_eff
#   V_hi_eff  = mu + center_shift + sigma_eff
#   k_lo      = k_edge_lo / sigma_eff
#   k_hi      = k_edge_hi / sigma_eff
#   g(x)      = sigmoid(k_lo*(x - V_lo_eff)) * sigmoid(k_hi*(V_hi_eff - x))

def _sigmoid(z):
    z = np.clip(np.asarray(z, float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))

def txl_window_v4(x, mu, sigma, k_edge_lo=2.0, k_edge_hi=2.0,
                   w_scale=1.0, center_shift=0.0):
    """v4 double-sigmoid matching window."""
    sigma_eff = np.maximum(w_scale * np.asarray(sigma, float), 1e-9)
    mu_eff    = np.asarray(mu, float) + center_shift
    V_lo_eff  = mu_eff - sigma_eff
    V_hi_eff  = mu_eff + sigma_eff
    k_lo      = k_edge_lo / sigma_eff
    k_hi      = k_edge_hi / sigma_eff
    x         = np.asarray(x, float)
    return _sigmoid(k_lo * (x - V_lo_eff)) * _sigmoid(k_hi * (V_hi_eff - x))

def txl_gauss_v3(x, mu, sigma):
    """ideal-Gaussian response. Kept as a labelled reference for the
    response-shape ablation in the main experiment script."""
    return np.exp(-0.5 * ((np.asarray(x,float) - mu) /
                           np.maximum(sigma, 1e-12))**2)

# ─────────────────────────────────────────────────────────────────────────────
# SEC 3  MATCHLINE EQUATIONS: I_ML, N_hat, unified similarity-to-distance map
# ─────────────────────────────────────────────────────────────────────────────

def txl_I_ML(x_vec, mu_vec, sigma_vec, hw=HW, response_fn=txl_window_v4, **shape_kw):
    """KCL matchline accumulation: I_ML = sum_i (V_ML/R_lim)*g_i(x_i)."""
    g = response_fn(x_vec, mu_vec, sigma_vec, **shape_kw) if response_fn is txl_window_v4 \
        else response_fn(x_vec, mu_vec, sigma_vec)
    return float((hw['V_ML'] / hw['R_lim']) * g.sum())

def txl_N_hat(I_ML_val, D, hw=HW):
    """N_hat = I_ML / I_ML_max, I_ML_max = D*V_ML/R_lim."""
    I_max = D * hw['V_ML'] / hw['R_lim']
    return float(np.clip(I_ML_val / I_max, 0.0, 1.0))

def txl_d2_tilde(N_hat, D):
    """The distance-like quantity used for array-level reliability in txl model.
    Derived purely from N_hat (the row's normalised accumulated current) --
    never from an independent per-cell sum of squares.

        d~^2 = -2 * D * ln(N_hat)

    This is strictly monotonically DECREASING in N_hat, so
        argmax_m N_hat_m  ==  argmin_m d~^2_m         (always, by construction)
    """
    N = np.clip(np.asarray(N_hat, float), 1e-12, 1.0)
    return -2.0 * D * np.log(N)

def txl_d2_legacy(x_vec, mu_vec, sigma_vec):
    """independently-computed diagonal Mahalanobis^2. Used as a
    labelled reference for the ranking-consistency."""
    x_vec, mu_vec, sigma_vec = map(np.asarray, (x_vec, mu_vec, sigma_vec))
    return float(((x_vec - mu_vec)**2 / np.maximum(sigma_vec, 1e-12)**2).sum())

# ─────────────────────────────────────────────────────────────────────────────
# SEC 4  TXLCell
# ─────────────────────────────────────────────────────────────────────────────

class TXLCell:
    """
    Single TXL-ACAM cell.

    State stored as (R_M1, R_M2) in Ohms using the active hw dict.
    mu, sigma, V_lo, V_hi are always derived from the resistance state.

    k_edge_lo, k_edge_hi, w_scale, center_shift are the v4 response-shape
    parameters
    """

    def __init__(self, R_M1: float, R_M2: float,
                 hw: Dict = None,
                 k_edge_lo: float = 2.0, k_edge_hi: float = 2.0,
                 w_scale: float = 1.0, center_shift: float = 0.0,
                 name: str = "TXL",
                 sweep_param: str = "RM2",
                 sweep_val:   int = 0,
                 source:      str = "analytical_v4"):
        self.hw           = hw if hw is not None else HW
        self.R_M1          = float(R_M1)
        self.R_M2          = float(R_M2)
        self.k_edge_lo      = float(k_edge_lo)
        self.k_edge_hi      = float(k_edge_hi)
        self.w_scale        = float(w_scale)
        self.center_shift   = float(center_shift)
        self.name          = name
        self.sweep_param   = sweep_param
        self.sweep_val     = sweep_val
        self.source        = source

    @classmethod
    def from_params(cls, mu: float, sigma: float,
                    hw: Dict = None, **kwargs) -> "TXLCell":
        """Create from prototype statistics (mu, sigma)."""
        if hw is None: hw = HW
        rm1, rm2 = txl_encode(np.array([mu]), np.array([sigma]), hw)
        kwargs.setdefault('source', 'analytical_v4')
        return cls(float(rm1[0]), float(rm2[0]), hw=hw, **kwargs)

    @property
    def mu(self) -> float:
        m, _ = txl_decode(self.R_M1, self.R_M2, self.hw)
        return float(m)

    @property
    def sigma(self) -> float:
        _, s = txl_decode(self.R_M1, self.R_M2, self.hw)
        return max(float(s), 1e-9)

    @property
    def V_lo(self) -> float:
        return float(txl_V_TH(self.R_M1, self.hw['R_B'], self.hw))

    @property
    def V_hi(self) -> float:
        return float(txl_V_TH(self.R_M2, self.hw['R_B'], self.hw))

    @property
    def gap(self) -> float:
        return self.R_M1 - self.R_M2

    def response(self, x, response_fn=None) -> np.ndarray:
        """v4 double-sigmoid response by default."""
        if response_fn is None or response_fn is txl_window_v4:
            return txl_window_v4(np.asarray(x, float), self.mu, self.sigma,
                                  self.k_edge_lo, self.k_edge_hi,
                                  self.w_scale, self.center_shift)
        return response_fn(np.asarray(x, float), self.mu, self.sigma)

    def adapt(self, x, eta: float = 0.15):
        """On-the-fly EMA update of (mu,sigma), then re-encode through the
        txl_encode used at construction."""
        mu_new = (1 - eta) * self.mu + eta * x
        sig_new_sq = (1 - eta) * self.sigma**2 + eta * (x - mu_new)**2
        sig_new = max(float(np.sqrt(sig_new_sq)), 1e-9)
        rm1, rm2 = txl_encode(np.array([mu_new]), np.array([sig_new]), self.hw)
        self.R_M1, self.R_M2 = float(rm1[0]), float(rm2[0])

    def __repr__(self):
        return (f"TXLCell({self.sweep_param}={self.sweep_val}  "
                f"V_lo={self.V_lo:.3f}  V_hi={self.V_hi:.3f}  "
                f"mu={self.mu:.3f}  sigma={self.sigma:.3f}  "
                f"k_lo={self.k_edge_lo:.1f} k_hi={self.k_edge_hi:.1f} "
                f"w={self.w_scale:.2f} shift={self.center_shift:+.3f}  "
                f"src={self.source})")


# ─────────────────────────────────────────────────────────────────────────────
# SEC 5  TXLArray  (KCL + single-method three-zone decision)
# ─────────────────────────────────────────────────────────────────────────────

class TXLArray:
    """1-D matchline of D TXLCell objects."""

    def __init__(self, cells: List[TXLCell], name: str = "ML"):
        self.cells   = cells
        self.name    = name
        self.D       = len(cells)
        self.hw      = cells[0].hw if cells else HW
        self.tau_IDO = float(chi2.ppf(self.hw['P_IDO'], df=self.D))
        self.tau_OOD = float(chi2.ppf(self.hw['P_OOD'], df=self.D))

    def cell_responses(self, x_vec) -> np.ndarray:
        return np.array([c.response(float(xi))
                         for c, xi in zip(self.cells, x_vec)])

    def I_ML_value(self, x_vec) -> float:
        return float((self.hw['V_ML'] / self.hw['R_lim']) *
                     self.cell_responses(x_vec).sum())

    @property
    def I_ML_max(self) -> float:
        return float(self.D * self.hw['V_ML'] / self.hw['R_lim'])

    def N_hat(self, x_vec) -> float:
        return float(np.clip(self.I_ML_value(x_vec) / self.I_ML_max, 0.0, 1.0))

    def d2_tilde(self, x_vec) -> float:
        """ similarity-derived distance used for reliability zoning."""
        return float(txl_d2_tilde(self.N_hat(x_vec), self.D))

    def d2(self, x_vec) -> float:
        return self.d2_tilde(x_vec)

    def d2_legacy(self, x_vec) -> float:
        """independent diagonal Mahalanobis^2."""
        mu    = np.array([c.mu    for c in self.cells])
        sigma = np.array([c.sigma for c in self.cells])
        return txl_d2_legacy(x_vec, mu, sigma)

    def calibrate_thresholds(self, X_calib: np.ndarray) -> None:
        """Empirical threshold calibration"""
        if len(X_calib) == 0:
            return
        d2_pool = np.array([self.d2_tilde(x) for x in X_calib])
        self.tau_IDO = float(np.percentile(d2_pool, self.hw['P_IDO'] * 100))
        self.tau_OOD = float(np.percentile(d2_pool, self.hw['P_OOD'] * 100))

    def zone(self, x_vec) -> Tuple[str, float, float]:
        n_val  = self.N_hat(x_vec)
        d2_val = txl_d2_tilde(n_val, self.D)
        if   d2_val <= self.tau_IDO: z = 'RELIABLE'
        elif d2_val <= self.tau_OOD: z = 'UNRELIABLE'
        else:                         z = 'OOD'
        return z, n_val, d2_val

    def sweep_alpha(self, x_far, x_exact, n=150) -> dict:
        alphas  = np.linspace(0, 1, n)
        I_vals  = [self.I_ML_value((1-a)*x_far + a*x_exact) for a in alphas]
        N_vals  = [self.N_hat((1-a)*x_far + a*x_exact)      for a in alphas]
        d2_vals = [self.d2_tilde((1-a)*x_far + a*x_exact)   for a in alphas]
        return dict(alpha=alphas, I_ML=np.array(I_vals),
                    N_hat=np.array(N_vals), d2=np.array(d2_vals))
