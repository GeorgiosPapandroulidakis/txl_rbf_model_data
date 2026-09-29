# -*- coding: utf-8 -*-
"""
artificial_txl_hardware_core.py

Step 1 core integration for the artificial-symbol TXL experiment.

Purpose
-------
This module reuses the artificial 5x5 shape dataset framework from the older
test script, but replaces the simplified high-level TXL classifier with a
hardware-grounded model based on txl_model_v3.py.

Included in this step
---------------------
1. Artificial dataset generation:
   - Cross
   - Circle
   - Triangle
   - Square as OOD/new class

2. Feature/voltage conversion utilities.

3. Prototype generation:
   - Class mean
   - Class uncertainty
   - Voltage-domain prototype statistics
   - RRAM state encoding through txl_encode / TXLCell

4. Hardware TXL classifier wrapper:
   - One TXLArray per class
   - One TXLCell per feature dimension
   - Matchline current
   - Normalized match score
   - Hardware/statistical winner-take-all options
   - Reliable / unreliable / OOD zone decision

Not included in this step
-------------------------
- Plotting functions
- Full experiment execution script
- Final benchmark metrics/report generation
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Optional PyTorch support, to preserve compatibility with the old script
# ---------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object


# ---------------------------------------------------------------------
# Hardware-grounded TXL model (v4)
# ---------------------------------------------------------------------
# HW, TXLCell, TXLArray, and the core equations (txl_encode, txl_decode,
# txl_window_v4, txl_I_ML, txl_N_hat, txl_d2_tilde) are defined in the
# core-model cell above this one in the notebook and are already in scope
# here -- no import needed within a single notebook kernel.


# =====================================================================
# 1. GLOBAL ARTIFICIAL-DATA CONFIGURATION
# =====================================================================

FEATURE_MAP_SIDE: int = 5
FEATURE_MAP_DIM: int = FEATURE_MAP_SIDE * FEATURE_MAP_SIDE

# Keep the older test-script voltage range for continuity.
V_MIN: float = 0.1
V_MAX: float = 0.8

# Default confidence values matching the older artificial test.
DEFAULT_IDO_CONFIDENCE: float = 0.95
DEFAULT_OOD_CONFIDENCE: float = 0.999

# Default few-shot/adaptation parameters.
DEFAULT_N_SHOTS: int = 5
DEFAULT_ADAPT_RATIO: float = 0.5

# Known classes used in the original artificial test.
KNOWN_CLASSES: List[int] = [0, 1, 2]

# Use "Circle" rather than the old typo "Cycle".
CLASS_NAMES: Dict[int, str] = {
    0: "Cross",
    1: "Circle",
    2: "Triangle",
    3: "Square",
}

# Shape templates.
SHAPE_TEMPLATES: Dict[int, np.ndarray] = {
    0: np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    ),
    1: np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    ),
    2: np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    3: np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    ),
}


# =====================================================================
# 2. GENERAL UTILITY FUNCTIONS
# =====================================================================

def set_global_seed(seed: int = 42) -> None:
    """
    Set NumPy, Python, and optionally PyTorch random seeds.

    Parameters
    ----------
    seed:
        Reproducibility seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def as_numpy_vector(x: Any, dtype=np.float32) -> np.ndarray:
    """
    Convert a tensor/list/array-like object into a 1D NumPy vector.
    """
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x, dtype=dtype).reshape(-1)
    return arr


def validate_feature_vector(
    x: Any,
    dim: int = FEATURE_MAP_DIM,
    dtype=np.float32,
) -> np.ndarray:
    """
    Validate and return one feature vector.
    """
    arr = as_numpy_vector(x, dtype=dtype)

    if arr.shape[0] != dim:
        raise ValueError(f"Expected feature vector of length {dim}, got {arr.shape[0]}.")

    return arr


def validate_feature_matrix(
    X: Any,
    dim: int = FEATURE_MAP_DIM,
    dtype=np.float32,
) -> np.ndarray:
    """
    Validate and return a 2D feature matrix of shape [N, D].

    Accepts:
    - NumPy arrays
    - Python lists
    - lists of PyTorch tensors
    - lists of NumPy vectors
    """
    if torch is not None and isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    if isinstance(X, list) or isinstance(X, tuple):
        X = np.stack([as_numpy_vector(v, dtype=dtype) for v in X], axis=0)
    else:
        X = np.asarray(X, dtype=dtype)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}.")

    if X.shape[1] != dim:
        raise ValueError(f"Expected feature dimension {dim}, got {X.shape[1]}.")

    return X.astype(dtype, copy=False)


def labels_to_numpy(y: Any) -> np.ndarray:
    """
    Convert labels into a 1D NumPy integer array.
    """
    if torch is not None and isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if isinstance(y, list) or isinstance(y, tuple):
        vals = []
        for item in y:
            if torch is not None and isinstance(item, torch.Tensor):
                item = item.detach().cpu().item()
            vals.append(int(item))
        return np.asarray(vals, dtype=int)

    return np.asarray(y, dtype=int).reshape(-1)


# =====================================================================
# 3. FEATURE <-> VOLTAGE CONVERSION
# =====================================================================

def feature_to_voltage(
    x: Any,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    clip: bool = True,
) -> np.ndarray:
    """
    Map artificial feature values from [0, 1] to voltage range [v_min, v_max].

    This preserves the mapping used in the original high-level test script.
    """
    arr = np.asarray(x, dtype=np.float64)

    if clip:
        arr = np.clip(arr, 0.0, 1.0)

    return v_min + arr * (v_max - v_min)


def voltage_to_feature(
    v: Any,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    clip: bool = True,
) -> np.ndarray:
    """
    Inverse mapping from voltage values to normalized artificial features.
    """
    arr = np.asarray(v, dtype=np.float64)
    x = (arr - v_min) / (v_max - v_min)

    if clip:
        x = np.clip(x, 0.0, 1.0)

    return x


def feature_sigma_to_voltage_sigma(
    sigma_x: Any,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
) -> np.ndarray:
    """
    Convert a feature-domain standard deviation to voltage-domain sigma.
    """
    return np.asarray(sigma_x, dtype=np.float64) * (v_max - v_min)


def voltage_sigma_to_feature_sigma(
    sigma_v: Any,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
) -> np.ndarray:
    """
    Convert a voltage-domain sigma to feature-domain standard deviation.
    """
    return np.asarray(sigma_v, dtype=np.float64) / (v_max - v_min)


# =====================================================================
# 4. ARTIFICIAL DATASET GENERATION
# =====================================================================

def get_shape_templates() -> Dict[int, np.ndarray]:
    """
    Return a copy of all shape templates.
    """
    return {k: v.copy() for k, v in SHAPE_TEMPLATES.items()}


def generate_noisy_image(
    template: np.ndarray,
    noise_level_value: float = 0.08,
    pixels_affected_by_noise_count: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate one noisy 5x5 image from a binary template.

    This preserves the old behavior:
    1. Add Gaussian noise to all pixels.
    2. Randomly choose a number of pixels and flip them toward the opposite
       binary value, again with Gaussian perturbation.
    3. Clip final image to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    template = np.asarray(template, dtype=np.float32)
    noisy_image = template.copy()

    rows, cols = noisy_image.shape
    n_pixels = rows * cols

    noisy_image += rng.normal(
        loc=0.0,
        scale=noise_level_value,
        size=(rows, cols),
    ).astype(np.float32)

    n_flip = int(min(max(pixels_affected_by_noise_count, 0), n_pixels))

    if n_flip > 0:
        idx = rng.choice(np.arange(n_pixels), size=n_flip, replace=False)

        for flat_idx in idx:
            r = flat_idx // cols
            c = flat_idx % cols
            noisy_image[r, c] = (
                1.0 - template[r, c]
            ) + rng.normal(loc=0.0, scale=noise_level_value)

    return np.clip(noisy_image, 0.0, 1.0).astype(np.float32)


class ShapeDataset(Dataset):
    """
    PyTorch-compatible artificial 5x5 shape dataset.

    This class is retained for compatibility with the old script.

    Parameters
    ----------
    num_samples:
        Number of generated samples per class.

    noise:
        Gaussian noise level.

    flips:
        Number of pixels to flip per sample.

    class_labels:
        Classes to include. Default is [0, 1, 2].

    templates:
        Optional custom template dictionary.

    seed:
        Optional dataset seed.

    return_tensors:
        If True and PyTorch is available, returns torch.Tensor features.
        Otherwise returns NumPy vectors.
    """

    def __init__(
        self,
        num_samples: int = 100,
        noise: float = 0.08,
        flips: int = 2,
        class_labels: Sequence[int] = tuple(KNOWN_CLASSES),
        templates: Optional[Dict[int, np.ndarray]] = None,
        seed: Optional[int] = None,
        return_tensors: bool = True,
    ):
        self.num_samples = int(num_samples)
        self.noise = float(noise)
        self.flips = int(flips)
        self.class_labels = list(class_labels)
        self.templates = get_shape_templates() if templates is None else {
            int(k): np.asarray(v, dtype=np.float32).copy()
            for k, v in templates.items()
        }
        self.seed = seed
        self.return_tensors = bool(return_tensors and torch is not None)

        self.data: List[Any] = []
        self.labels: List[int] = []

        rng = np.random.default_rng(seed)

        for cls in self.class_labels:
            if cls not in self.templates:
                raise KeyError(f"No template found for class {cls}.")

            template = self.templates[cls]

            for _ in range(self.num_samples):
                img = generate_noisy_image(
                    template=template,
                    noise_level_value=self.noise,
                    pixels_affected_by_noise_count=self.flips,
                    rng=rng,
                )
                feat = img.flatten().astype(np.float32)

                if self.return_tensors:
                    feat = torch.from_numpy(feat)

                self.data.append(feat)
                self.labels.append(int(cls))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self.data[idx], self.labels[idx]


def dataset_to_numpy(dataset: ShapeDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ShapeDataset into NumPy arrays X, y.
    """
    X = np.stack([as_numpy_vector(v) for v in dataset.data], axis=0)
    y = np.asarray(dataset.labels, dtype=int)
    return X, y


def create_artificial_dataset_arrays(
    num_samples_per_class: int = 100,
    noise: float = 0.08,
    flips: int = 2,
    class_labels: Sequence[int] = tuple(KNOWN_CLASSES),
    seed: Optional[int] = None,
    templates: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create artificial shape data directly as NumPy arrays.

    Returns
    -------
    X:
        Shape [N, 25], feature values in [0, 1].

    y:
        Shape [N], integer class labels.
    """
    ds = ShapeDataset(
        num_samples=num_samples_per_class,
        noise=noise,
        flips=flips,
        class_labels=class_labels,
        templates=templates,
        seed=seed,
        return_tensors=False,
    )
    return dataset_to_numpy(ds)


def create_known_train_test_split(
    train_samples_per_class: int = 100,
    test_samples_per_class: int = 100,
    noise: float = 0.08,
    flips: int = 2,
    train_seed: int = 41,
    test_seed: int = 42,
    class_labels: Sequence[int] = tuple(KNOWN_CLASSES),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create independent known-class train/test arrays.

    This is preferred for benchmark-style evaluation because prototypes are
    trained on samples separate from those used for testing.
    """
    X_train, y_train = create_artificial_dataset_arrays(
        num_samples_per_class=train_samples_per_class,
        noise=noise,
        flips=flips,
        class_labels=class_labels,
        seed=train_seed,
    )

    X_test, y_test = create_artificial_dataset_arrays(
        num_samples_per_class=test_samples_per_class,
        noise=noise,
        flips=flips,
        class_labels=class_labels,
        seed=test_seed,
    )

    return X_train, y_train, X_test, y_test


def generate_ood_samples(
    num_samples: int = 100,
    noise: float = 0.08,
    flips: int = 2,
    start_label: int = 3,
    seed: Optional[int] = None,
    return_tensors: bool = True,
    template: Optional[np.ndarray] = None,
) -> Tuple[List[Any], List[int]]:
    """
    Generate OOD/new-class samples.

    Default OOD class is Square with label 3.

    This preserves the old script's pattern of returning lists of data and
    labels, while supporting deterministic seeds.
    """
    if template is None:
        template = SHAPE_TEMPLATES[start_label]

    rng = np.random.default_rng(seed)

    data: List[Any] = []
    labels: List[int] = []

    use_tensors = bool(return_tensors and torch is not None)

    for _ in range(int(num_samples)):
        img = generate_noisy_image(
            template=template,
            noise_level_value=noise,
            pixels_affected_by_noise_count=flips,
            rng=rng,
        )

        feat = img.flatten().astype(np.float32)

        if use_tensors:
            feat = torch.from_numpy(feat)

        data.append(feat)
        labels.append(int(start_label))

    return data, labels


def create_ood_dataset_arrays(
    num_samples: int = 100,
    noise: float = 0.08,
    flips: int = 2,
    label: int = 3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create OOD/new-class samples directly as NumPy arrays.
    """
    data, labels = generate_ood_samples(
        num_samples=num_samples,
        noise=noise,
        flips=flips,
        start_label=label,
        seed=seed,
        return_tensors=False,
    )

    X = np.stack([as_numpy_vector(v) for v in data], axis=0)
    y = np.asarray(labels, dtype=int)

    return X, y


def stratified_split_arrays(
    X: Any,
    y: Any,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split for already-generated arrays.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    X = validate_feature_matrix(X)
    y = labels_to_numpy(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have inconsistent sample counts.")

    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    test_idx: List[int] = []

    for cls in sorted(np.unique(y)):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        n_train = int(round(len(cls_idx) * train_fraction))

        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# =====================================================================
# 5. HARDWARE CONFIG FOR ARTIFICIAL TEST
# =====================================================================

def make_artificial_hw_config(
    p_ido: float = DEFAULT_IDO_CONFIDENCE,
    p_ood: float = DEFAULT_OOD_CONFIDENCE,
    base_hw: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a TXL hardware parameter dictionary for the artificial test.

    The returned dictionary starts from txl_model_v3.HW and updates the
    reliable and OOD confidence levels to match the old artificial script.
    """
    hw = deepcopy(HW if base_hw is None else base_hw)

    hw["P_IDO"] = float(p_ido)
    hw["P_OOD"] = float(p_ood)

    if overrides is not None:
        hw.update(overrides)

    return hw


# =====================================================================
# 6. PROTOTYPE REPRESENTATION
# =====================================================================

@dataclass
class PrototypeStats:
    """
    Trained class prototype and its corresponding hardware encoding.

    The training pipeline computes statistics in feature space, maps them
    to voltage space, clips uncertainty windows, and then encodes the result
    into RRAM states.
    """

    label: int
    class_name: str
    n_samples: int
    dim: int

    # Feature-domain statistics.
    mu_x: np.ndarray
    sigma_x_raw: np.ndarray
    sigma_x_effective: np.ndarray

    # Voltage-domain intended prototype.
    mu_v: np.ndarray
    sigma_v_raw: np.ndarray
    sigma_v: np.ndarray

    # Sigma clipping diagnostics.
    sigma_clipped_low: np.ndarray
    sigma_clipped_high: np.ndarray

    # Encoded hardware state.
    R_M1: np.ndarray
    R_M2: np.ndarray

    # Decoded hardware state.
    mu_v_decoded: np.ndarray
    sigma_v_decoded: np.ndarray

    # Resistance clipping diagnostics.
    resistance_clipped_m1: np.ndarray
    resistance_clipped_m2: np.ndarray

    # Configuration metadata.
    v_min: float
    v_max: float
    sigma_v_min: float
    sigma_v_max: float

    def as_summary_dict(self) -> Dict[str, Any]:
        """
        Return scalar summary values suitable for a DataFrame row.
        """
        mu_err = np.abs(self.mu_v_decoded - self.mu_v)
        sig_err = np.abs(self.sigma_v_decoded - self.sigma_v)

        return {
            "label": self.label,
            "class_name": self.class_name,
            "n_samples": self.n_samples,
            "dim": self.dim,
            "feature_mu_mean": float(np.mean(self.mu_x)),
            "feature_sigma_raw_mean": float(np.mean(self.sigma_x_raw)),
            "feature_sigma_effective_mean": float(np.mean(self.sigma_x_effective)),
            "voltage_mu_mean": float(np.mean(self.mu_v)),
            "voltage_sigma_raw_mean": float(np.mean(self.sigma_v_raw)),
            "voltage_sigma_effective_mean": float(np.mean(self.sigma_v)),
            "voltage_sigma_min": float(np.min(self.sigma_v)),
            "voltage_sigma_max": float(np.max(self.sigma_v)),
            "sigma_clip_low_count": int(np.sum(self.sigma_clipped_low)),
            "sigma_clip_high_count": int(np.sum(self.sigma_clipped_high)),
            "R_M1_mean_ohm": float(np.mean(self.R_M1)),
            "R_M2_mean_ohm": float(np.mean(self.R_M2)),
            "R_M1_min_ohm": float(np.min(self.R_M1)),
            "R_M1_max_ohm": float(np.max(self.R_M1)),
            "R_M2_min_ohm": float(np.min(self.R_M2)),
            "R_M2_max_ohm": float(np.max(self.R_M2)),
            "R_gap_mean_ohm": float(np.mean(self.R_M1 - self.R_M2)),
            "R_M1_clipped_count": int(np.sum(self.resistance_clipped_m1)),
            "R_M2_clipped_count": int(np.sum(self.resistance_clipped_m2)),
            "decode_mu_mae_v": float(np.mean(mu_err)),
            "decode_mu_maxerr_v": float(np.max(mu_err)),
            "decode_sigma_mae_v": float(np.mean(sig_err)),
            "decode_sigma_maxerr_v": float(np.max(sig_err)),
        }

    def as_cells_dataframe(self) -> pd.DataFrame:
        """
        Return one row per feature/cell for this prototype.
        """
        rows = []

        for i in range(self.dim):
            rows.append(
                {
                    "label": self.label,
                    "class_name": self.class_name,
                    "feature_index": i,
                    "mu_x": float(self.mu_x[i]),
                    "sigma_x_raw": float(self.sigma_x_raw[i]),
                    "sigma_x_effective": float(self.sigma_x_effective[i]),
                    "mu_v": float(self.mu_v[i]),
                    "sigma_v_raw": float(self.sigma_v_raw[i]),
                    "sigma_v": float(self.sigma_v[i]),
                    "sigma_clipped_low": bool(self.sigma_clipped_low[i]),
                    "sigma_clipped_high": bool(self.sigma_clipped_high[i]),
                    "R_M1_ohm": float(self.R_M1[i]),
                    "R_M2_ohm": float(self.R_M2[i]),
                    "R_gap_ohm": float(self.R_M1[i] - self.R_M2[i]),
                    "mu_v_decoded": float(self.mu_v_decoded[i]),
                    "sigma_v_decoded": float(self.sigma_v_decoded[i]),
                    "decode_mu_abs_err_v": float(abs(self.mu_v_decoded[i] - self.mu_v[i])),
                    "decode_sigma_abs_err_v": float(abs(self.sigma_v_decoded[i] - self.sigma_v[i])),
                    "R_M1_clipped": bool(self.resistance_clipped_m1[i]),
                    "R_M2_clipped": bool(self.resistance_clipped_m2[i]),
                }
            )

        return pd.DataFrame(rows)


def build_prototype_stats_from_moments(
    mu_x: np.ndarray,
    sigma_x_raw: np.ndarray,
    label: int,
    n_samples: int,
    class_name: Optional[str] = None,
    hw: Optional[Dict[str, Any]] = None,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    sigma_v_min: float = 0.02,
    sigma_v_max: float = 0.20,
) -> PrototypeStats:
    """
    Build a hardware-encoded TXL prototype from feature-domain moments.

    This function performs the central training-to-hardware operation:
    feature statistics -> voltage-domain statistics -> RRAM states.
    """
    if hw is None:
        hw = make_artificial_hw_config()

    mu_x = np.asarray(mu_x, dtype=np.float64).reshape(-1)
    sigma_x_raw = np.asarray(sigma_x_raw, dtype=np.float64).reshape(-1)

    if mu_x.shape != sigma_x_raw.shape:
        raise ValueError("mu_x and sigma_x_raw must have matching shapes.")

    dim = mu_x.shape[0]

    if class_name is None:
        class_name = CLASS_NAMES.get(int(label), f"Class {label}")

    # Feature mean -> voltage mean.
    mu_v = feature_to_voltage(mu_x, v_min=v_min, v_max=v_max, clip=True)

    # Feature sigma -> voltage sigma.
    sigma_v_raw = feature_sigma_to_voltage_sigma(
        sigma_x_raw,
        v_min=v_min,
        v_max=v_max,
    )

    sigma_clipped_low = sigma_v_raw < sigma_v_min
    sigma_clipped_high = sigma_v_raw > sigma_v_max

    sigma_v = np.clip(sigma_v_raw, sigma_v_min, sigma_v_max)

    # Effective feature sigma after voltage-domain clipping.
    sigma_x_effective = voltage_sigma_to_feature_sigma(
        sigma_v,
        v_min=v_min,
        v_max=v_max,
    )

    # Encode intended voltage-domain prototype into RRAM states.
    R_M1, R_M2 = txl_encode(mu_v, sigma_v, hw=hw)

    R_min = float(hw["R_min"])
    R_max = float(hw["R_max"])

    resistance_clipped_m1 = np.isclose(R_M1, R_min) | np.isclose(R_M1, R_max)
    resistance_clipped_m2 = np.isclose(R_M2, R_min) | np.isclose(R_M2, R_max)

    # Decode back to inspect how faithfully hardware stores the prototype.
    mu_v_decoded, sigma_v_decoded = txl_decode(R_M1, R_M2, hw=hw)

    return PrototypeStats(
        label=int(label),
        class_name=str(class_name),
        n_samples=int(n_samples),
        dim=int(dim),
        mu_x=mu_x,
        sigma_x_raw=sigma_x_raw,
        sigma_x_effective=sigma_x_effective,
        mu_v=mu_v,
        sigma_v_raw=sigma_v_raw,
        sigma_v=sigma_v,
        sigma_clipped_low=sigma_clipped_low,
        sigma_clipped_high=sigma_clipped_high,
        R_M1=np.asarray(R_M1, dtype=np.float64),
        R_M2=np.asarray(R_M2, dtype=np.float64),
        mu_v_decoded=np.asarray(mu_v_decoded, dtype=np.float64),
        sigma_v_decoded=np.asarray(sigma_v_decoded, dtype=np.float64),
        resistance_clipped_m1=resistance_clipped_m1,
        resistance_clipped_m2=resistance_clipped_m2,
        v_min=float(v_min),
        v_max=float(v_max),
        sigma_v_min=float(sigma_v_min),
        sigma_v_max=float(sigma_v_max),
    )


def build_prototype_stats_from_samples(
    X_class: Any,
    label: int,
    class_name: Optional[str] = None,
    hw: Optional[Dict[str, Any]] = None,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    sigma_v_min: float = 0.02,
    sigma_v_max: float = 0.20,
    dim: int = FEATURE_MAP_DIM,
) -> PrototypeStats:
    """
    Compute a class prototype from training samples and encode it into
    hardware-compatible TXL state.
    """
    X_class = validate_feature_matrix(X_class, dim=dim, dtype=np.float64)

    n_samples = X_class.shape[0]

    if n_samples < 1:
        raise ValueError("Cannot build prototype from zero samples.")

    mu_x = np.mean(X_class, axis=0)

    # Use sample standard deviation when possible.
    ddof = 1 if n_samples > 1 else 0
    sigma_x_raw = np.std(X_class, axis=0, ddof=ddof)

    return build_prototype_stats_from_moments(
        mu_x=mu_x,
        sigma_x_raw=sigma_x_raw,
        label=label,
        n_samples=n_samples,
        class_name=class_name,
        hw=hw,
        v_min=v_min,
        v_max=v_max,
        sigma_v_min=sigma_v_min,
        sigma_v_max=sigma_v_max,
    )


# =====================================================================
# 7. HARDWARE-GROUNDED TXL CLASSIFIER
# =====================================================================

class HardwareTXLClassifier:
    """
    Hardware-grounded TXL classifier for the artificial 5x5 dataset.

    Main behavior
    -------------
    - Each class is stored as one TXLArray.
    - Each TXLArray contains D TXLCell objects.
    - Each TXLCell stores feature statistics as RRAM states R_M1/R_M2.
    - Inference applies a voltage vector to every class row.
    - Winner can be selected by:
        1. max_nhat: hardware-style highest normalized matchline score.
        2. min_d2: statistical-style lowest diagonal Mahalanobis distance.

    Compatibility
    -------------
    The method inference(x) returns:

        pred_label, best_d2, status

    similar to the old preliminary TXL_Classifier.

    For full hardware diagnostics, use:

        inference_full(x)
    """

    def __init__(
        self,
        dim: int = FEATURE_MAP_DIM,
        hw: Optional[Dict[str, Any]] = None,
        v_min: float = V_MIN,
        v_max: float = V_MAX,
        sigma_v_min: float = 0.02,
        sigma_v_max: float = 0.20,
        winner_rule: str = "max_nhat",
        class_names: Optional[Dict[int, str]] = None,
        k_edge_lo: float = 2.0,
        k_edge_hi: float = 2.0,
        w_scale: float = 1.0,
        center_shift: float = 0.0,
        calibrate_thresholds: bool = True,
    ):
        self.dim = int(dim)
        self.hw = make_artificial_hw_config() if hw is None else deepcopy(hw)

        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.sigma_v_min = float(sigma_v_min)
        self.sigma_v_max = float(sigma_v_max)

        # v4 double-sigmoid response-shape parameters (SEC 2 of the core
        # model). Uniform defaults across all cells/rows; override here to
        # sweep steepness/width/placement for this dataset.
        self.k_edge_lo = float(k_edge_lo)
        self.k_edge_hi = float(k_edge_hi)
        self.w_scale = float(w_scale)
        self.center_shift = float(center_shift)

        # If True, tau_IDO/tau_OOD are set by empirical calibration (SEC 5 of
        # the core model) rather than the theoretical chi2 reference -- see
        # the migration notes markdown cell for why this matters.
        self.calibrate_thresholds_flag = bool(calibrate_thresholds)

        if winner_rule not in {"max_nhat", "min_d2"}:
            raise ValueError("winner_rule must be either 'max_nhat' or 'min_d2'.")

        self.winner_rule = winner_rule

        self.class_names: Dict[int, str] = dict(CLASS_NAMES)
        if class_names is not None:
            self.class_names.update({int(k): str(v) for k, v in class_names.items()})

        # Main learned hardware state.
        self.prototype_stats_: Dict[int, PrototypeStats] = {}
        self.arrays_: Dict[int, TXLArray] = {}

        # Compatibility dictionary similar to the older script.
        # Stores feature-domain mean/variance plus voltage-domain state.
        self.prototypes: Dict[int, Dict[str, np.ndarray]] = {}

        # Training/adaptation event log.
        self.training_history_: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> List[int]:
        return sorted(self.arrays_.keys())

    @property
    def n_classes_(self) -> int:
        return len(self.arrays_)

    @property
    def n_cells_(self) -> int:
        return self.n_classes_ * self.dim

    def is_fitted(self) -> bool:
        return len(self.arrays_) > 0

    # ------------------------------------------------------------------
    # Training / prototype encoding
    # ------------------------------------------------------------------

    def _stats_to_array(self, stats: PrototypeStats) -> TXLArray:
        """
        Convert PrototypeStats into a TXLArray containing one TXLCell per
        feature.
        """
        cells: List[TXLCell] = []

        for i in range(stats.dim):
            cell = TXLCell.from_params(
                mu=float(stats.mu_v[i]),
                sigma=float(stats.sigma_v[i]),
                hw=self.hw,
                k_edge_lo=self.k_edge_lo,
                k_edge_hi=self.k_edge_hi,
                w_scale=self.w_scale,
                center_shift=self.center_shift,
                name=f"TXL_cls{stats.label}_f{i}",
                sweep_param="ARTIFICIAL",
                sweep_val=i,
                source="trained_prototype",
            )
            cells.append(cell)

        return TXLArray(cells, name=f"Class_{stats.label}_{stats.class_name}")

    def _update_compatibility_prototype(self, label: int) -> None:
        """
        Update old-script-style prototype dictionary.
        """
        stats = self.prototype_stats_[label]

        self.prototypes[label] = {
            "mu": stats.mu_x.copy(),
            "var": np.maximum(stats.sigma_x_effective ** 2, 1e-12),
            "sigma": stats.sigma_x_effective.copy(),
            "mu_v": stats.mu_v_decoded.copy(),
            "sigma_v": stats.sigma_v_decoded.copy(),
            "R_M1": stats.R_M1.copy(),
            "R_M2": stats.R_M2.copy(),
        }

    def fit_class(
        self,
        label: int,
        X_class: Any,
        class_name: Optional[str] = None,
        replace: bool = True,
        event_name: str = "fit_class",
    ) -> "HardwareTXLClassifier":
        """
        Fit or replace one class row in the TXL hardware classifier.

        Parameters
        ----------
        label:
            Class label for the row.

        X_class:
            Training samples for this class, shape [N, D].

        class_name:
            Optional human-readable class name.

        replace:
            If False, raises an error when the class already exists.
        """
        label = int(label)

        if label in self.arrays_ and not replace:
            raise ValueError(f"Class {label} already exists and replace=False.")

        if class_name is None:
            class_name = self.class_names.get(label, f"Class {label}")

        stats = build_prototype_stats_from_samples(
            X_class=X_class,
            label=label,
            class_name=class_name,
            hw=self.hw,
            v_min=self.v_min,
            v_max=self.v_max,
            sigma_v_min=self.sigma_v_min,
            sigma_v_max=self.sigma_v_max,
            dim=self.dim,
        )

        array = self._stats_to_array(stats)

        if self.calibrate_thresholds_flag:
            # Empirical calibration (TXLArray.calibrate_thresholds, core
            # model SEC 5): tau_IDO/tau_OOD are set from percentiles of this
            # row's own d2_tilde distribution over the samples it was just
            # fit on, rather than a theoretical chi2 reference. See the
            # migration notes markdown for why this matters in practice.
            X_v_calib = feature_to_voltage(
                validate_feature_matrix(X_class, dim=self.dim, dtype=np.float64),
                v_min=self.v_min, v_max=self.v_max, clip=True,
            )
            array.calibrate_thresholds(X_v_calib)

        self.prototype_stats_[label] = stats
        self.arrays_[label] = array
        self.class_names[label] = class_name
        self._update_compatibility_prototype(label)

        self.training_history_.append(
            {
                "event": event_name,
                "label": label,
                "class_name": class_name,
                "n_samples": stats.n_samples,
                "n_classes": self.n_classes_,
                "n_cells_total": self.n_cells_,
                "sigma_clip_low_count": int(np.sum(stats.sigma_clipped_low)),
                "sigma_clip_high_count": int(np.sum(stats.sigma_clipped_high)),
                "R_M1_clipped_count": int(np.sum(stats.resistance_clipped_m1)),
                "R_M2_clipped_count": int(np.sum(stats.resistance_clipped_m2)),
            }
        )

        return self

    def fit(
        self,
        X: Any,
        y: Any,
        class_labels: Optional[Sequence[int]] = None,
    ) -> "HardwareTXLClassifier":
        """
        Fit TXL hardware rows for all requested classes.
        """
        X = validate_feature_matrix(X, dim=self.dim, dtype=np.float64)
        y = labels_to_numpy(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent sample counts.")

        if class_labels is None:
            class_labels = sorted(np.unique(y).tolist())

        for label in class_labels:
            label = int(label)
            X_class = X[y == label]

            if len(X_class) == 0:
                raise ValueError(f"No training samples found for class {label}.")

            self.fit_class(
                label=label,
                X_class=X_class,
                class_name=self.class_names.get(label, f"Class {label}"),
                replace=True,
                event_name="fit",
            )

        return self

    def allocate_new_class(
        self,
        label: int,
        X_support: Any,
        class_name: Optional[str] = None,
        allow_overwrite: bool = False,
    ) -> "HardwareTXLClassifier":
        """
        Allocate a new TXL row from few-shot support samples.

        This is used for OOD/new-class learning, e.g. Square allocation.

        Raises a ValueError if `label` already names an existing TXL row,
        unless `allow_overwrite=True` is passed explicitly. This prevents an
        OOD/new-class allocation call from silently overwriting an existing
        class prototype via fit_class(..., replace=True).
        """
        label = int(label)

        if (label in self.arrays_) and (not allow_overwrite):
            raise ValueError(
                f"allocate_new_class: label {label} already exists as a TXL row "
                f"({self.class_names.get(label, f'Class {label}')}). "
                "Pass allow_overwrite=True to intentionally replace it, "
                "or use adapt_class(...) to update an existing prototype instead."
            )

        return self.fit_class(
            label=label,
            X_class=X_support,
            class_name=class_name or self.class_names.get(label, f"Class {label}"),
            replace=True,
            event_name="allocate_new_class",
        )

    def adapt_class(
        self,
        label: int,
        X_support: Any,
        adapt_ratio: float = DEFAULT_ADAPT_RATIO,
        mode: str = "pooled_ema",
    ) -> "HardwareTXLClassifier":
        """
        Adapt an existing class prototype using few-shot support samples.

        Parameters
        ----------
        label:
            Existing class to update.

        X_support:
            Few-shot samples.

        adapt_ratio:
            Interpolation strength between old prototype and support moments.

        mode:
            "pooled_ema":
                Blend means and approximate variance with an EMA-style pooled
                variance term.

            "sigma_ema":
                Blend means and standard deviations directly.

            "replace":
                Replace the existing prototype using only support samples.
        """
        label = int(label)

        if label not in self.prototype_stats_:
            raise KeyError(f"Cannot adapt missing class {label}.")

        X_support = validate_feature_matrix(X_support, dim=self.dim, dtype=np.float64)

        if len(X_support) == 0:
            raise ValueError("Cannot adapt from zero support samples.")

        adapt_ratio = float(adapt_ratio)
        adapt_ratio = float(np.clip(adapt_ratio, 0.0, 1.0))

        if mode == "replace":
            return self.fit_class(
                label=label,
                X_class=X_support,
                class_name=self.class_names.get(label, f"Class {label}"),
                replace=True,
                event_name="adapt_replace",
            )

        old = self.prototype_stats_[label]

        support_mu = np.mean(X_support, axis=0)
        ddof = 1 if len(X_support) > 1 else 0
        support_sigma = np.std(X_support, axis=0, ddof=ddof)

        new_mu = (1.0 - adapt_ratio) * old.mu_x + adapt_ratio * support_mu

        if mode == "pooled_ema":
            # Approximate variance blend with between-mean correction.
            old_var = old.sigma_x_raw ** 2
            support_var = support_sigma ** 2

            between_mean = (old.mu_x - support_mu) ** 2

            new_var = (
                (1.0 - adapt_ratio) * old_var
                + adapt_ratio * support_var
                + adapt_ratio * (1.0 - adapt_ratio) * between_mean
            )

            new_sigma = np.sqrt(np.maximum(new_var, 0.0))

        elif mode == "sigma_ema":
            new_sigma = (
                (1.0 - adapt_ratio) * old.sigma_x_raw
                + adapt_ratio * support_sigma
            )

        else:
            raise ValueError("mode must be 'pooled_ema', 'sigma_ema', or 'replace'.")

        new_stats = build_prototype_stats_from_moments(
            mu_x=new_mu,
            sigma_x_raw=new_sigma,
            label=label,
            n_samples=old.n_samples + len(X_support),
            class_name=old.class_name,
            hw=self.hw,
            v_min=self.v_min,
            v_max=self.v_max,
            sigma_v_min=self.sigma_v_min,
            sigma_v_max=self.sigma_v_max,
        )

        pre_R_M1 = old.R_M1.copy()
        pre_R_M2 = old.R_M2.copy()

        # Preserve this row's calibrated tau_IDO/tau_OOD across the rebuild:
        # adaptation is a small nudge to an already-calibrated row, not a
        # fresh fit, so we carry the existing empirical thresholds forward
        # rather than resetting to the theoretical chi2 default.
        old_tau_IDO = self.arrays_[label].tau_IDO
        old_tau_OOD = self.arrays_[label].tau_OOD

        self.prototype_stats_[label] = new_stats
        new_array = self._stats_to_array(new_stats)
        if self.calibrate_thresholds_flag:
            new_array.tau_IDO, new_array.tau_OOD = old_tau_IDO, old_tau_OOD
        self.arrays_[label] = new_array
        self._update_compatibility_prototype(label)

        self.training_history_.append(
            {
                "event": "adapt_class",
                "label": label,
                "class_name": old.class_name,
                "mode": mode,
                "adapt_ratio": adapt_ratio,
                "n_support": len(X_support),
                "mean_abs_mu_shift_feature": float(np.mean(np.abs(new_stats.mu_x - old.mu_x))),
                "mean_abs_sigma_shift_feature": float(
                    np.mean(np.abs(new_stats.sigma_x_raw - old.sigma_x_raw))
                ),
                "mean_abs_R_M1_shift_ohm": float(np.mean(np.abs(new_stats.R_M1 - pre_R_M1))),
                "mean_abs_R_M2_shift_ohm": float(np.mean(np.abs(new_stats.R_M2 - pre_R_M2))),
            }
        )

        return self

    # ------------------------------------------------------------------
    # Hardware row accessors
    # ------------------------------------------------------------------

    def get_txl_array(self, label: int) -> TXLArray:
        label = int(label)
        if label not in self.arrays_:
            raise KeyError(f"No TXLArray for class {label}.")
        return self.arrays_[label]

    def get_prototype_stats(self, label: int) -> PrototypeStats:
        label = int(label)
        if label not in self.prototype_stats_:
            raise KeyError(f"No PrototypeStats for class {label}.")
        return self.prototype_stats_[label]

    def get_class_cells(self, label: int) -> List[TXLCell]:
        return self.get_txl_array(label).cells

    def get_class_voltage_prototype(
        self,
        label: int,
        decoded: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return voltage-domain prototype mean and sigma for a class.

        Parameters
        ----------
        decoded:
            If True, return the hardware-decoded values from RRAM states.
            If False, return the intended training values before hardware
            encode/decode.
        """
        stats = self.get_prototype_stats(label)

        if decoded:
            return stats.mu_v_decoded.copy(), stats.sigma_v_decoded.copy()

        return stats.mu_v.copy(), stats.sigma_v.copy()

    def prototype_summary_dataframe(self) -> pd.DataFrame:
        """
        Return one summary row per trained class.
        """
        rows = [
            self.prototype_stats_[label].as_summary_dict()
            for label in self.classes_
        ]
        return pd.DataFrame(rows)

    def prototype_cells_dataframe(self) -> pd.DataFrame:
        """
        Return one row per class-feature TXL cell.
        """
        frames = [
            self.prototype_stats_[label].as_cells_dataframe()
            for label in self.classes_
        ]

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=0, ignore_index=True)

    def training_history_dataframe(self) -> pd.DataFrame:
        """
        Return a log of training/allocation/adaptation events.
        """
        return pd.DataFrame(self.training_history_)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def _legacy_status(zone: str) -> str:
        """
        Map txl_model_v3 zone naming to the old figure vocabulary.
        """
        if zone == "UNRELIABLE":
            return "IDO"
        return zone

    def _evaluate_row(
        self,
        label: int,
        array: TXLArray,
        x_v: np.ndarray,
        include_cell_responses: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate one class row for a voltage-domain query vector.
        """
        zone, n_hat, d2_val = array.zone(x_v)
        i_ml = array.I_ML_value(x_v)

        row = {
            "label": int(label),
            "class_name": self.class_names.get(int(label), f"Class {label}"),
            "zone": zone,
            "status_legacy": self._legacy_status(zone),
            "N_hat": float(n_hat),
            "I_ML": float(i_ml),
            "d2": float(d2_val),
            "tau_IDO": float(array.tau_IDO),
            "tau_OOD": float(array.tau_OOD),
        }

        if include_cell_responses:
            responses = array.cell_responses(x_v)
            row["cell_responses"] = np.asarray(responses, dtype=np.float64).reshape(-1)

        return row

    def inference_full(
        self,
        x_feat: Any,
        include_cell_responses: bool = False,
    ) -> Dict[str, Any]:
        """
        Full hardware-grounded inference diagnostics for one query.

        Returns a dictionary with:
        - predicted class
        - best row diagnostics
        - winner by max N_hat
        - winner by min d2
        - agreement flag
        - all per-class row diagnostics
        """
        if not self.is_fitted():
            raise RuntimeError("HardwareTXLClassifier has no trained TXL rows.")

        x_feat = validate_feature_vector(x_feat, dim=self.dim, dtype=np.float64)
        x_v = feature_to_voltage(
            x_feat,
            v_min=self.v_min,
            v_max=self.v_max,
            clip=True,
        )

        row_results: Dict[int, Dict[str, Any]] = {}

        for label in self.classes_:
            row_results[label] = self._evaluate_row(
                label=label,
                array=self.arrays_[label],
                x_v=x_v,
                include_cell_responses=include_cell_responses,
            )

        winner_by_nhat = max(
            row_results.keys(),
            key=lambda lbl: (row_results[lbl]["N_hat"], -row_results[lbl]["d2"]),
        )

        winner_by_d2 = min(
            row_results.keys(),
            key=lambda lbl: (row_results[lbl]["d2"], -row_results[lbl]["N_hat"]),
        )

        if self.winner_rule == "max_nhat":
            pred_label = winner_by_nhat
            sorted_labels = sorted(
                row_results.keys(),
                key=lambda lbl: row_results[lbl]["N_hat"],
                reverse=True,
            )

            if len(sorted_labels) > 1:
                margin = (
                    row_results[sorted_labels[0]]["N_hat"]
                    - row_results[sorted_labels[1]]["N_hat"]
                )
                second_label = sorted_labels[1]
            else:
                margin = np.nan
                second_label = None

        elif self.winner_rule == "min_d2":
            pred_label = winner_by_d2
            sorted_labels = sorted(
                row_results.keys(),
                key=lambda lbl: row_results[lbl]["d2"],
            )

            if len(sorted_labels) > 1:
                margin = (
                    row_results[sorted_labels[1]]["d2"]
                    - row_results[sorted_labels[0]]["d2"]
                )
                second_label = sorted_labels[1]
            else:
                margin = np.nan
                second_label = None

        else:  # pragma: no cover
            raise RuntimeError(f"Unknown winner_rule: {self.winner_rule}")

        best = row_results[pred_label]

        return {
            "pred_label": int(pred_label),
            "pred_class_name": self.class_names.get(int(pred_label), f"Class {pred_label}"),
            "winner_rule": self.winner_rule,
            "status": best["zone"],
            "status_legacy": best["status_legacy"],
            "best_label": int(pred_label),
            "best_class_name": best["class_name"],
            "best_d2": float(best["d2"]),
            "best_N_hat": float(best["N_hat"]),
            "best_I_ML": float(best["I_ML"]),
            "second_label": None if second_label is None else int(second_label),
            "winner_margin": float(margin) if not np.isnan(margin) else np.nan,
            "winner_by_nhat": int(winner_by_nhat),
            "winner_by_d2": int(winner_by_d2),
            "winner_agreement": bool(winner_by_nhat == winner_by_d2),
            "x_feat": x_feat.copy(),
            "x_voltage": x_v.copy(),
            "row_results": row_results,
        }

    def inference(
        self,
        x_feat: Any,
        gt_label: Optional[int] = None,
        adaptation_mode: str = "OFF",
        few_shot_samples: Optional[Any] = None,
    ) -> Tuple[int, float, str]:
        """
        Old-script-compatible inference interface.

        Returns
        -------
        pred_label:
            Predicted class.

        best_d2:
            Distance of the winning row.

        status:
            Legacy-compatible status:
            - RELIABLE
            - IDO
            - OOD

        Notes
        -----
        If adaptation_mode == "FULL" and few_shot_samples is provided, this
        method performs a replacement/update similar to the preliminary code.
        For more controlled adaptation, prefer adapt_class() or
        allocate_new_class().
        """
        adaptation_mode = str(adaptation_mode).upper()

        if adaptation_mode == "FULL":
            if gt_label is None:
                raise ValueError("gt_label is required when adaptation_mode='FULL'.")
            if few_shot_samples is None:
                raise ValueError("few_shot_samples is required when adaptation_mode='FULL'.")

            gt_label = int(gt_label)

            event_preexists = gt_label in self.arrays_

            self.fit_class(
                label=gt_label,
                X_class=few_shot_samples,
                class_name=self.class_names.get(gt_label, f"Class {gt_label}"),
                replace=True,
                event_name="inference_full_adaptation",
            )

            result = self.inference_full(x_feat)

            status = "CLASS UPDATED" if event_preexists else "NEW CLASS ALLOCATED"
            return int(gt_label), float(result["best_d2"]), status

        result = self.inference_full(x_feat)
        return (
            int(result["pred_label"]),
            float(result["best_d2"]),
            str(result["status_legacy"]),
        )

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict labels for a batch of samples.
        """
        X = validate_feature_matrix(X, dim=self.dim, dtype=np.float64)
        preds = [self.inference_full(x)["pred_label"] for x in X]
        return np.asarray(preds, dtype=int)

    def predict_full(
        self,
        X: Any,
        include_cell_responses: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return full inference dictionaries for a batch of samples.
        """
        X = validate_feature_matrix(X, dim=self.dim, dtype=np.float64)

        return [
            self.inference_full(
                x,
                include_cell_responses=include_cell_responses,
            )
            for x in X
        ]

    def inference_dataframe(
        self,
        X: Any,
        y: Optional[Any] = None,
        include_per_class: bool = True,
    ) -> pd.DataFrame:
        """
        Run inference and return a flat DataFrame.

        This is useful for later plotting and metric functions.
        """
        X = validate_feature_matrix(X, dim=self.dim, dtype=np.float64)

        if y is not None:
            y_arr = labels_to_numpy(y)
            if len(y_arr) != len(X):
                raise ValueError("X and y have inconsistent sample counts.")
        else:
            y_arr = np.full(len(X), fill_value=-1, dtype=int)

        rows: List[Dict[str, Any]] = []

        for idx, x in enumerate(X):
            result = self.inference_full(x)

            row = {
                "sample_index": idx,
                "true_label": int(y_arr[idx]),
                "true_class_name": self.class_names.get(int(y_arr[idx]), "UNKNOWN"),
                "pred_label": int(result["pred_label"]),
                "pred_class_name": result["pred_class_name"],
                "winner_rule": result["winner_rule"],
                "status": result["status"],
                "status_legacy": result["status_legacy"],
                "best_d2": float(result["best_d2"]),
                "best_N_hat": float(result["best_N_hat"]),
                "best_I_ML": float(result["best_I_ML"]),
                "second_label": result["second_label"],
                "winner_margin": float(result["winner_margin"]),
                "winner_by_nhat": int(result["winner_by_nhat"]),
                "winner_by_d2": int(result["winner_by_d2"]),
                "winner_agreement": bool(result["winner_agreement"]),
            }

            if include_per_class:
                for label, rr in result["row_results"].items():
                    prefix = f"class_{label}"
                    row[f"{prefix}_d2"] = float(rr["d2"])
                    row[f"{prefix}_N_hat"] = float(rr["N_hat"])
                    row[f"{prefix}_I_ML"] = float(rr["I_ML"])
                    row[f"{prefix}_zone"] = rr["zone"]
                    row[f"{prefix}_status_legacy"] = rr["status_legacy"]

            rows.append(row)

        return pd.DataFrame(rows)

    def decision_scores(
        self,
        X: Any,
        score_type: str = "best_N_hat",
    ) -> np.ndarray:
        """
        Return scalar confidence/OOD scores for a batch.

        Useful future options for OOD metrics:
        - best_N_hat: larger means more ID-like.
        - negative_best_d2: larger means more ID-like.
        - best_I_ML: larger means more ID-like.
        - best_d2: smaller means more ID-like.
        """
        X = validate_feature_matrix(X, dim=self.dim, dtype=np.float64)
        results = self.predict_full(X)

        if score_type == "best_N_hat":
            return np.asarray([r["best_N_hat"] for r in results], dtype=float)

        if score_type == "negative_best_d2":
            return np.asarray([-r["best_d2"] for r in results], dtype=float)

        if score_type == "best_I_ML":
            return np.asarray([r["best_I_ML"] for r in results], dtype=float)

        if score_type == "best_d2":
            return np.asarray([r["best_d2"] for r in results], dtype=float)

        raise ValueError(
            "score_type must be one of: "
            "'best_N_hat', 'negative_best_d2', 'best_I_ML', 'best_d2'."
        )


# =====================================================================
# 8. OUTLIER / ADAPTATION SAMPLE GENERATORS
# =====================================================================

def generate_directional_outliers(
    template: np.ndarray,
    indices_to_bias: Sequence[int],
    bias_value: float = 0.4,
    n_samples: int = 5,
    noise: float = 0.05,
    flips: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate directional outliers by biasing selected feature indices.

    This preserves the old Figure 3-style adaptation setup.
    """
    rng = np.random.default_rng(seed)
    outliers = []

    for _ in range(int(n_samples)):
        img = generate_noisy_image(
            template=template,
            noise_level_value=noise,
            pixels_affected_by_noise_count=flips,
            rng=rng,
        )

        feat = img.flatten().astype(np.float32)

        for idx in indices_to_bias:
            feat[int(idx)] = np.clip(feat[int(idx)] + bias_value, 0.0, 1.0)

        outliers.append(feat)

    return np.asarray(outliers, dtype=np.float32)


def find_edge_outliers(
    txl: HardwareTXLClassifier,
    target_cls: int,
    template: np.ndarray,
    n_needed: int = 5,
    search_trials: int = 500,
    noise: float = 0.15,
    flips: int = 3,
    distance_low_factor: float = 1.0,
    distance_high_factor: float = 2.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Find samples for a known class that sit near/outside the reliable boundary.

    This is used for IDO/unreliable adaptation demos.
    """
    if target_cls not in txl.arrays_:
        raise KeyError(f"Class {target_cls} is not trained in the classifier.")

    rng = np.random.default_rng(seed)
    array = txl.get_txl_array(target_cls)

    candidates: List[Tuple[np.ndarray, float]] = []

    for _ in range(int(search_trials)):
        img = generate_noisy_image(
            template=template,
            noise_level_value=noise,
            pixels_affected_by_noise_count=flips,
            rng=rng,
        )

        feat = img.flatten().astype(np.float32)
        feat_v = feature_to_voltage(feat, v_min=txl.v_min, v_max=txl.v_max)

        d2_val = array.d2(feat_v)

        lower = array.tau_IDO * distance_low_factor
        # Cap at tau_OOD so "edge outlier" (IDO-band) candidates can never
        # actually be true OOD samples -- matches find_mnist_edge_samples()
        # in the MNIST notebook, which already enforces this bound.
        upper = min(array.tau_IDO * distance_high_factor, array.tau_OOD)

        if lower < d2_val < upper:
            candidates.append((feat, d2_val))

    if not candidates:
        raise RuntimeError(
            "No edge outliers found. Try increasing search_trials, "
            "noise, flips, or the distance_high_factor."
        )

    candidates.sort(key=lambda item: item[1])

    selected = [feat for feat, _ in candidates[:n_needed]]

    return np.asarray(selected, dtype=np.float32)
