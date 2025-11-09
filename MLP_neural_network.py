#!/usr/bin/env python3
"""
MLP training script with fixed parameters (no Optuna search or cross-validation)
- Uses predefined best parameters for each dataset and target combination
- 90/10 holdout (spatial when coords present)
- Skips missing .txt files gracefully
"""

import os
import time
import json
import random
import warnings
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import torch.cuda.amp as amp

import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings('ignore')

# -------------------------
# CONFIG and PARAMETER DICT
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    try:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

BASE_OUTPUT_DIR = "DP/MLP_v2_update/output_4/TEST_ICE/FIXED"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

SEASONS = ['winter', 'spring', 'summer', 'autumn']
ANNUAL_TARGETS = ['PM_2km', 'PM_6km']

BASE_FEATURES = [
    'BuiltUpDensity', 'DomesticFuelUse', 'Elevation',
    'IndurialDensity', 'InformalDensity', 'PopDensity', 'RoadLength'
]

# Fixed parameters for each dataset and target combination
FIXED_PARAMS = {
    "wards": {
        "winter": {"num_layers": 2, "hidden_size": 256, "dropout": 0.206, "learning_rate": 0.00043402393107401364, "weight_decay": 5.877142164013034e-07},
        "spring": {"num_layers": 2, "hidden_size": 256, "dropout": 0.4095, "learning_rate": 0.0005593848730519583, "weight_decay": 1.2566774629660676e-07},
        "summer": {"num_layers": 4, "hidden_size": 256, "dropout": 0.1014, "learning_rate": 0.0004534861435376022, "weight_decay": 1.065486573002254e-06},
        "autumn": {"num_layers": 4, "hidden_size": 128, "dropout": 0.1567, "learning_rate": 0.0005885533707024971, "weight_decay": 1.210847214778256e-07},
        "PM_2km": {"num_layers": 4, "hidden_size": 128, "dropout": 0.1567, "learning_rate": 0.0005885533707024971, "weight_decay": 1.210847214778256e-07},
        "PM_6km": {"num_layers": 2, "hidden_size": 64, "dropout": 0.0883, "learning_rate": 0.0008529434787455408, "weight_decay": 6.5295564478045155e-06}
    },
    "pixels": {
        "winter": {"num_layers": 2, "hidden_size": 128, "dropout": 0.2329, "learning_rate": 0.004198102860416989, "weight_decay": 9.893656898204735e-05},
        "spring": {"num_layers": 2, "hidden_size": 256, "dropout": 0.1623, "learning_rate": 0.0007742347886844931, "weight_decay": 2.4549763701438176e-06},
        "summer": {"num_layers": 3, "hidden_size": 32, "dropout": 0.2492, "learning_rate": 0.005952519612729307, "weight_decay": 7.380238570206174e-05},
        "autumn": {"num_layers": 2, "hidden_size": 64, "dropout": 0.1014, "learning_rate": 0.0054911296616057224, "weight_decay": 9.727612885203307e-05},
        "PM_2km": {"num_layers": 2, "hidden_size": 64, "dropout": 0.1483, "learning_rate": 0.002671590856292085, "weight_decay": 7.220497208826461e-05},
        "PM_6km": {"num_layers": 3, "hidden_size": 64, "dropout": 0.1014, "learning_rate": 0.005491279689423919, "weight_decay": 1.4175714331955482e-05}
    },
    "gauteng_pixels": {
        "winter": {"num_layers": 3, "hidden_size": 512, "dropout": 0.3585, "learning_rate": 0.0007384007987811051, "weight_decay": 0.0007095346628997277},
        "spring": {"num_layers": 4, "hidden_size": 256, "dropout": 0.51, "learning_rate": 0.005955855942149213, "weight_decay": 7.1298527487291914e-06},
        "summer": {"num_layers": 3, "hidden_size": 192, "dropout": 0.0883, "learning_rate": 0.0031360745276739293, "weight_decay": 2.186544454545035e-05},
        "autumn": {"num_layers": 4, "hidden_size": 256, "dropout": 0.0883, "learning_rate": 0.0009159587139072505, "weight_decay": 0.0004133376913704576},
        "PM_2km": {"num_layers": 2, "hidden_size": 256, "dropout": 0.148, "learning_rate": 0.005376162715279499, "weight_decay": 1.1872955064643643e-06},
        "PM_6km": {"num_layers": 3, "hidden_size": 512, "dropout": 0.3585, "learning_rate": 0.0007384007987811051, "weight_decay": 0.0007095346628997277}
    }
}

# dataset_config: update file paths & params as needed
dataset_config = {
    "wards": {
        "path": "/home/siya/test_1/DP/MLP/TXT/VARS_348_2.txt", # 348 samples
        "batch_size": 128,
        "epochs": 300,
        "num_workers": 0,
        "use_weighted_sampling": True,
        "weight_clip": [0.6, 6.0],
        "patience": 50,
        "max_hidden": 128,
        "lr_scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 50},
        "val_frac": 0.10,
    },
    "pixels": {
        "path": "/home/siya/test_1/DP/MLP/TXT/VARS_Pixel_2.txt", # 860 samples
        "batch_size": 256,
        "epochs": 300,
        "num_workers": 0,
        "use_weighted_sampling": True,
        "weight_clip": [0.9, 1.0],
        "patience": 50,
        "max_hidden": 256,
        "lr_scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 50},
        "val_frac": 0.10,
    },
    "gauteng_pixels": {
        "path": "/home/siya/test_1/DP/MLP/TXT/VARS_GT_2.txt", # 3600 samples
        "batch_size": 512,
        "epochs": 300,
        "num_workers": 0,
        "use_weighted_sampling": True,
        "weight_clip": [0.9, 3.0],
        "patience": 50,
        "max_hidden": 512,
        "lr_scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 50},
        "val_frac": 0.10,
    }
}

USE_LOG_TARGET = True
CALIBRATE_PREDICTIONS = True

TARGET_COLORS = {
    'winter': '#A80000',
    'spring': '#FF5B04',
    'summer': '#00A884',
    'autumn': '#0070FF',
    'PM_2km': '#000000',
    'PM_6km': '#E60000'
}

TARGET_LINESTYLES = {
    'winter': '-',
    'spring': '-',
    'summer': '-',
    'autumn': '-',
    'PM_2km': '--',
    'PM_6km': '--'
}

MODEL_DISPLAY_NAMES = {
    'winter': 'Winter',
    'spring': 'Spring',
    'summer': 'Summer',
    'autumn': 'Autumn',
    'PM_2km': '2 km annual',
    'PM_6km': 'Annual'
}

# -------------------------
# UTILITIES
# -------------------------
def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def print_gpu_utilization():
    if device.type == 'cuda':
        try:
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            print(f"[GPU] allocated: {alloc:.1f} MB | reserved: {reserved:.1f} MB")
        except Exception:
            pass
    else:
        print("[GPU] not available")


# -------------------------
# MODEL
# -------------------------
class PMPredictor(nn.Module):
    def __init__(self, input_size, num_layers=3, hidden_size=256, dropout=0.3,
                 use_residual=True, use_attention=False):
        super(PMPredictor, self).__init__()
        self.use_residual = use_residual
        self.use_attention = use_attention

        # Initial projection layer
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Hidden layers with optional residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size) if i > 0 else nn.Identity(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)

        # Optional self-attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_size)
            self.attn_dropout = nn.Dropout(dropout)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        residual = x

        # Process through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            new_x = layer(x)
            if self.use_residual and x.shape == new_x.shape:
                x = x + new_x  # Residual connection
            else:
                x = new_x

        # Optional attention mechanism
        if self.use_attention:
            attn_output, _ = self.attention(x, x, x)
            attn_output = self.attn_dropout(attn_output)
            x = self.attn_norm(x + attn_output)  # Residual connection

        # Final output
        output = self.output_layer(x)
        return output


class CustomLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=0.05):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.alpha = alpha

    def forward(self, pred, target):
        base = self.huber(pred, target)
        errors = pred - target
        bias_penalty = torch.mean(torch.abs(torch.mean(errors, dim=0)))
        return base + self.alpha * bias_penalty


# -------------------------
# Data loading & processing
# -------------------------
def get_features_for_target(target: str) -> List[str]:
    features = BASE_FEATURES.copy()
    if target in SEASONS:
        features.append(f'NDVI{target.capitalize()}')
    else:
        features.append('NDVI_annual')
    return features


def load_and_process_data(file_path: str, target: str):
    """
    Load data and return unscaled X and y (scaling is applied per-fold to avoid leakage).
    """
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path, sep='\t')
    features = get_features_for_target(target)
    has_spatial = 'Lon' in df.columns and 'Lat' in df.columns
    coords = df[['Lon', 'Lat']].values if has_spatial else None

    # ensure features exist in df
    used_features = [f for f in features if f in df.columns]
    if len(used_features) == 0:
        raise KeyError(f"No required features present for target {target}. Expected one of {features}")

    X_df = df[used_features].copy()
    X = X_df.values.astype(np.float32)

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not in {file_path}")

    y = df[target].values.reshape(-1, 1).astype(np.float32)

    if USE_LOG_TARGET:
        y = np.log(np.clip(y, a_min=1e-8, a_max=None))

    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=np.nanmean(X))
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=np.nanmean(y))

    return X.astype(np.float32), y.astype(np.float32), None, None, used_features, X_df, coords, df


def inverse_transform_target(y_scaled: np.ndarray, y_scaler: Optional[StandardScaler]) -> np.ndarray:
    y_scaled = np.asarray(y_scaled)
    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)
    if y_scaler is None:
        y_inv = y_scaled
    else:
        y_inv = y_scaler.inverse_transform(y_scaled)
    if USE_LOG_TARGET:
        return np.exp(y_inv)
    else:
        return y_inv


# -------------------------
# Splitting helpers: holdout + spatial folds (plain KMeans on coords only)
# -------------------------
def create_spatial_holdout(coords: Optional[np.ndarray], n_samples: int, holdout_frac: float = 0.10, random_state: int = SEED, n_clusters_override: Optional[int] = None):
    """
    Return holdout indices selected via whole-cluster selection (~holdout_frac of samples).
    If coords is None, returns None (caller should perform random split elsewhere).
    n_clusters_override: if provided, force this many KMeans clusters.
    """
    if coords is None:
        return None
    if n_clusters_override is not None:
        n_clusters = max(2, int(n_clusters_override))
    else:
        # choose cluster count heuristic relative to samples (not too small)
        n_clusters = min(max(2, n_samples // 25), 50)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(coords)
    unique, counts = np.unique(clusters, return_counts=True)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(unique))
    selected = []
    accum = 0
    target = max(1, int(round(holdout_frac * n_samples)))
    for idx in perm:
        selected.append(unique[idx])
        accum += counts[idx]
        if accum >= target:
            break
    chosen_clusters = np.array(selected, dtype=int) if len(selected) > 0 else np.array([], dtype=int)
    holdout_mask = np.isin(clusters, chosen_clusters)
    holdout_idx = np.where(holdout_mask)[0]
    if len(holdout_idx) == 0:
        holdout_idx = rng.choice(np.arange(n_samples), size=max(1, int(round(holdout_frac * n_samples))), replace=False)
    return holdout_idx


#######
# ICE plots
#######
def compute_ice_mlp(model: nn.Module, X_scaled: np.ndarray, feature_idx: int,
                   original_feature_values: np.ndarray, X_scaler: StandardScaler, 
                   y_scaler: StandardScaler, grid_resolution: int = 50, 
                   sample_limit: Optional[int] = None, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ICE curves for MLP model.
    Returns (vals, ice_array, sample_indices) where ice_array has shape (n_samples, len(vals)).
    """
    # Use observed feature values as grid
    vals_all = np.sort(original_feature_values)
    if vals_all.size == 0:
        raise ValueError(f"Feature {feature_idx} contains no non-null values")

    if grid_resolution is None or grid_resolution <= 0:
        vals_orig = vals_all
    else:
        if vals_all.size <= grid_resolution:
            vals_orig = vals_all
        else:
            probs = np.linspace(0.0, 1.0, grid_resolution)
            vals_orig = np.unique(np.quantile(vals_all, probs))

    # Standardize the grid values
    feat_mean = X_scaler.mean_[feature_idx]
    feat_std = X_scaler.scale_[feature_idx]
    vals_scaled = (vals_orig - feat_mean) / feat_std

    n = X_scaled.shape[0]
    ice = np.zeros((n, len(vals_scaled)))

    # Create modified inputs and get predictions
    X_mod = X_scaled.copy()
    for i, v in enumerate(vals_scaled):
        X_mod[:, feature_idx] = v
        with torch.no_grad():
            preds = model(torch.tensor(X_mod, dtype=torch.float32).to(device)).detach().cpu().numpy()
        preds_inv = inverse_transform_target(preds, y_scaler)
        ice[:, i] = preds_inv.ravel()

    # Subsample if needed
    if sample_limit is not None and sample_limit < n:
        rng = np.random.RandomState(random_state)
        keep = rng.choice(n, size=sample_limit, replace=False)
        return vals_orig, ice[keep, :], keep
    return vals_orig, ice, np.arange(n)

def plot_ice_with_rug(vals: np.ndarray, ice_array: np.ndarray, rug_values: np.ndarray, 
                      output_path: str, feature_name: str, target_name: str,
                      pdp_values: Optional[np.ndarray] = None):
    """
    Plot ICE curves with rug plot and optional PDP line.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot ICE curves
    for i in range(ice_array.shape[0]):
        plt.plot(vals, ice_array[i, :], color='gray', alpha=0.3, linewidth=0.8)
    
    # Plot PDP line if provided
    if pdp_values is not None:
        plt.plot(vals, pdp_values, color='red', linewidth=2.5, label='PDP (mean)')
    
    # Add rug plot
    y_min, y_max = plt.ylim()
    rug_height = 0.05 * (y_max - y_min)
    plt.ylim(y_min - rug_height, y_max)
    rug_y = y_min - 0.5 * rug_height
    plt.plot(rug_values, [rug_y] * len(rug_values), '|', color='black', alpha=0.5)
    
    plt.xlabel(feature_name)
    plt.ylabel('Predicted PM₂.₅ (µg/m³)')
    plt.title(f'ICE Plot - {feature_name} - {target_name}')
    plt.grid(True, alpha=0.3)
    if pdp_values is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -------------------------
# Permutation importance & PDP
# -------------------------
def compute_permutation_importance(model: nn.Module, X_scaled: np.ndarray, y_true_inv: np.ndarray,
                                   features: List[str], y_scaler: Optional[StandardScaler], n_repeats: int = 20) -> np.ndarray:
    model.eval()
    baseline_preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).detach().cpu().numpy()
    baseline_preds_inv = inverse_transform_target(baseline_preds, y_scaler).ravel()
    baseline_score = r2_score(y_true_inv.ravel(), baseline_preds_inv)
    importances = np.zeros(len(features))
    for i in range(len(features)):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_scaled.copy()
            np.random.shuffle(X_perm[:, i])
            with torch.no_grad():
                preds_perm = model(torch.tensor(X_perm, dtype=torch.float32).to(device)).detach().cpu().numpy()
            preds_perm_inv = inverse_transform_target(preds_perm, y_scaler).ravel()
            scores.append(r2_score(y_true_inv.ravel(), preds_perm_inv))
        importances[i] = baseline_score - np.mean(scores)
    return importances


def compute_partial_dependence(model: nn.Module, X_scaled: np.ndarray, feature_idx: int,
                               original_feature_values: np.ndarray,
                               X_scaler: StandardScaler, y_scaler: StandardScaler,
                               grid_resolution: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    feat_min = np.min(original_feature_values)
    feat_max = np.max(original_feature_values)
    grid = np.linspace(feat_min, feat_max, grid_resolution)
    feat_mean = X_scaler.mean_[feature_idx]
    feat_std = X_scaler.scale_[feature_idx]
    standardized_grid = (grid - feat_mean) / feat_std
    pdp_vals = []
    for val in standardized_grid:
        X_mod = X_scaled.copy()
        X_mod[:, feature_idx] = val
        with torch.no_grad():
            preds = model(torch.tensor(X_mod, dtype=torch.float32).to(device)).detach().cpu().numpy()
        preds_inv = inverse_transform_target(preds, y_scaler)
        pdp_vals.append(np.mean(preds_inv))
    return grid, np.array(pdp_vals)


# -------------------------
# Calibration & plotting helpers
# -------------------------
def calibrate_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.array(y_pred).ravel()
    y_true = np.array(y_true).ravel()
    bins = np.quantile(y_true, np.linspace(0, 1, 11))
    calibrated = y_pred.copy()
    for i in range(len(bins)-1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if np.sum(mask) >= 8:
            err = np.mean(y_true[mask] - y_pred[mask])
            calibrated[mask] = y_pred[mask] + 0.7 * err
    return calibrated


def plot_prediction_bias(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title_suffix: str = "") -> Dict[str, Any]:
    errors = np.array(y_pred).ravel() - np.array(y_true).ravel()
    bins = np.linspace(np.min(y_true), np.max(y_true), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_errors = []
    for i in range(len(bins)-1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if np.sum(mask) > 0:
            mean_errors.append(np.mean(errors[mask]))
        else:
            mean_errors.append(np.nan)
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, mean_errors, 'bo-')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('True Concentration (µg/m³)')
    plt.ylabel('Mean Prediction Error (µg/m³)')
    plt.title(f'Systematic Prediction Bias {title_suffix}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return {'bin_centers': bin_centers, 'mean_errors': mean_errors}


def plot_spatial_clusters(df: pd.DataFrame,
                          clusters: np.ndarray,
                          target_values: np.ndarray,
                          output_dir: str,
                          dataset_name: str,
                          target: str,
                          n_bins: int = 4,
                          cmap_name: str = 'viridis'):
    os.makedirs(output_dir, exist_ok=True)

    if 'Lon' not in df.columns or 'Lat' not in df.columns:
        raise KeyError("DataFrame must contain 'Lon' and 'Lat' columns")
    if len(target_values) != len(df):
        raise ValueError("target_values length must match df length")

    try:
        cat, edges = pd.qcut(target_values, q=n_bins, retbins=True, labels=False, duplicates='drop')
        unique_edges = np.unique(edges)
        edges = unique_edges
        bin_indices = np.digitize(target_values, edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(edges)-2)
    except Exception:
        edges = np.quantile(target_values, np.linspace(0.0, 1.0, n_bins+1))
        edges = np.unique(edges)
        if len(edges) < 2:
            bin_indices = np.zeros(len(target_values), dtype=int)
        else:
            interior = edges[1:-1] if len(edges) > 2 else edges[1:-1]
            if len(interior) == 0:
                bin_indices = np.zeros(len(target_values), dtype=int)
            else:
                bin_indices = np.digitize(target_values, interior, right=False)
                bin_indices = np.clip(bin_indices, 0, len(interior))

    n_bins_effective = len(np.unique(bin_indices))
    counts = [int(np.sum(bin_indices == i)) for i in range(n_bins_effective)]

    if 'edges' in locals() and len(edges) >= 2:
        labels = []
        for i in range(len(edges)-1):
            labels.append(f"{edges[i]:.2f}–{edges[i+1]:.2f} µg/m³")
    else:
        labels = [f"bin {i}" for i in range(n_bins_effective)]

    cmap = plt.get_cmap(cmap_name, n_bins_effective)
    colors = cmap(np.arange(n_bins_effective))
    cmap_list = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_bins_effective + 0.5, 1), n_bins_effective)

    plt.figure(figsize=(12, 5))
    ax0 = plt.subplot(1, 2, 1)
    sc0 = ax0.scatter(df['Lon'], df['Lat'], c=clusters, cmap='tab20', s=60, alpha=0.7)
    plt.colorbar(sc0, ax=ax0, label='Cluster ID')
    ax0.set_title(f'Clusters - {dataset_name} - {target}')
    ax0.set_xlabel('Lon'); ax0.set_ylabel('Lat')
    ax0.grid(True)

    ax1 = plt.subplot(1, 2, 2)
    sc1 = ax1.scatter(df['Lon'], df['Lat'], c=bin_indices, cmap=cmap_list, norm=norm, s=60, alpha=0.7)
    cbar = plt.colorbar(sc1, ax=ax1, boundaries=np.arange(n_bins_effective+1)-0.5, ticks=np.arange(n_bins_effective))
    cbar.ax.set_yticklabels(labels[:n_bins_effective])
    cbar.set_label('PM2.5 bin (quantile ranges)')
    ax1.set_title(f'Concentration Bins - {dataset_name} - {target}')
    ax1.set_xlabel('Lon'); ax1.set_ylabel('Lat')
    ax1.grid(True)

    plt.tight_layout()
    out_png = os.path.join(output_dir, 'spatial_clusters_with_concentration.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    cluster_df = pd.DataFrame({
        'Lon': df['Lon'].values,
        'Lat': df['Lat'].values,
        'Cluster': clusters,
        'Concentration': target_values,
        'Bin': bin_indices
    })
    cluster_df.to_csv(os.path.join(output_dir, 'spatial_clusters_with_concentration.csv'), index=False)

    meta = {
        'bin_edges': edges.tolist() if 'edges' in locals() else None,
        'n_bins_requested': n_bins,
        'n_bins_effective': n_bins_effective,
        'counts_per_bin': counts,
        'labels': labels[:n_bins_effective]
    }
    pd.DataFrame([meta]).to_json(os.path.join(output_dir, 'spatial_bins_meta.json'), orient='records', indent=2)

    return {
        'png': out_png,
        'csv': os.path.join(output_dir, 'spatial_clusters_with_concentration.csv'),
        'meta': meta
    }


# -------------------------
# TRAIN per dataset & target with holdout (no cross-validation)
# -------------------------
def train_model_for_dataset(dataset_name: str, target: str, conf: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n[TRAIN] Dataset: {dataset_name} | Target: {target}")
    start = time.time()
    dataset_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    target_dir = os.path.join(dataset_dir, target)
    os.makedirs(target_dir, exist_ok=True)

    # load unscaled data
    try:
        X, y, _, _, features, X_df, coords, df_full = load_and_process_data(conf['path'], target)
    except FileNotFoundError:
        raise RuntimeError(f"Data file not found: {conf['path']}")
    except KeyError as e:
        raise RuntimeError(str(e))

    n_samples = X.shape[0]
    print(f"[DATA] samples={n_samples} features={len(features)}")

    # ---------- create 90/10 HOLDOUT (spatial when coords available) ----------
    holdout_frac = 0.10
    all_idx = np.arange(n_samples)
    if coords is not None:
        holdout_idx = create_spatial_holdout(coords, n_samples, holdout_frac, random_state=SEED)
        holdout_mask = np.zeros(n_samples, dtype=bool)
        holdout_mask[holdout_idx] = True
        trainval_idx = all_idx[~holdout_mask]
    else:
        tr_idx, ho_idx = train_test_split(all_idx, test_size=holdout_frac, random_state=SEED)
        trainval_idx = tr_idx
        holdout_idx = ho_idx

    save_json({
        'holdout_frac': holdout_frac,
        'n_samples': int(n_samples),
        'n_holdout': int(len(holdout_idx)),
        'holdout_indices': holdout_idx.tolist()
    }, os.path.join(target_dir, 'holdout_meta.json'))

    # plotting clusters for diagnostics (cluster on all coords if available)
    try:
        if coords is not None:
            n_plot_clusters = min(max(2, n_samples // 25), 50)
            kplot = KMeans(n_clusters=n_plot_clusters, random_state=SEED, n_init=10)
            clusters_all = kplot.fit_predict(coords)
            plot_spatial_clusters(df_full, clusters_all, df_full[target].values, target_dir, dataset_name, target)
            np.save(os.path.join(target_dir, 'spatial_clusters_all.npy'), clusters_all)
            np.save(os.path.join(target_dir, 'holdout_mask.npy'), holdout_mask)
    except Exception as e:
        print("[WARN] Spatial cluster plotting failed:", e)

    # ---------- Get fixed parameters for this dataset and target ----------
    best_params = FIXED_PARAMS[dataset_name][target]
    print(f"[PARAMS] Using fixed parameters: {best_params}")

    # ---------- Split trainval into train and validation ----------
    val_frac = conf.get('val_frac', 0.10)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_frac, random_state=SEED)

    # raw arrays
    X_train_raw = X[train_idx]
    y_train_raw = y[train_idx]
    X_val_raw = X[val_idx]
    y_val_raw = y[val_idx]

    # fit scalers on training data only
    X_scaler = StandardScaler().fit(X_train_raw)
    y_scaler = StandardScaler().fit(y_train_raw)

    X_train = X_scaler.transform(X_train_raw)
    X_val = X_scaler.transform(X_val_raw)
    y_train = y_scaler.transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)

    # build model with fixed parameters (use AdamW)
    model = PMPredictor(input_size=X.shape[1],
                     num_layers=best_params.get('num_layers', 2),
                     hidden_size=best_params.get('hidden_size', 64),
                     dropout=best_params.get('dropout', 0.2)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params.get('learning_rate', 1e-3), weight_decay=best_params.get('weight_decay', 1e-5))
    criterion = CustomLoss(delta=1.0, alpha=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # DataLoader on scaled training data
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    if conf.get('use_weighted_sampling', False):
        yflat = y_train.flatten()
        weights = np.clip(np.abs(yflat - np.median(yflat)) ** 1.5, conf['weight_clip'][0], conf['weight_clip'][1])
        sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=conf['batch_size'], sampler=sampler, drop_last=False, 
                                 num_workers=conf['num_workers'], pin_memory=(device.type=='cuda'))
    else:
        train_loader = DataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=False, 
                                 num_workers=conf['num_workers'], pin_memory=(device.type=='cuda'))

    scaler_amp = amp.GradScaler()
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    train_losses = []
    val_losses = []
    epochs = conf.get('epochs', 300)
    patience = conf.get('patience', 50)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                out = model(xb)
                loss = criterion(out, yb)
            # AMP-safe clipping/unscale -> clip -> step
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            epoch_train_loss += loss.item() * xb.size(0)
        epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
            with amp.autocast():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop] epoch {epoch+1} stopping (no improvement in {patience})")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | TrainLoss: {epoch_train_loss:.4f} | ValLoss: {val_loss:.4f} | LR: {lr_now:.6g}")
            print_gpu_utilization()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        preds_val_scaled = model(torch.tensor(X_val, dtype=torch.float32).to(device)).detach().cpu().numpy()
    preds_val_inv = inverse_transform_target(preds_val_scaled, y_scaler).ravel()
    y_val_inv = inverse_transform_target(y_val, y_scaler).ravel()

    if CALIBRATE_PREDICTIONS:
        preds_val_inv = calibrate_predictions(preds_val_inv, y_val_inv)

    val_r2 = r2_score(y_val_inv.ravel(), preds_val_inv.ravel())
    val_rmse = np.sqrt(mean_squared_error(y_val_inv.ravel(), preds_val_inv.ravel()))
    print(f"[Validation Result] R2={val_r2:.4f} RMSE={val_rmse:.4f}")

    bias = plot_prediction_bias(y_val_inv.ravel(), preds_val_inv.ravel(), os.path.join(target_dir, "bias_validation.png"), title_suffix="Validation")

    # ---------- Final training on FULL trainval data ----------
    print("[FINAL] Training final model on all trainval data...")
    X_trainval_raw = X[trainval_idx]
    y_trainval_raw = y[trainval_idx]

    final_X_scaler = StandardScaler().fit(X_trainval_raw)
    final_y_scaler = StandardScaler().fit(y_trainval_raw)

    X_trainval = final_X_scaler.transform(X_trainval_raw)
    y_trainval = final_y_scaler.transform(y_trainval_raw)

    # Build final model with same parameters
    final_model = PMPredictor(input_size=X.shape[1],
                           num_layers=best_params.get('num_layers', 2),
                           hidden_size=best_params.get('hidden_size', 64),
                           dropout=best_params.get('dropout', 0.2)).to(device)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params.get('learning_rate', 1e-3), weight_decay=best_params.get('weight_decay', 1e-5))
    final_criterion = CustomLoss(delta=1.0, alpha=0.05)
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=5)

    full_dataset = TensorDataset(torch.tensor(X_trainval, dtype=torch.float32), torch.tensor(y_trainval, dtype=torch.float32))
    if conf.get('use_weighted_sampling', False):
        yflat = y_trainval.flatten()
        weights = np.clip(np.abs(yflat - np.median(yflat)) ** 1.5, conf['weight_clip'][0], conf['weight_clip'][1])
        sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        full_loader = DataLoader(full_dataset, batch_size=conf['batch_size'], sampler=sampler, drop_last=False, 
                                num_workers=conf['num_workers'], pin_memory=(device.type=='cuda'))
    else:
        full_loader = DataLoader(full_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=False, 
                                num_workers=conf['num_workers'], pin_memory=(device.type=='cuda'))

    final_epochs = min(epochs, 200)
    scaler_amp = amp.GradScaler()
    final_train_losses = []
    for epoch in range(final_epochs):
        final_model.train()
        epoch_loss = 0.0
        for xb, yb in full_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            final_optimizer.zero_grad()
            with amp.autocast():
                out = final_model(xb)
                loss = final_criterion(out, yb)
            # AMP-safe unscale -> clip -> step
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(final_optimizer)
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=2.0)
            scaler_amp.step(final_optimizer)
            scaler_amp.update()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss = epoch_loss / len(full_loader.dataset)
        final_train_losses.append(epoch_loss)
        final_scheduler.step(epoch_loss)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"[Final Train] Epoch {epoch+1}/{final_epochs} Loss: {epoch_loss:.4f}")

    final_model.cpu()
    torch.save(final_model.state_dict(), os.path.join(target_dir, "final_model.pth"))
    save_json({'features': features}, os.path.join(target_dir, "final_features.json"))
    np.save(os.path.join(target_dir, "X_scaler_mean.npy"), final_X_scaler.mean_)
    np.save(os.path.join(target_dir, "X_scaler_scale.npy"), final_X_scaler.scale_)
    np.save(os.path.join(target_dir, "y_scaler_mean.npy"), final_y_scaler.mean_)
    np.save(os.path.join(target_dir, "y_scaler_scale.npy"), final_y_scaler.scale_)

    final_model.to(device)
    print("[Interpretation] Computing permutation importance and PDPs...")
    
    #############
    # ICE plot
    #############
    # ICE computation for each feature
    ice_dir = os.path.join(target_dir, "ice")
    os.makedirs(ice_dir, exist_ok=True)
    ice_records_long = []

    for i, feat in enumerate(features):
        try:
            # Get original feature values for rug plot
            rug_values = X_df[feat].values[trainval_idx]
            
            # Compute ICE
            ice_vals, ice_array, sample_idx = compute_ice_mlp(
                final_model, X_trainval, i, rug_values, 
                final_X_scaler, final_y_scaler, grid_resolution=50
            )
            
            # Save ICE data
            ice_df = pd.DataFrame(ice_array, columns=[f"x_{v:.6f}" for v in ice_vals])
            ice_df['Sample_Index'] = sample_idx
            ice_df.to_csv(os.path.join(ice_dir, f"ice_{feat}_rug.csv"), index=False)
            
            # Save long format
            for s_i, s_global in enumerate(sample_idx):
                for xi, xval in enumerate(ice_vals):
                    record = {
                        'Feature': feat,
                        'Sample_Index': int(s_global),
                        'X_Value': float(xval),
                        'Predicted': float(ice_array[s_i, xi])
                    }
                    ice_records_long.append(record)
            
            # Plot ICE with rug
            pdp_vals = None
            plot_ice_with_rug(
                ice_vals, ice_array, rug_values,
                os.path.join(ice_dir, f"ice_{feat}_plot_rug.png"),
                feat, target, pdp_vals
            )
            
        except Exception as e:
            print(f"[WARN] ICE failed for {feat}: {e}")

    # Save combined ICE data
    if ice_records_long:
        ice_combined_df = pd.DataFrame(ice_records_long)
        ice_combined_df.to_csv(os.path.join(ice_dir, "combined_ice_long_format_rug.csv"), index=False)

    y_inv_full_trainval = inverse_transform_target(y_trainval, final_y_scaler).ravel()
    preds_full_trainval_scaled = final_model(torch.tensor(X_trainval, dtype=torch.float32).to(device)).detach().cpu().numpy()
    preds_full_trainval_inv = inverse_transform_target(preds_full_trainval_scaled, final_y_scaler).ravel()

    if CALIBRATE_PREDICTIONS:
        preds_full_trainval_inv = calibrate_predictions(preds_full_trainval_inv, y_inv_full_trainval)

    full_pred_df = pd.DataFrame({
        'Actual': y_inv_full_trainval,
        'Predicted': preds_full_trainval_inv
    })
    full_pred_df.to_csv(os.path.join(target_dir, "full_predictions.csv"), index=False)

    perm_imp = compute_permutation_importance(final_model, X_trainval, y_inv_full_trainval.reshape(-1, 1), features, final_y_scaler, n_repeats=20)
    importance_df = pd.DataFrame({'Feature': features, 'Permutation_Importance': perm_imp})
    importance_df = importance_df.sort_values('Permutation_Importance', ascending=False)
    importance_df.to_csv(os.path.join(target_dir, "permutation_importance.csv"), index=False)

    pdp_records = []
    pdp_combined_rows = []
    for i, feat in enumerate(features):
        try:
            grid_vals, pdp_vals = compute_partial_dependence(final_model, X_trainval, i, X_df[feat].values[trainval_idx], final_X_scaler, final_y_scaler, grid_resolution=50)
            pdp_df = pd.DataFrame({'Feature': feat, 'X_Value': grid_vals, 'PDP': pdp_vals})
            pdp_df.to_csv(os.path.join(target_dir, f"pdp_{feat}.csv"), index=False)
            pdp_records.append({'feature': feat, 'values': grid_vals.tolist(), 'pdp': pdp_vals.tolist()})
            for gv, pv in zip(grid_vals, pdp_vals):
                pdp_combined_rows.append({'Dataset': dataset_name, 'Target': target, 'Feature': feat, 'X_Value': float(gv), 'PDP': float(pv)})
        except Exception as e:
            print(f"[WARN] PDP failed for {feat}: {e}")

    pdp_all_df = pd.DataFrame(pdp_combined_rows)
    pdp_all_df.to_csv(os.path.join(target_dir, "combined_pdp_summary.csv"), index=False)

    # SHAP (small sample)
    print("[Interpretation] Running SHAP (small sample)...")
    shap_dir = os.path.join(target_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)
    X_df_vals = X_df.values[trainval_idx]
    sample_n = min(100, len(X_df_vals))
    sample_idx = np.random.choice(len(X_df_vals), size=sample_n, replace=False)
    X_sample = X_df_vals[sample_idx]

    try:
        explainer = shap.DeepExplainer(final_model, torch.tensor(X_sample[:10], dtype=torch.float32).to(device))
        shap_vals = explainer.shap_values(torch.tensor(X_sample, dtype=torch.float32).to(device))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.array(shap_vals)
        shap_df = pd.DataFrame(shap_vals, columns=features)
        shap_df.to_csv(os.path.join(shap_dir, f"shap_deep_{target}.csv"), index=False)
        shap.summary_plot(shap_vals, X_sample, feature_names=features, show=False)
        plt.savefig(os.path.join(shap_dir, f"shap_summary_deep_{target}.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("[SHAP] DeepExplainer failed, fallback to KernelExplainer:", str(e))
        try:
            bg = X_sample[:10]
            model_predict = lambda x: final_model(torch.tensor(x, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()
            explainer = shap.KernelExplainer(model_predict, bg)
            shap_vals = explainer.shap_values(X_sample, nsamples=100)
            shap_df = pd.DataFrame(shap_vals, columns=features)
            shap_df.to_csv(os.path.join(shap_dir, f"shap_kernel_{target}.csv"), index=False)
            shap.summary_plot(shap_vals, X_sample, feature_names=features, show=False)
            plt.savefig(os.path.join(shap_dir, f"shap_summary_kernel_{target}.png"), bbox_inches='tight')
            plt.close()
        except Exception as e2:
            print("[SHAP] KernelExplainer failed:", str(e2))
            open(os.path.join(shap_dir, 'shap_failed.txt'), 'w').write(str(e2))

    # ---------- EVALUATE HOLDOUT (untouched 10%) ----------
    print("[HOLDOUT] Evaluating final model on untouched holdout set...")
    X_hold_raw = X[holdout_idx]
    y_hold_raw = y[holdout_idx]
    if len(X_hold_raw) > 0:
        X_hold = final_X_scaler.transform(X_hold_raw)
        y_hold_scaled = final_y_scaler.transform(y_hold_raw)
        final_model.to(device)
        final_model.eval()
        with torch.no_grad():
            preds_hold_scaled = final_model(torch.tensor(X_hold, dtype=torch.float32).to(device)).detach().cpu().numpy()
        preds_hold_inv = inverse_transform_target(preds_hold_scaled, final_y_scaler).ravel()
        y_hold_inv = inverse_transform_target(y_hold_scaled, final_y_scaler).ravel()
        if CALIBRATE_PREDICTIONS:
            preds_hold_inv = calibrate_predictions(preds_hold_inv, y_hold_inv)
        hold_r2 = r2_score(y_hold_inv.ravel(), preds_hold_inv.ravel()) if len(y_hold_inv) > 1 else float('nan')
        hold_rmse = np.sqrt(mean_squared_error(y_hold_inv.ravel(), preds_hold_inv.ravel())) if len(y_hold_inv) > 1 else float('nan')
        print(f"[HOLDOUT RESULT] R2={hold_r2:.4f} RMSE={hold_rmse:.4f}")

        holdout_df = pd.DataFrame({
            'Actual': y_hold_inv,
            'Predicted': preds_hold_inv
        })
        if 'Lon' in df_full.columns and 'Lat' in df_full.columns:
            holdout_df['Lon'] = df_full['Lon'].values[holdout_idx]
            holdout_df['Lat'] = df_full['Lat'].values[holdout_idx]
        holdout_df.to_csv(os.path.join(target_dir, "holdout_predictions.csv"), index=False)
        save_json({
            'holdout_r2': float(hold_r2) if not np.isnan(hold_r2) else None,
            'holdout_rmse': float(hold_rmse) if not np.isnan(hold_rmse) else None,
            'n_holdout': int(len(holdout_idx))
        }, os.path.join(target_dir, "holdout_results.json"))
    else:
        print("[HOLDOUT] No holdout samples found; skipping holdout evaluation.")
        save_json({'n_holdout': 0}, os.path.join(target_dir, "holdout_results.json"))

    total_time = time.time() - start
    print(f"[DONE] {dataset_name} - {target} in {total_time:.1f}s")

    # prepare permutation df for combining
    perm_df = pd.DataFrame({'Feature': features, 'Permutation_Importance': perm_imp})
    perm_df['Dataset'] = dataset_name
    perm_df['Target'] = target

    return {
        'dataset': dataset_name,
        'target': target,
        'val_r2': val_r2,
        'val_rmse': val_rmse,
        'holdout_r2': hold_r2,
        'holdout_rmse': hold_rmse,
        'time': total_time,
        'final_model': final_model.cpu(),
        'full_y_true': y_inv_full_trainval,
        'full_y_pred': preds_full_trainval_inv,
        'features': features,
        'pdp_records': pdp_records,
        'pdp_df': pdp_all_df,
        'permutation_df': perm_df,
        'final_train_losses': final_train_losses,
        'best_params': best_params
    }


# -------------------------
# Combined plotting functions
# -------------------------
def create_combined_pdp_plot(dataset_name: str, results: List[Dict[str, Any]]):
    out_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for r in results:
        if r.get('pdp_df') is None:
            continue
        df = r['pdp_df']
        df['Target'] = r['target']
        rows.append(df)
    if not rows:
        print("[PDP] No PDP results to combine.")
        return
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "combined_pdp_all_targets.csv"), index=False)

    features = sorted(combined['Feature'].unique())
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), squeeze=False)
    axs = axs.flatten()
    legend_added = False
    for i, feat in enumerate(features):
        ax = axs[i]
        feat_df = combined[combined['Feature'] == feat]
        for target in feat_df['Target'].unique():
            tdf = feat_df[feat_df['Target'] == target].sort_values('X_Value')
            ax.plot(tdf['X_Value'], tdf['PDP'],
                    color=TARGET_COLORS.get(target, 'gray'),
                    linestyle=TARGET_LINESTYLES.get(target, '-'),
                    linewidth=2, label=MODEL_DISPLAY_NAMES.get(target, target))
        ax.set_title(feat, fontsize=12)
        ax.grid(alpha=0.3)
        if not legend_added:
            ax.legend(fontsize=10)
            legend_added = True
    for j in range(len(features), len(axs)):
        axs[j].axis('off')
    plt.suptitle(f"Partial Dependence - {dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "combined_pdp_plot.png"), dpi=300)
    plt.close()
    print(f"[PDP] Combined PDP plot + CSV saved to {out_dir}")


def create_combined_scatter_plots(dataset_name: str, results: List[Dict[str, Any]]):
    out_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    
    full_rows = []
    for r in results:
        tname = r['target']
        y_true = r.get('full_y_true', None)
        y_pred = r.get('full_y_pred', None)
        if y_true is None or y_pred is None:
            continue
        for a, p in zip(y_true, y_pred):
            full_rows.append({'Dataset': dataset_name, 'Target': tname, 'Actual': float(a), 'Predicted': float(p)})
    full_df = pd.DataFrame(full_rows)
    if not full_df.empty:
        full_df.to_csv(os.path.join(out_dir, "combined_full_predictions.csv"), index=False)
        plt.figure(figsize=(10, 8))
        for t in full_df['Target'].unique():
            sub = full_df[full_df['Target'] == t]
            plt.scatter(sub['Actual'], sub['Predicted'], alpha=0.5, label=t, color=TARGET_COLORS.get(t, None))
        minv = full_df[['Actual', 'Predicted']].min().min()
        maxv = full_df[['Actual', 'Predicted']].max().max()
        plt.plot([minv, maxv], [minv, maxv], 'r--')
        plt.xlabel('Actual PM2.5 (µg/m³)'); plt.ylabel('Predicted PM2.5 (µg/m³)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "combined_full_scatter.png"))
        plt.close()
    print(f"[SCATTER] Combined scatter plots + CSV saved to {out_dir}")

    # Combined HOLDOUT scatter (final untouched 10% per target)
    holdout_rows = []
    for r in results:
        tname = r['target']
        tdir = os.path.join(BASE_OUTPUT_DIR, dataset_name, tname)
        hold_csv = os.path.join(tdir, 'holdout_predictions.csv')
        if os.path.exists(hold_csv):
            try:
                dfh = pd.read_csv(hold_csv)
            except Exception as e:
                print(f"[HOLDOUT] Failed to read {hold_csv}: {e}")
                continue
            # require 'Actual' and 'Predicted' columns
            if {'Actual', 'Predicted'}.issubset(dfh.columns):
                dfh = dfh.rename(columns={'Actual': 'Actual', 'Predicted': 'Predicted'})
                dfh['Target'] = tname
                # preserve optional coords if present
                holdout_rows.append(dfh)
            else:
                print(f"[HOLDOUT] {hold_csv} missing 'Actual'/'Predicted' columns - skipping.")

    if holdout_rows:
        hold_df = pd.concat(holdout_rows, ignore_index=True)
        hold_df.to_csv(os.path.join(out_dir, "combined_holdout_predictions.csv"), index=False)

        plt.figure(figsize=(10, 8))
        for t in hold_df['Target'].unique():
            sub = hold_df[hold_df['Target'] == t]
            plt.scatter(sub['Actual'], sub['Predicted'], alpha=0.7, label=t, color=TARGET_COLORS.get(t, None))
        mn = hold_df[['Actual', 'Predicted']].min().min()
        mx = hold_df[['Actual', 'Predicted']].max().max()
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('Actual PM2.5 (µg/m³)'); plt.ylabel('Predicted PM2.5 (µg/m³)')
        plt.title('Combined Holdout (final test) Predictions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "combined_holdout_scatter.png"))
        plt.close()
        print(f"[HOLDOUT] Combined holdout CSV + scatter saved to {out_dir}")
    else:
        print(f"[HOLDOUT] No holdout prediction files found for dataset {dataset_name} (combined_holdout_scatter skipped).")


def create_combined_permutation_csv(dataset_name: str, results: List[Dict[str, Any]]):
    out_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for r in results:
        perm_df = r.get('permutation_df', None)
        if perm_df is not None:
            rows.append(perm_df)
    if not rows:
        print("[PERM] No permutation data found.")
        return
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "all_permutations_combined.csv"), index=False)
    print(f"[PERM] Combined permutation CSV saved to {out_dir}")


def create_combined_learning_curves(dataset_name: str, results: List[Dict[str, Any]]):
    out_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for r in results:
        tname = r['target']
        final_tr = r.get('final_train_losses', [])
        for e, v in enumerate(final_tr):
            rows.append({'Dataset': dataset_name, 'Target': tname, 'Epoch': e+1, 'Train_Loss': float(v)})
    if not rows:
        print("[LRN] No learning curves found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "combined_learning_curves.csv"), index=False)

    targets = sorted(df['Target'].unique())
    n_cols = 3
    n_rows = (len(targets) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), squeeze=False)
    axs = axs.flatten()
    for i, t in enumerate(targets):
        ax = axs[i]
        sub = df[df['Target'] == t]
        if not sub.empty:
            ax.plot(sub['Epoch'], sub['Train_Loss'], label='Final Train')
        ax.set_title(t)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()
    for j in range(len(targets), len(axs)):
        axs[j].axis('off')
    plt.suptitle(f"Learning Curves - {dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "combined_learning_curves.png"))
    plt.close()
    print(f"[LRN] Combined learning curves + CSV saved to {out_dir}")


# -------------------------
# MAIN: iterate dataset_config, skip missing files
# -------------------------
def main():
    summary = []
    for ds_name, conf in dataset_config.items():
        print(f"\n===== DATASET: {ds_name} =====")
        path = conf.get('path', '')
        if not path or not os.path.isfile(path):
            print(f"[SKIP] Data file for dataset '{ds_name}' missing or path empty: '{path}'")
            continue
        results_per_dataset = []
        for t in SEASONS + ANNUAL_TARGETS:
            try:
                res = train_model_for_dataset(ds_name, t, conf)
                results_per_dataset.append(res)
                summary.append({
                    'Dataset': ds_name,
                    'Target': t,
                    'Validation_R2': res['val_r2'],
                    'Validation_RMSE': res['val_rmse'],
                    'Holdout_R2': res['holdout_r2'],
                    'Holdout_RMSE': res['holdout_rmse'],
                    'Time_s': res['time'],
                    'Best_Params': json.dumps(res.get('best_params', {}))
                })
            except Exception as ex:
                print(f"[ERROR] training {ds_name} {t}: {ex}")

        # Combined outputs for dataset
        create_combined_pdp_plot(ds_name, results_per_dataset)
        create_combined_scatter_plots(ds_name, results_per_dataset)
        create_combined_permutation_csv(ds_name, results_per_dataset)
        create_combined_learning_curves(ds_name, results_per_dataset)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "summary_results.csv"), index=False)
    print("[MAIN] All done. Summary saved.")


if __name__ == "__main__":
    main()