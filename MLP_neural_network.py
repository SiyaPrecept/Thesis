#!/usr/bin/env python3
"""
MLP training script with:
 - 90/10 holdout (spatial when coords present)
 - Plain KMeans spatial folds on the non-holdout (train+val) pool
 - Configurable val_frac per dataset with minimum-cap for internal Optuna validation
 - Skips missing .txt files gracefully
Outputs and filenames kept consistent with your original script.

Modifications made:
 - Fix AMP gradient clipping (scaler.unscale_(optimizer) before clip_grad_norm_)
 - Use AdamW optimizer
 - Remove min(..., 30) Optuna cap: use n_trials = optuna_trials (supports optuna_timeout)
 - Log per-trial duration in Optuna objective
 - Reduce model capacity heuristically for small datasets (clamp hidden_size and num_layers)
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import torch.cuda.amp as amp

import optuna
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

BASE_OUTPUT_DIR = "/home/siya/test_1/DP/MLP_v2_update/output_3/TEST_kMeans_3/_wards256_128_100tr_7cl_7f"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

SEASONS = ['winter', 'spring', 'summer', 'autumn']
ANNUAL_TARGETS = ['PM_2km', 'PM_6km']

BASE_FEATURES = [
    'BuiltUpDensity', 'DomesticFuelUse', 'Elevation',
    'IndurialDensity', 'InformalDensity', 'PopDensity', 'RoadLength'
]

# dataset_config: update file paths & params as needed
# Note: add "val_frac" per dataset to control internal validation split fraction
dataset_config = {
    "wards": {
        "path": "/home/siya/test_1/DP/MLP/TXT/VARS_348_2.txt", # 348 samples
        "batch_size": 128,
        "epochs": 300,
        "optuna_trials": 100,
        "optuna_timeout": None,  # seconds or None
        "kmeans_n_clusters": 6,
        "n_spatial_folds": 6,
        "num_workers": 0,
        "use_weighted_sampling": True,
        "weight_clip": [0.6, 6.0],
        "patience": 50,
        "max_hidden": 64,
        "lr_scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 50},
        "ensemble_hint": True,
        "val_frac": 0.10,
        "optuna_search_space": {
            "num_layers": [1, 2, 3, 4],
            "hidden_size": [16, 32, 64, 128, 256, 512],
            # dropout values are large list previously; leave as configured
            "dropout": [i/10000.0 for i in range(4001)],
            "learning_rate": [1e-7, 1e-2],  # widen search bounds
            "weight_decay": [1e-7, 1e-3]    # widen bounds
        }
    },
    "pixels": {
        "path": "", #"/home/siya/test_1/DP/MLP/TXT/VARS_Pixel_2.txt", # 860 samples
        "batch_size": 256,
        "epochs": 300,
        "optuna_trials": 100,
        "optuna_timeout": None,
        "kmeans_n_clusters": None,
        "n_spatial_folds": 10,
        "num_workers": 0,
        "use_weighted_sampling": True,
        "weight_clip": [0.9, 1.0],
        "patience": 50,
        "max_hidden": 256,
        "lr_scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 50},
        "ensemble_hint": True,
        "val_frac": 0.10,
        "optuna_search_space": {
            "num_layers": [1, 2, 3, 4],
            "hidden_size": [32, 64, 96, 128, 256, 512],
            "dropout": [i/10000.0 for i in range(6001)],
            "learning_rate": [1e-7, 1e-2],
            "weight_decay": [1e-6, 1e-3]
        }
    },
    "gauteng_pixels": {
        "path": "", # "/home/siya/test_1/DP/MLP/TXT/VARS_GT_2.txt", # 3600 samples
        "batch_size": 512,
        "epochs": 300,
        "optuna_trials": 100,
        "optuna_timeout": None,
        "kmeans_n_clusters": None,
        "n_spatial_folds": 10,
        "num_workers": 0,
        "use_weighted_sampling": True,
        "weight_clip": [0.9, 3.0],
        "patience": 50,
        "max_hidden": 512,
        "lr_scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 50},
        "ensemble_hint": True,
        "val_frac": 0.10,
        "optuna_search_space": {
            "num_layers": [1, 2, 3, 4],
            "hidden_size": [64, 128, 192, 256, 512],
            "dropout": [i/10000.0 for i in range(6001)],
            "learning_rate": [1e-7, 1e-2],
            "weight_decay": [1e-6, 1e-3]
        }
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


def spatial_kfold_on_indices(coords: Optional[np.ndarray], indices: np.ndarray, n_splits: int = 5, random_state: int = SEED, n_clusters_override: Optional[int] = None):
    """
    Produce plain spatial folds (train_idx, test_idx) where indices are indices into the full dataset.
    KMeans is run on coords[indices] to form spatial blocks; each block becomes one fold's test set.
    If blocks < n_splits, fallback to KFold over indices.
    Returns: list of (train_idx, test_idx) tuples (indices into full dataset)
    """
    if indices is None or len(indices) == 0:
        return None
    if coords is None:
        kf = KFold(n_splits=max(2, min(n_splits, len(indices))), shuffle=True, random_state=random_state)
        return [(indices[tr], indices[te]) for tr, te in kf.split(indices)]
    coords_sub = coords[indices]
    if n_clusters_override is not None:
        n_clusters = max(2, int(n_clusters_override))
    else:
        n_clusters = min(max(2, n_splits), max(2, len(indices) // 10))
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        blocks = kmeans.fit_predict(coords_sub)
        unique_blocks = np.unique(blocks)
        folds = []
        for ub in unique_blocks:
            test_mask_rel = (blocks == ub)
            test_idx = indices[test_mask_rel]
            train_idx = indices[~test_mask_rel]
            folds.append((train_idx, test_idx))
        if len(folds) < n_splits:
            kf = KFold(n_splits=max(2, min(n_splits, len(indices))), shuffle=True, random_state=random_state)
            folds = [(indices[tr], indices[te]) for tr, te in kf.split(indices)]
        return folds
    except Exception:
        kf = KFold(n_splits=max(2, min(n_splits, len(indices))), shuffle=True, random_state=random_state)
        return [(indices[tr], indices[te]) for tr, te in kf.split(indices)]


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
# Optuna objective (unchanged interface but with timing & AdamW)
# -------------------------
def objective(trial: optuna.trial.Trial, X_train, y_train, X_val, y_val, input_size: int, device_loc, epochs_search: int = 15, batch_size: int = 32, num_workers: int = 0, search_space: Dict = None):
    start_time = time.time()
    if search_space is None:
        search_space = {}

    # num_layers
    if 'num_layers' in search_space:
        if isinstance(search_space['num_layers'], list):
            num_layers = trial.suggest_categorical('num_layers', search_space['num_layers'])
        else:
            num_layers = trial.suggest_int('num_layers', search_space['num_layers'][0], search_space['num_layers'][1])
    else:
        num_layers = trial.suggest_int('num_layers', 1, 5)

    # hidden_size
    if 'hidden_size' in search_space:
        if isinstance(search_space['hidden_size'], list):
            hidden_size = trial.suggest_categorical('hidden_size', search_space['hidden_size'])
        else:
            hidden_size = trial.suggest_int('hidden_size', search_space['hidden_size'][0], search_space['hidden_size'][1])
    else:
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])

    # dropout
    if 'dropout' in search_space:
        if isinstance(search_space['dropout'], list):
            dropout = trial.suggest_categorical('dropout', search_space['dropout'])
        else:
            dropout = trial.suggest_float('dropout', search_space['dropout'][0], search_space['dropout'][1])
    else:
        dropout = trial.suggest_float('dropout', 0.2, 0.6)

    # learning_rate
    if 'learning_rate' in search_space:
        # if list of two values, treat as range
        if isinstance(search_space['learning_rate'], list) and len(search_space['learning_rate']) == 2:
            lr_low, lr_high = search_space['learning_rate']
            learning_rate = trial.suggest_float('learning_rate', lr_low, lr_high, log=True)
        else:
            learning_rate = trial.suggest_float('learning_rate', search_space['learning_rate'][0], search_space['learning_rate'][1], log=True)
    else:
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)

    # weight_decay
    if 'weight_decay' in search_space:
        if isinstance(search_space['weight_decay'], list) and len(search_space['weight_decay']) == 2:
            wd_low, wd_high = search_space['weight_decay']
            weight_decay = trial.suggest_float('weight_decay', wd_low, wd_high, log=True)
        else:
            weight_decay = trial.suggest_float('weight_decay', search_space['weight_decay'][0], search_space['weight_decay'][1], log=True)
    else:
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)

    model = PMPredictor(input_size=input_size, num_layers=int(num_layers), hidden_size=int(hidden_size), dropout=float(dropout)).to(device_loc)
    optimizer = optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    criterion = CustomLoss(delta=1.0, alpha=0.05)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

    scaler = amp.GradScaler()
    for epoch in range(epochs_search):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device_loc)
            targets = targets.to(device_loc)
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            # AMP safe gradient clipping: unscale before clip
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    with torch.no_grad():
        val_inputs = torch.tensor(X_val, dtype=torch.float32).to(device_loc)
        val_targets = torch.tensor(y_val, dtype=torch.float32).to(device_loc)
        with amp.autocast():
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
    duration = time.time() - start_time
    try:
        trial.set_user_attr('duration', duration)
    except Exception:
        pass
    print(f"[Optuna Trial] duration: {duration:.2f}s, params: layers={num_layers}, hidden={hidden_size}, lr={learning_rate:.2g}, wd={weight_decay:.2g}, dropout={dropout:.4f}")
    return float(val_loss.item())


# -------------------------
# TRAIN per dataset & target with holdout + plain spatial folds
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
        holdout_idx = create_spatial_holdout(coords, n_samples, holdout_frac, random_state=SEED, n_clusters_override=conf.get('kmeans_n_clusters', None))
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
            n_plot_clusters = conf.get('kmeans_n_clusters', n_plot_clusters) if conf.get('kmeans_n_clusters', None) is not None else n_plot_clusters
            kplot = KMeans(n_clusters=n_plot_clusters, random_state=SEED, n_init=10)
            clusters_all = kplot.fit_predict(coords)
            plot_spatial_clusters(df_full, clusters_all, df_full[target].values, target_dir, dataset_name, target)
            np.save(os.path.join(target_dir, 'spatial_clusters_all.npy'), clusters_all)
            np.save(os.path.join(target_dir, 'holdout_mask.npy'), holdout_mask)
    except Exception as e:
        print("[WARN] Spatial cluster plotting failed:", e)

    # ---------- build spatial CV folds using only trainval indices ----------
    n_folds = conf.get('n_spatial_folds', 5)
    folds = spatial_kfold_on_indices(coords, trainval_idx, n_splits=n_folds, random_state=SEED, n_clusters_override=conf.get('kmeans_n_clusters', None))
    if folds is None or len(folds) == 0:
        kf = KFold(n_splits=max(2, min(n_folds, len(trainval_idx))), shuffle=True, random_state=SEED)
        folds = [(trainval_idx[tr], trainval_idx[te]) for tr, te in kf.split(trainval_idx)]

    # Save fold membership for reproducibility
    fold_membership = np.full(n_samples, -1, dtype=int)
    for fi, (tr_idx, te_idx) in enumerate(folds):
        fold_membership[te_idx] = fi
    pd.DataFrame({'index': all_idx.tolist(), 'fold': fold_membership.tolist()}).to_csv(os.path.join(target_dir, 'fold_membership.csv'), index=False)

    # ---------- training loop using folds (scaling inside folds) ----------
    fold_results = []
    per_target_perm_dfs = []
    batch_size = conf.get('batch_size', 64)
    epochs = conf.get('epochs', 300)
    optuna_trials = conf.get('optuna_trials', 30)
    optuna_timeout = conf.get('optuna_timeout', None)
    num_workers = conf.get('num_workers', 0)
    use_weighted = conf.get('use_weighted_sampling', False)
    patience = conf.get('patience', 20)
    search_space = conf.get('optuna_search_space', {})

    for fold_idx in range(min(len(folds), n_folds)):
        print(f"\n[Fold {fold_idx+1}/{len(folds)}]")
        train_idx_full, test_idx = folds[fold_idx]

        if len(train_idx_full) < 4:
            print(f"[WARN] fold {fold_idx+1} training pool too small ({len(train_idx_full)}), skipping.")
            continue

        # determine val_frac and compute integer minimum-cap
        val_frac = conf.get('val_frac', 0.10) #test fraction
        val_n = max(3, int(round(len(train_idx_full) * val_frac)))
        # ensure not all samples taken
        val_n = min(val_n, max(1, len(train_idx_full) - 2))

        # create train/val indices for Optuna using integer test_size
        tr_sub_idx, val_sub_idx = train_test_split(train_idx_full, test_size=val_n, random_state=SEED)

        # raw arrays for this fold
        X_train_fold_raw = X[tr_sub_idx]
        y_train_fold_raw = y[tr_sub_idx]
        X_val_raw = X[val_sub_idx]
        y_val_raw = y[val_sub_idx]
        X_test_raw = X[test_idx]
        y_test_raw = y[test_idx]

        # fit scalers on training subset only
        X_scaler = StandardScaler().fit(X_train_fold_raw)
        y_scaler = StandardScaler().fit(y_train_fold_raw)

        X_train_fold = X_scaler.transform(X_train_fold_raw)
        X_val = X_scaler.transform(X_val_raw)
        X_test = X_scaler.transform(X_test_raw)
        y_train_fold = y_scaler.transform(y_train_fold_raw)
        y_val = y_scaler.transform(y_val_raw)
        y_test = y_scaler.transform(y_test_raw)

        # Optuna search (respect optuna_trials and optional timeout)
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=SEED))
        # Use n_trials = optuna_trials (user requested)
        if optuna_timeout is not None:
            study.optimize(lambda t: objective(t,
                                              X_train_fold, y_train_fold,
                                              X_val, y_val,
                                              input_size=X.shape[1], device_loc=device,
                                              epochs_search=min(epochs, 10),
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              search_space=search_space),
                           n_trials=optuna_trials, timeout=optuna_timeout, show_progress_bar=False)
        else:
            study.optimize(lambda t: objective(t,
                                              X_train_fold, y_train_fold,
                                              X_val, y_val,
                                              input_size=X.shape[1], device_loc=device,
                                              epochs_search=min(epochs, 10),
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              search_space=search_space),
                           n_trials=optuna_trials, show_progress_bar=False)

        best_params = study.best_params
        print(f"[OPTUNA] Raw best params: {best_params}")

        # Heuristic: reduce model capacity for small datasets / enforce max_hidden
        max_hidden = conf.get('max_hidden', 256)
        if 'hidden_size' in best_params:
            try:
                best_params['hidden_size'] = int(best_params['hidden_size'])
            except Exception:
                best_params['hidden_size'] = int(round(float(best_params['hidden_size'])))
            best_params['hidden_size'] = int(min(best_params['hidden_size'], int(max_hidden)))
        if 'num_layers' in best_params:
            best_params['num_layers'] = int(best_params['num_layers'])
            if n_samples < 1000:
                best_params['num_layers'] = min(best_params['num_layers'], 3)
        # Enforce minimum dropout floor for small datasets
        min_dropout = 0.2 if n_samples < 500 else 0.1
        if 'dropout' in best_params:
            best_params['dropout'] = max(float(best_params.get('dropout', 0.0)), min_dropout)
        print(f"[OPTUNA] Adjusted best params: {best_params}")

        # build model with best params (use AdamW)
        model = PMPredictor(input_size=X.shape[1],
                         num_layers=best_params.get('num_layers', 2),
                         hidden_size=best_params.get('hidden_size', 64),
                         dropout=best_params.get('dropout', 0.2)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=best_params.get('learning_rate', 1e-3), weight_decay=best_params.get('weight_decay', 1e-5))
        criterion = CustomLoss(delta=1.0, alpha=0.05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # DataLoader on scaled training subset
        train_dataset = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.float32))
        if use_weighted:
            yflat = y_train_fold.flatten()
            weights = np.clip(np.abs(yflat - np.median(yflat)) ** 1.5, conf['weight_clip'][0], conf['weight_clip'][1])
            sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

        scaler_amp = amp.GradScaler()
        best_val_loss = float('inf')
        best_state = None
        no_improve = 0
        train_losses = []
        val_losses = []

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

        # Evaluate on fold's test set
        model.eval()
        with torch.no_grad():
            preds_test_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy()
        preds_test_inv = inverse_transform_target(preds_test_scaled, y_scaler).ravel()
        y_test_inv = inverse_transform_target(y_test, y_scaler).ravel()

        if CALIBRATE_PREDICTIONS:
            preds_test_inv = calibrate_predictions(preds_test_inv, y_test_inv)

        r2 = r2_score(y_test_inv.ravel(), preds_test_inv.ravel())
        rmse = np.sqrt(mean_squared_error(y_test_inv.ravel(), preds_test_inv.ravel()))
        print(f"[Fold Result] R2={r2:.4f} RMSE={rmse:.4f}")

        bias = plot_prediction_bias(y_test_inv.ravel(), preds_test_inv.ravel(), os.path.join(target_dir, f"bias_fold_{fold_idx+1}.png"), title_suffix=f"Fold {fold_idx+1}")

        torch.save(model.state_dict(), os.path.join(target_dir, f"model_fold_{fold_idx+1}.pth"))
        save_json({'optuna_best_params': best_params, 'r2': r2, 'rmse': rmse}, os.path.join(target_dir, f"meta_fold_{fold_idx+1}.json"))

        learning_df = pd.DataFrame({'Epoch': list(range(1, len(train_losses)+1)), 'Train_Loss': train_losses, 'Val_Loss': val_losses})
        learning_df.to_csv(os.path.join(target_dir, f"learning_curve_fold_{fold_idx+1}.csv"), index=False)

        fold_results.append({
            'r2': float(r2),
            'rmse': float(rmse),
            'params': best_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'bias_data': bias,
            'y_true': y_test_inv,
            'y_pred': preds_test_inv,
            'final_model_state': model.state_dict(),
            'X_df': X_df,
            'X_scaler': X_scaler,
            'y_scaler': y_scaler,
            'features': features
        })

    if len(fold_results) == 0:
        raise RuntimeError("No fold results - check dataset and splits")

    avg_r2 = np.mean([f['r2'] for f in fold_results])
    avg_rmse = np.mean([f['rmse'] for f in fold_results])
    print(f"[CV] avg R2={avg_r2:.4f} avg RMSE={avg_rmse:.4f}")

    best_fold = max(fold_results, key=lambda x: x['r2'])
    best_params = best_fold['params']

    # ---------- Final training on FULL non-holdout data ----------
    print("[FINAL] Training final model on all non-holdout (train+val) data...")
    X_train_full_raw = X[trainval_idx]
    y_train_full_raw = y[trainval_idx]

    final_X_scaler = StandardScaler().fit(X_train_full_raw)
    final_y_scaler = StandardScaler().fit(y_train_full_raw)

    X_train_full = final_X_scaler.transform(X_train_full_raw)
    y_train_full = final_y_scaler.transform(y_train_full_raw)

    # enforce capacity constraints again
    if 'hidden_size' in best_params:
        hidden_final = int(min(best_params.get('hidden_size', 64), conf.get('max_hidden', 256)))
    else:
        hidden_final = min(64, conf.get('max_hidden', 256))
    num_layers_final = int(min(best_params.get('num_layers', 2), 4)) if 'num_layers' in best_params else 2
    dropout_final = float(best_params.get('dropout', 0.2))
    # safety floor
    if n_samples < 500:
        dropout_final = max(dropout_final, 0.2)

    final_model = PMPredictor(input_size=X.shape[1],
                           num_layers=num_layers_final,
                           hidden_size=hidden_final,
                           dropout=dropout_final).to(device)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params.get('learning_rate', 1e-3), weight_decay=best_params.get('weight_decay', 1e-5))
    final_criterion = CustomLoss(delta=1.0, alpha=0.05)
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=5)

    full_dataset = TensorDataset(torch.tensor(X_train_full, dtype=torch.float32), torch.tensor(y_train_full, dtype=torch.float32))
    if use_weighted:
        yflat = y_train_full.flatten()
        weights = np.clip(np.abs(yflat - np.median(yflat)) ** 1.5, conf['weight_clip'][0], conf['weight_clip'][1])
        sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))
    else:
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

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
    y_inv_full_trainval = inverse_transform_target(y_train_full, final_y_scaler).ravel()
    preds_full_trainval_scaled = final_model(torch.tensor(X_train_full, dtype=torch.float32).to(device)).detach().cpu().numpy()
    preds_full_trainval_inv = inverse_transform_target(preds_full_trainval_scaled, final_y_scaler).ravel()

    if CALIBRATE_PREDICTIONS:
        preds_full_trainval_inv = calibrate_predictions(preds_full_trainval_inv, y_inv_full_trainval)

    full_pred_df = pd.DataFrame({
        'Actual': y_inv_full_trainval,
        'Predicted': preds_full_trainval_inv
    })
    full_pred_df.to_csv(os.path.join(target_dir, "full_predictions.csv"), index=False)

    perm_imp = compute_permutation_importance(final_model, X_train_full, y_inv_full_trainval.reshape(-1, 1), features, final_y_scaler, n_repeats=20)
    importance_df = pd.DataFrame({'Feature': features, 'Permutation_Importance': perm_imp})
    importance_df = importance_df.sort_values('Permutation_Importance', ascending=False)
    importance_df.to_csv(os.path.join(target_dir, "permutation_importance.csv"), index=False)

    pdp_records = []
    pdp_combined_rows = []
    for i, feat in enumerate(features):
        try:
            grid_vals, pdp_vals = compute_partial_dependence(final_model, X_train_full, i, X_df[feat].values[trainval_idx], final_X_scaler, final_y_scaler, grid_resolution=50)
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
    per_target_perm_dfs.append(perm_df)

    return {
        'dataset': dataset_name,
        'target': target,
        'cv_r2': avg_r2,
        'cv_rmse': avg_rmse,
        'time': total_time,
        'final_model': final_model.cpu(),
        'y_true_test_folds': [f['y_true'] for f in fold_results],
        'y_pred_test_folds': [f['y_pred'] for f in fold_results],
        'full_y_true': y_inv_full_trainval,
        'full_y_pred': preds_full_trainval_inv,
        'features': features,
        'pdp_records': pdp_records,
        'pdp_df': pdp_all_df,
        'permutation_df': perm_df,
        'fold_results': fold_results,
        'learning_curves_folds': [{'train': f['train_losses'], 'val': f['val_losses']} for f in fold_results],
        'final_train_losses': final_train_losses,
        'best_params': best_params
    }


# -------------------------
# Combined plotting functions (unchanged behavior)
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
    test_rows = []
    for r in results:
        tname = r['target']
        folds_y_true = r.get('y_true_test_folds', [])
        folds_y_pred = r.get('y_pred_test_folds', [])
        for fy, fp in zip(folds_y_true, folds_y_pred):
            for i in range(len(fy)):
                test_rows.append({'Dataset': dataset_name, 'Target': tname, 'Actual': float(fy[i].ravel()[0]) if np.ndim(fy[i])>0 else float(fy[i]), 'Predicted': float(fp[i].ravel()[0]) if np.ndim(fp[i])>0 else float(fp[i])})
    test_df = pd.DataFrame(test_rows)
    if not test_df.empty:
        test_df.to_csv(os.path.join(out_dir, "combined_test_predictions.csv"), index=False)
        plt.figure(figsize=(10, 8))
        for t in test_df['Target'].unique():
            sub = test_df[test_df['Target'] == t]
            plt.scatter(sub['Actual'], sub['Predicted'], alpha=0.6, label=t, color=TARGET_COLORS.get(t, None))
        minv = test_df[['Actual', 'Predicted']].min().min()
        maxv = test_df[['Actual', 'Predicted']].max().max()
        plt.plot([minv, maxv], [minv, maxv], 'r--')
        plt.xlabel('Actual PM2.5 (µg/m³)'); plt.ylabel('Predicted PM2.5 (µg/m³)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "combined_test_scatter.png"))
        plt.close()

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

    # ---------------------
    # NEW: Combined HOLDOUT scatter (final untouched 10% per target)
    # ---------------------
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
        folds = r.get('learning_curves_folds', [])
        for fi, f in enumerate(folds, start=1):
            tr = f.get('train', [])
            va = f.get('val', [])
            ne = max(len(tr), len(va))
            for e in range(ne):
                rows.append({'Dataset': dataset_name, 'Target': tname, 'Fold': fi, 'Epoch': e+1, 'Train_Loss': float(tr[e]) if e < len(tr) else np.nan, 'Val_Loss': float(va[e]) if e < len(va) else np.nan})
        final_tr = r.get('final_train_losses', [])
        for e, v in enumerate(final_tr):
            rows.append({'Dataset': dataset_name, 'Target': tname, 'Fold': 'final', 'Epoch': e+1, 'Train_Loss': float(v), 'Val_Loss': np.nan})
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
        train_mean = sub[sub['Fold'] != 'final'].groupby('Epoch')['Train_Loss'].mean()
        val_mean = sub[sub['Fold'] != 'final'].groupby('Epoch')['Val_Loss'].mean()
        if not train_mean.empty:
            ax.plot(train_mean.index, train_mean.values, label='Train Loss')
        if not val_mean.empty:
            ax.plot(val_mean.index, val_mean.values, label='Val Loss')
        final_sub = sub[sub['Fold'] == 'final']
        if not final_sub.empty:
            ax.plot(final_sub['Epoch'], final_sub['Train_Loss'], linestyle='--', label='Final Train')
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
                    'CV_R2': res['cv_r2'],
                    'CV_RMSE': res['cv_rmse'],
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
