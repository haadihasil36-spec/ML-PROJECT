"""Core logic for Traffic Flow Pattern Analysis (no Streamlit imports, so it is testable)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

FEATURES = ["speed_kmh", "traffic_volume", "occupancy"]
REFERENCE_COL = "illustrative_state"  # optional column used only for external validation


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def missing_columns(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in features if c not in df.columns]


def clean_data(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (clean, bad) based on the *selected* features only.

    A row is bad if a selected feature is missing / non-numeric, or has an impossible
    value (negative speed or volume, occupancy outside 0-1). `bad` gets a `reason` column.
    """
    work = df.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")

    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    reason = pd.Series("", index=work.index, dtype=object)
    reason[work[features].isna().any(axis=1)] = "missing or non-numeric value"

    impossible = pd.Series(False, index=work.index)
    for col in features:
        if col == "occupancy":
            impossible |= (work[col] < 0) | (work[col] > 1)
        else:
            impossible |= work[col] < 0
    reason[(reason == "") & impossible] = "impossible value"

    is_bad = reason != ""
    bad = work[is_bad].assign(reason=reason[is_bad])
    clean = work[~is_bad]
    return clean.reset_index(drop=True), bad.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def internal_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz. DBSCAN noise (-1) is excluded.
    Returns NaNs when fewer than 2 clusters (or too few points) remain."""
    mask = labels != -1
    if mask.sum() < 3 or len(np.unique(labels[mask])) < 2:
        return np.nan, np.nan, np.nan
    x, y = X[mask], labels[mask]
    # silhouette needs 2 <= n_clusters <= n_samples - 1
    if len(np.unique(y)) > len(y) - 1:
        return np.nan, np.nan, np.nan
    return silhouette_score(x, y), davies_bouldin_score(x, y), calinski_harabasz_score(x, y)


def external_metrics(reference: pd.Series, labels: np.ndarray) -> tuple[float, float]:
    """Adjusted Rand Index and NMI against a reference labelling (noise counts as its own group)."""
    return (
        adjusted_rand_score(reference, labels),
        normalized_mutual_info_score(reference, labels),
    )


def k_sweep(X: np.ndarray, k_max: int = 8, seed: int = 42) -> pd.DataFrame:
    """Silhouette and inertia for K = 2..k_max, to help choose K."""
    rows = []
    for k in range(2, min(k_max, len(X) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        rows.append({"K": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Modelling (run once, reused by every tab)
# ----------------------------------------------------------------------------
@dataclass
class ModelRun:
    X: np.ndarray
    km_labels: np.ndarray
    db_labels: np.ndarray
    pcs: np.ndarray
    explained_variance: np.ndarray
    result: pd.DataFrame


def run_models(clean: pd.DataFrame, features: list[str], k: int, eps: float, min_samples: int,
               seed: int = 42) -> ModelRun:
    X = StandardScaler().fit_transform(clean[features])
    km_labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_comp = min(2, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    pcs = pca.fit_transform(X)
    if n_comp == 1:  # keep a 2-column array for plotting
        pcs = np.column_stack([pcs[:, 0], np.zeros(len(pcs))])

    result = clean.copy()
    result["KMeans_Cluster"] = km_labels
    result["DBSCAN_Cluster"] = db_labels
    result["DBSCAN_Noise"] = db_labels == -1
    return ModelRun(X, km_labels, db_labels, pcs, pca.explained_variance_ratio_, result)
