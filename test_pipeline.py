"""Run with:  python test_pipeline.py   (or pytest)"""
import numpy as np
import pandas as pd

from pipeline import (FEATURES, REFERENCE_COL, clean_data, external_metrics,
                      internal_metrics, k_sweep, missing_columns, run_models)


def load():
    return pd.read_csv("data/traffic_sample.csv")


def test_cleaning_counts():
    clean, bad = clean_data(load(), FEATURES)
    assert len(clean) == 17 and len(bad) == 7
    assert (bad["reason"] == "missing or non-numeric value").all()


def test_impossible_values_removed():
    df = pd.DataFrame({"speed_kmh": [50, -5, 60], "traffic_volume": [10, 10, 10],
                       "occupancy": [0.2, 0.2, 1.7]})
    clean, bad = clean_data(df, FEATURES)
    assert len(clean) == 1 and set(bad["reason"]) == {"impossible value"}


def test_deselected_feature_does_not_drop_rows():
    df = load()
    clean, _ = clean_data(df, ["speed_kmh", "traffic_volume"])
    assert len(clean) > 17  # occupancy-only problems are no longer dropped


def test_missing_columns():
    assert missing_columns(pd.DataFrame({"speed_kmh": [1]}), FEATURES) == ["traffic_volume", "occupancy"]


def test_models_and_metrics():
    clean, _ = clean_data(load(), FEATURES)
    run = run_models(clean, FEATURES, k=3, eps=0.65, min_samples=4)
    assert run.X.shape == (17, 3) and len(run.km_labels) == 17
    sil, dbi, ch = internal_metrics(run.X, run.km_labels)
    assert -1 <= sil <= 1 and dbi > 0 and ch > 0
    ari, nmi = external_metrics(run.result[REFERENCE_COL], run.km_labels)
    assert -1 <= ari <= 1 and 0 <= nmi <= 1


def test_all_noise_does_not_crash():
    X = np.random.RandomState(0).rand(10, 3)
    assert all(np.isnan(internal_metrics(X, np.full(10, -1))))


def test_k_sweep():
    clean, _ = clean_data(load(), FEATURES)
    run = run_models(clean, FEATURES, 3, 0.65, 4)
    assert list(k_sweep(run.X)["K"]) == list(range(2, 9))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
