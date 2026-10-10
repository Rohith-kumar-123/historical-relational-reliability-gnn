
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

from .data import load_transactions, add_historical_persistence



def chronological_split(df, train_fraction=0.70, val_fraction=0.15):
    """Chronological split by transaction counts without splitting tied timestamps."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Train + validation fractions must be < 1")

    # Count transactions at each timestamp, in chronological order.
    counts = (
        df.groupby("Timestamp", sort=True)
        .size()
        .rename("count")
    )

    if len(counts) < 3:
        raise ValueError("At least 3 distinct timestamps are required")

    cumulative = counts.cumsum().to_numpy()
    total = len(df)

    # Choose timestamp boundaries closest to the desired row counts.
    train_target = int(total * train_fraction)
    val_target = int(total * (train_fraction + val_fraction))

    train_idx = int(np.searchsorted(cumulative, train_target))
    train_idx = min(train_idx, len(counts) - 2)

    val_idx = int(np.searchsorted(cumulative, val_target))
    val_idx = max(val_idx, train_idx + 1)
    val_idx = min(val_idx, len(counts) - 1)

    timestamps = counts.index
    train_end = timestamps[train_idx]
    val_end = timestamps[val_idx]

    train_df = df[df["Timestamp"] < train_end].copy()
    val_df = df[
        (df["Timestamp"] >= train_end)
        & (df["Timestamp"] < val_end)
    ].copy()
    test_df = df[df["Timestamp"] >= val_end].copy()

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    assert train_df["Timestamp"].max() < val_df["Timestamp"].min()
    assert val_df["Timestamp"].max() < test_df["Timestamp"].min()

    return train_df, val_df, test_df

def calculate_class_weights(labels):
    """Calculate inverse-frequency weights for binary classification."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float()
    return counts.sum() / (2.0 * counts.clamp_min(1.0))


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    """Calculate metrics for the laundering class."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def prepare_dataset(csv_path):
    """Load the full dataset, compute persistence, diagnose, and split."""
    df = load_transactions(csv_path)

    print(f"\nLoaded transactions: {len(df):,}")
    print(f"Timestamp dtype: {df['Timestamp'].dtype}")
    print(f"Missing timestamps: {df['Timestamp'].isna().sum():,}")
    print(f"Timestamp range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    # Diagnose the daily distribution before splitting.
    day = df["Timestamp"].dt.floor("D")
    daily_counts = day.value_counts().sort_index()

    print(f"\nDistinct calendar days: {len(daily_counts)}")
    print("\n--- DEBUG: transactions per calendar day ---")
    print(daily_counts.to_string())

    # Compute persistence using prior days only.
    print("\nCalculating 7-day historical persistence...")
    df = add_historical_persistence(df, lookback_days=7)

    # Split using whole calendar days.
    train_df, val_df, test_df = chronological_split(df)

    print("\n--- FINAL SPLIT ---")
    print(f"All transactions: {len(df):,}")

    for name, part in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(f"\n{name} transactions: {len(part):,}")
        print(
            f"{name} dates: {part['Timestamp'].min()} "
            f"to {part['Timestamp'].max()}"
        )
        print(
            f"{name} laundering transactions: "
            f"{int(part['Is Laundering'].sum()):,}"
        )
        print(
            f"{name} laundering rate: "
            f"{part['Is Laundering'].mean():.6f}"
        )

    total_split = len(train_df) + len(val_df) + len(test_df)
    print(f"\nSplit total: {total_split:,}")
    print(f"Original total: {len(df):,}")

    if total_split != len(df):
        raise RuntimeError("Some transactions were lost during splitting")

    return df, train_df, val_df, test_df


if __name__ == "__main__":
    csv_path = Path.home() / "Downloads" / "HI-Small_Trans.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df, train_df, val_df, test_df = prepare_dataset(csv_path)
