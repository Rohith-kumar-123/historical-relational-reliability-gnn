
from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import (
    load_transactions,
    add_historical_persistence,
    build_account_mapping,
    make_edge_index,
    build_account_features,
)
from .model import ReliabilityAwareGNN


# ============================================================
# Configuration
# ============================================================

SEED = 42
HIDDEN_CHANNELS = 32
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 5

TARGET_BATCH_SIZE = 1024
MAX_CONTEXT_EDGES = 100_000
LOOKBACK_DAYS = 7


def get_data_path():
    """Allow Colab or local data paths without editing this file."""
    configured_path = os.environ.get("GNN_DATA_PATH")

    if configured_path:
        return Path(configured_path)

    colab_path = Path("/content/drive/MyDrive/HI-Small_Trans.csv")
    if colab_path.exists():
        return colab_path

    return Path.home() / "Downloads" / "HI-Small_Trans.csv"


def get_output_dir():
    """Prefer persistent Drive storage when it is mounted."""
    configured_dir = os.environ.get("GNN_OUTPUT_DIR")

    if configured_dir:
        output_dir = Path(configured_dir)
    elif Path("/content/drive/MyDrive").exists():
        output_dir = (
            Path("/content/drive/MyDrive")
            / "historical-relational-reliability-gnn"
            / "reports"
        )
    else:
        output_dir = Path("reports")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# Reproducibility and device
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device


# ============================================================
# Chronological split
# ============================================================

def chronological_split(df, train_fraction=0.70, val_fraction=0.15):
    """Split near target row counts without splitting tied timestamps."""
    counts = df.groupby("Timestamp", sort=True).size()
    cumulative = counts.cumsum().to_numpy()
    timestamps = counts.index
    total = len(df)

    if len(timestamps) < 3:
        raise ValueError(
            "At least three distinct timestamps are required."
        )

    train_idx = min(
        int(np.searchsorted(
            cumulative, int(total * train_fraction)
        )),
        len(timestamps) - 2,
    )

    val_idx = min(
        max(
            int(np.searchsorted(
                cumulative,
                int(total * (train_fraction + val_fraction)),
            )),
            train_idx + 1,
        ),
        len(timestamps) - 1,
    )

    train_end = timestamps[train_idx]
    val_end = timestamps[val_idx]

    train = df[df["Timestamp"] < train_end].copy()

    val = df[
        (df["Timestamp"] >= train_end)
        & (df["Timestamp"] < val_end)
    ].copy()

    test = df[df["Timestamp"] >= val_end].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")

    assert train["Timestamp"].max() < val["Timestamp"].min()
    assert val["Timestamp"].max() < test["Timestamp"].min()

    return train, val, test


# ============================================================
# Historical context graph
# ============================================================

def sample_context(context, max_edges=MAX_CONTEXT_EDGES):
    """
    Sample historical edges to bound graph memory.

    All target transactions are retained. This approximation means
    message passing does not use every historical context edge.
    """
    if len(context) <= max_edges:
        return context

    return context.sample(
        n=max_edges,
        replace=False,
        random_state=SEED,
    ).sort_values("Timestamp").copy()


def build_context(context, account_to_id, device):
    """Construct graph features and edges from historical data only."""
    context = sample_context(context)

    if context.empty:
        x = torch.zeros(
            (len(account_to_id), 6),
            dtype=torch.float32,
            device=device,
        )
        edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=device
        )
        persistence = torch.empty(
            0, dtype=torch.float32, device=device
        )
        return x, edge_index, persistence, 0

    x = build_account_features(
        context, account_to_id
    ).to(device)

    edge_index = make_edge_index(
        context, account_to_id
    ).to(device)

    persistence = torch.from_numpy(
        context["persistence"].to_numpy(dtype=np.float32)
    ).to(device)

    return x, edge_index, persistence, len(context)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) == 0:
        return {
            "average_precision": float("nan"),
            "roc_auc": float("nan"),
            "precision_at_0_5": float("nan"),
            "recall_at_0_5": float("nan"),
        }

    predicted = y_prob >= 0.5
    true_positives = int(((predicted) & (y_true == 1)).sum())
    predicted_positives = int(predicted.sum())
    actual_positives = int((y_true == 1).sum())

    precision = (
        true_positives / predicted_positives
        if predicted_positives else 0.0
    )
    recall = (
        true_positives / actual_positives
        if actual_positives else float("nan")
    )

    ap = (
        float(average_precision_score(y_true, y_prob))
        if actual_positives else float("nan")
    )

    auc = (
        float(roc_auc_score(y_true, y_prob))
        if np.unique(y_true).size == 2 else float("nan")
    )

    return {
        "average_precision": ap,
        "roc_auc": auc,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
    }


@torch.no_grad()
def evaluate(
    model,
    x,
    context_edge_index,
    context_persistence,
    targets,
    account_to_id,
):
    """Evaluate all targets against a frozen historical snapshot."""
    if targets.empty:
        return calculate_metrics([], [])

    model.eval()

    # One graph encoding per evaluation snapshot.
    h0, h = model.encode_graph(
        x,
        context_edge_index,
        context_persistence,
    )

    probabilities = []
    labels = []
    device = x.device

    for start in range(0, len(targets), TARGET_BATCH_SIZE):
        batch = targets.iloc[start:start + TARGET_BATCH_SIZE]

        target_edge_index = make_edge_index(
            batch, account_to_id
        ).to(device)

        target_persistence = torch.from_numpy(
            batch["persistence"].to_numpy(dtype=np.float32)
        ).to(device)

        logits, _, _ = model.score_edges(
            h0,
            h,
            target_edge_index,
            target_persistence,
        )

        probabilities.extend(
            torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        )
        labels.extend(batch["Is Laundering"].astype(int).tolist())

    return calculate_metrics(labels, probabilities)


# ============================================================
# Main
# ============================================================

def main():
    set_seed()
    device = get_device()
    csv_path = get_data_path()
    output_dir = get_output_dir()

    checkpoint_path = output_dir / "best_model_full.pt"
    metrics_path = output_dir / "full_run_metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Set GNN_DATA_PATH to the location of HI-Small_Trans.csv."
        )

    print(f"Dataset: {csv_path}")
    print(f"Output directory: {output_dir}")
    print("\nLoading complete dataset...")

    df = load_transactions(csv_path)

    print(f"Transactions loaded: {len(df):,}")
    print(f"Unique timestamps: {df['Timestamp'].nunique():,}")

    # Historical persistence is causal: the data function must calculate
    # it from strictly prior calendar days before updating history.
    print("\nCalculating historical persistence...")
    df = add_historical_persistence(
        df, lookback_days=LOOKBACK_DAYS
    )

    train_df, val_df, test_df = chronological_split(df)

    print("\nChronological partitions:")
    for name, part in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        positive_count = int(part["Is Laundering"].sum())

        print(
            f"{name}: {len(part):,} rows | "
            f"positive: {positive_count:,} | "
            f"prevalence: {positive_count / len(part):.6%}"
        )

        if positive_count == 0:
            print(
                f"WARNING: {name} has no positive examples; "
                "fraud metrics will be limited."
            )

    print("\nBuilding account mapping...")
    account_to_id = build_account_mapping(df)
    print(f"Account nodes: {len(account_to_id):,}")

    model = ReliabilityAwareGNN(
        in_channels=6,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_labels = torch.as_tensor(
        train_df["Is Laundering"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    class_counts = torch.bincount(
        train_labels, minlength=2
    ).float()

    class_weights = (
        class_counts.sum()
        / (2.0 * class_counts.clamp_min(1.0))
    )

    history = []
    best_val_ap = -float("inf")
    saved_checkpoint_this_run = False

    train_df["day"] = train_df["Timestamp"].dt.floor("D")
    train_days = sorted(train_df["day"].unique())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0

        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        for day in train_days:
            target_day = train_df[train_df["day"] == day]

            # Only earlier training transactions are context.
            context = train_df[train_df["Timestamp"] < day]

            x, edge_index, edge_persistence, context_size = (
                build_context(context, account_to_id, device)
            )

            print(
                f"{pd.Timestamp(day).date()}: "
                f"{len(target_day):,} target transactions; "
                f"{context_size:,} context edges"
            )

            # Compute the exact denominator for weighted cross-entropy
            # over this complete daily target group.
            day_labels = torch.as_tensor(
                target_day["Is Laundering"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            day_weight_sum = class_weights[day_labels].sum().clamp_min(1e-12)

            optimizer.zero_grad(set_to_none=True)

            # Encode the snapshot only once. Accumulate gradients over
            # target chunks, and update weights after the final chunk.
            h0, h = model.encode_graph(
                x, edge_index, edge_persistence
            )

            daily_loss_value = 0.0
            number_of_batches = (
                len(target_day) + TARGET_BATCH_SIZE - 1
            ) // TARGET_BATCH_SIZE

            for batch_number, start in enumerate(
                range(0, len(target_day), TARGET_BATCH_SIZE)
            ):
                batch = target_day.iloc[
                    start:start + TARGET_BATCH_SIZE
                ]

                target_edge_index = make_edge_index(
                    batch, account_to_id
                ).to(device)

                target_persistence = torch.from_numpy(
                    batch["persistence"].to_numpy(dtype=np.float32)
                ).to(device)

                labels = torch.as_tensor(
                    batch["Is Laundering"].to_numpy(dtype=np.int64),
                    dtype=torch.long,
                    device=device,
                )

                logits, _, _ = model.score_edges(
                    h0,
                    h,
                    target_edge_index,
                    target_persistence,
                )

                # Summed loss divided by the daily weighted denominator.
                loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    reduction="sum",
                ) / day_weight_sum

                is_last_batch = batch_number == number_of_batches - 1

                # Preserve the shared graph computation until the final
                # target chunk has backpropagated.
                loss.backward(retain_graph=not is_last_batch)
                daily_loss_value += float(loss.detach().item())

            # Exactly one optimizer update for this daily snapshot.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss_sum += daily_loss_value * len(target_day)
            epoch_examples += len(target_day)

            del x, edge_index, edge_persistence, h0, h

        mean_loss = epoch_loss_sum / max(epoch_examples, 1)
        print(f"Mean training loss: {mean_loss:.6f}")

        # Validation sees training history only.
        x_val, edge_val, persist_val, val_context_size = build_context(
            train_df, account_to_id, device
        )

        print(
            f"Validation context edges: {val_context_size:,}"
        )

        val_metrics = evaluate(
            model,
            x_val,
            edge_val,
            persist_val,
            val_df,
            account_to_id,
        )

        print("Validation metrics:", val_metrics)

        history.append({
            "epoch": epoch,
            "mean_training_loss": mean_loss,
            **{
                f"validation_{key}": value
                for key, value in val_metrics.items()
            },
        })

        val_ap = val_metrics["average_precision"]

        if (
            not saved_checkpoint_this_run
            or (
                np.isfinite(val_ap)
                and val_ap > best_val_ap
            )
        ):
            if np.isfinite(val_ap):
                best_val_ap = val_ap

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "account_to_id": account_to_id,
                    "context_edge_limit": MAX_CONTEXT_EDGES,
                    "seed": SEED,
                },
                checkpoint_path,
            )

            saved_checkpoint_this_run = True
            print(f"Checkpoint saved: {checkpoint_path}")

        del x_val, edge_val, persist_val

    if not saved_checkpoint_this_run:
        raise RuntimeError("Training ended without saving a checkpoint.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Frozen historical context: train + validation. No test rows
    # are added to the context graph.
    test_context = pd.concat(
        [train_df, val_df],
        ignore_index=True,
    ).sort_values("Timestamp")

    x_test, edge_test, persist_test, test_context_size = build_context(
        test_context, account_to_id, device
    )

    print(f"\nTest context edges: {test_context_size:,}")

    test_metrics = evaluate(
        model,
        x_test,
        edge_test,
        persist_test,
        test_df,
        account_to_id,
    )

    print("\nFinal test metrics:", test_metrics)

    history.append({
        "epoch": "final_test",
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
        },
    })

    pd.DataFrame(history).to_csv(metrics_path, index=False)

    print(f"\nModel checkpoint: {checkpoint_path}")
    print(f"Metrics CSV: {metrics_path}")
    print("Training finished.")

    print(
        "\nResearch notes: all target rows are retained, but historical "
        "context edges are sampled above the configured limit. Validation "
        "and test use frozen context snapshots, not rolling online graphs."
    )


if __name__ == "__main__":
    main()
