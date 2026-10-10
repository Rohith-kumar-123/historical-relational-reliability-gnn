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
    configured = os.environ.get("GNN_DATA_PATH")
    if configured:
        return Path(configured)

    colab = Path("/content/drive/MyDrive/HI-Small_Trans.csv")
    if colab.exists():
        return colab

    return Path.home() / "Downloads" / "HI-Small_Trans.csv"


def get_output_dir():
    configured = os.environ.get("GNN_OUTPUT_DIR")
    if configured:
        output = Path(configured)
    elif Path("/content/drive/MyDrive").exists():
        output = (
            Path("/content/drive/MyDrive")
            / "historical-relational-reliability-gnn"
            / "reports"
        )
    else:
        output = Path("reports")

    output.mkdir(parents=True, exist_ok=True)
    return output


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    return device


def chronological_split(df, train_fraction=0.70, val_fraction=0.15):
    counts = df.groupby("Timestamp", sort=True).size()
    cumulative = counts.cumsum().to_numpy()
    timestamps = counts.index
    total = len(df)

    if len(timestamps) < 3:
        raise ValueError("At least three distinct timestamps are required.")

    train_idx = min(
        int(np.searchsorted(cumulative, int(total * train_fraction))),
        len(timestamps) - 2,
    )
    val_idx = min(
        max(
            int(np.searchsorted(
                cumulative, int(total * (train_fraction + val_fraction))
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


def sample_context(context):
    if len(context) <= MAX_CONTEXT_EDGES:
        return context
    return context.sample(
        n=MAX_CONTEXT_EDGES, replace=False, random_state=SEED
    ).sort_values("Timestamp").copy()


def build_context(context, account_to_id, device):
    context = sample_context(context)

    if context.empty:
        x = torch.zeros(
            (len(account_to_id), 6), dtype=torch.float32, device=device
        )
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        persistence = torch.empty(0, dtype=torch.float32, device=device)
        return x, edge_index, persistence, 0

    x = build_account_features(context, account_to_id).to(device)
    edge_index = make_edge_index(context, account_to_id).to(device)
    persistence = torch.as_tensor(
        context["persistence"].to_numpy(dtype=np.float32), device=device
    )
    return x, edge_index, persistence, len(context)


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
    tp = int((predicted & (y_true == 1)).sum())
    predicted_positive = int(predicted.sum())
    actual_positive = int((y_true == 1).sum())

    return {
        "average_precision": (
            float(average_precision_score(y_true, y_prob))
            if actual_positive else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if np.unique(y_true).size == 2 else float("nan")
        ),
        "precision_at_0_5": (
            tp / predicted_positive if predicted_positive else 0.0
        ),
        "recall_at_0_5": (
            tp / actual_positive if actual_positive else float("nan")
        ),
    }


@torch.no_grad()
def evaluate(model, x, edge_index, edge_persistence,
             targets, account_to_id):
    if targets.empty:
        return calculate_metrics([], [])

    model.eval()
    h0, h = model.encode_graph(x, edge_index, edge_persistence)

    probabilities, labels = [], []
    device = x.device

    for start in range(0, len(targets), TARGET_BATCH_SIZE):
        batch = targets.iloc[start:start + TARGET_BATCH_SIZE]
        target_edges = make_edge_index(batch, account_to_id).to(device)
        target_persistence = torch.as_tensor(
            batch["persistence"].to_numpy(dtype=np.float32), device=device
        )

        logits, _, _ = model.score_edges(
            h0, h, target_edges, target_persistence
        )
        probabilities.extend(
            torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        )
        labels.extend(batch["Is Laundering"].astype(int).tolist())

    return calculate_metrics(labels, probabilities)


def main():
    set_seed()
    device = get_device()

    mode = os.environ.get("GNN_ABLATION", "full")
    if mode not in {"full", "A", "B"}:
        raise ValueError("GNN_ABLATION must be full, A, or B")

    csv_path = get_data_path()
    output_dir = get_output_dir()
    checkpoint_path = output_dir / f"best_model_{mode}.pt"
    metrics_path = output_dir / f"metrics_{mode}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print("Experiment:", mode)
    print("Dataset:", csv_path)
    print("Output directory:", output_dir)
    print("\nLoading complete dataset...")

    df = load_transactions(csv_path)
    print(f"Transactions loaded: {len(df):,}")
    print(f"Unique timestamps: {df['Timestamp'].nunique():,}")

    print("\nCalculating historical persistence...")
    df = add_historical_persistence(df, lookback_days=LOOKBACK_DAYS)

    train_df, val_df, test_df = chronological_split(df)

    for name, part in [
        ("Train", train_df), ("Validation", val_df), ("Test", test_df)
    ]:
        positives = int(part["Is Laundering"].sum())
        print(
            f"{name}: {len(part):,} rows | positives: {positives:,} | "
            f"prevalence: {positives / len(part):.6%}"
        )

    print("\nBuilding account mapping...")
    account_to_id = build_account_mapping(df)
    print("Account nodes:", f"{len(account_to_id):,}")

    model = ReliabilityAwareGNN(
        in_channels=6,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
        ablation_mode=mode,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_labels = torch.as_tensor(
        train_df["Is Laundering"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = counts.sum() / (2.0 * counts.clamp_min(1.0))

    history = []
    best_val_ap = -float("inf")
    saved_checkpoint = False

    train_df["day"] = train_df["Timestamp"].dt.floor("D")
    train_days = sorted(train_df["day"].unique())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        print(f"\n========== {mode}: Epoch {epoch}/{EPOCHS} ==========")

        for day in train_days:
            targets = train_df[train_df["day"] == day]
            context = train_df[train_df["Timestamp"] < day]
            x, edge_index, edge_persistence, context_size = build_context(
                context, account_to_id, device
            )

            print(
                f"{pd.Timestamp(day).date()}: {len(targets):,} targets; "
                f"{context_size:,} context edges"
            )

            labels_day = torch.as_tensor(
                targets["Is Laundering"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            denominator = class_weights[labels_day].sum().clamp_min(1e-12)

            optimizer.zero_grad(set_to_none=True)
            h0, h = model.encode_graph(x, edge_index, edge_persistence)

            daily_loss = 0.0
            n_batches = (len(targets) + TARGET_BATCH_SIZE - 1) // TARGET_BATCH_SIZE

            for batch_num, start in enumerate(
                range(0, len(targets), TARGET_BATCH_SIZE)
            ):
                batch = targets.iloc[start:start + TARGET_BATCH_SIZE]
                target_edges = make_edge_index(batch, account_to_id).to(device)
                target_persistence = torch.as_tensor(
                    batch["persistence"].to_numpy(dtype=np.float32),
                    device=device,
                )
                labels = torch.as_tensor(
                    batch["Is Laundering"].to_numpy(dtype=np.int64),
                    dtype=torch.long,
                    device=device,
                )

                logits, _, _ = model.score_edges(
                    h0, h, target_edges, target_persistence
                )
                loss = F.cross_entropy(
                    logits, labels, weight=class_weights, reduction="sum"
                ) / denominator

                last_batch = batch_num == n_batches - 1
                loss.backward(retain_graph=not last_batch)
                daily_loss += float(loss.detach().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += daily_loss * len(targets)
            examples += len(targets)
            del x, edge_index, edge_persistence, h0, h

        mean_loss = loss_sum / max(examples, 1)
        print(f"Mean training loss: {mean_loss:.6f}")

        x_val, edge_val, persist_val, val_size = build_context(
            train_df, account_to_id, device
        )
        print("Validation context edges:", f"{val_size:,}")

        val_metrics = evaluate(
            model, x_val, edge_val, persist_val, val_df, account_to_id
        )
        print("Validation metrics:", val_metrics)

        history.append({
            "epoch": epoch,
            "mean_training_loss": mean_loss,
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        })

        val_ap = val_metrics["average_precision"]
        if (
            not saved_checkpoint
            or (np.isfinite(val_ap) and val_ap > best_val_ap)
        ):
            if np.isfinite(val_ap):
                best_val_ap = val_ap

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "ablation_mode": mode,
                "validation_metrics": val_metrics,
                "account_to_id": account_to_id,
                "context_edge_limit": MAX_CONTEXT_EDGES,
                "seed": SEED,
            }, checkpoint_path)

            saved_checkpoint = True
            print("Checkpoint saved:", checkpoint_path)

        del x_val, edge_val, persist_val

    if not saved_checkpoint:
        raise RuntimeError("No checkpoint was saved.")

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_context = pd.concat(
        [train_df, val_df], ignore_index=True
    ).sort_values("Timestamp")

    x_test, edge_test, persist_test, test_size = build_context(
        test_context, account_to_id, device
    )
    print("\nTest context edges:", f"{test_size:,}")

    test_metrics = evaluate(
        model, x_test, edge_test, persist_test, test_df, account_to_id
    )
    print("Final test metrics:", test_metrics)

    history.append({
        "epoch": "final_test",
        **{f"test_{k}": v for k, v in test_metrics.items()},
    })
    pd.DataFrame(history).to_csv(metrics_path, index=False)

    print("Model checkpoint:", checkpoint_path)
    print("Metrics CSV:", metrics_path)
    print("Training finished.")
    print(
        "Research note: historical context is capped at 100,000 sampled "
        "edges; validation and test use frozen context snapshots."
    )


if __name__ == "__main__":
    main()
