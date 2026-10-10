
from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GCNConv

from .data import (
    load_transactions,
    add_historical_persistence,
    build_account_mapping,
    make_edge_index,
    build_account_features,
)
from .train_pipeline import (
    chronological_split,
    build_context,
    get_data_path,
    get_output_dir,
    set_seed,
    get_device,
    calculate_metrics,
    TARGET_BATCH_SIZE,
    LOOKBACK_DAYS,
)


SEED = 42
HIDDEN_CHANNELS = 32
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 5


class BasicGCN(nn.Module):
    """Standard two-layer GCN for transaction classification."""

    def __init__(self, in_channels=6, hidden_channels=32, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def encode_graph(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return h

    def score_edges(self, h, target_edge_index):
        source, destination = target_edge_index
        edge_features = torch.cat([h[source], h[destination]], dim=-1)
        return self.classifier(edge_features)


@torch.no_grad()
def evaluate_gcn(model, x, edge_index, targets, account_to_id, device):
    if targets.empty:
        return calculate_metrics([], [])

    model.eval()
    h = model.encode_graph(x, edge_index)

    probabilities = []
    labels = []

    for start in range(0, len(targets), TARGET_BATCH_SIZE):
        batch = targets.iloc[start:start + TARGET_BATCH_SIZE]
        target_edges = make_edge_index(batch, account_to_id).to(device)
        logits = model.score_edges(h, target_edges)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        probabilities.extend(probs.cpu().tolist())
        labels.extend(batch["Is Laundering"].astype(int).tolist())

    return calculate_metrics(labels, probabilities)


def main():
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = get_device()

    csv_path = get_data_path()
    output_dir = get_output_dir()
    metrics_path = output_dir / "gcn_baseline_metrics.csv"
    checkpoint_path = output_dir / "gcn_baseline.pt"

    print("Baseline: standard GCN")
    print("Dataset:", csv_path)

    df = load_transactions(csv_path)
    df = add_historical_persistence(df, lookback_days=LOOKBACK_DAYS)

    train_df, val_df, test_df = chronological_split(df)

    for name, part in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(part):,} rows | "
            f"positives: {int(part['Is Laundering'].sum()):,}"
        )

    account_to_id = build_account_mapping(df)

    model = BasicGCN(
        in_channels=6,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    labels = torch.tensor(
        train_df["Is Laundering"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    counts = torch.bincount(labels, minlength=2).float()
    class_weights = counts.sum() / (2 * counts.clamp_min(1))

    best_val_ap = -1.0
    best_state = None

    train_df = train_df.copy()
    train_df["day"] = train_df["Timestamp"].dt.floor("D")
    train_days = sorted(train_df["day"].unique())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rows = 0

        for day in train_days:
            targets = train_df[train_df["day"] == day]
            context = train_df[train_df["Timestamp"] < day]

            x, edge_index, _, context_size = build_context(
                context, account_to_id, device
            )

            target_edges = make_edge_index(targets, account_to_id).to(device)
            target_labels = torch.tensor(
                targets["Is Laundering"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(x, edge_index, target_edges) if False else (
                model.score_edges(
                    model.encode_graph(x, edge_index), target_edges
                )
            )

            loss = F.cross_entropy(
                logits, target_labels, weight=class_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item() * len(targets)
            epoch_rows += len(targets)

            del x, edge_index

        # Validation uses training transactions only as context.
        x_val, edge_val, _, _ = build_context(
            train_df, account_to_id, device
        )
        val_metrics = evaluate_gcn(
            model, x_val, edge_val, val_df, account_to_id, device
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={epoch_loss / max(epoch_rows, 1):.6f} | "
            f"validation AP={val_metrics['average_precision']:.6f}"
        )

        val_ap = val_metrics["average_precision"]
        if best_state is None or (
            np.isfinite(val_ap) and val_ap > best_val_ap
        ):
            if np.isfinite(val_ap):
                best_val_ap = val_ap
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        del x_val, edge_val

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)

    # Final test context matches the proposed model's frozen snapshot.
    test_context = pd.concat(
        [train_df, val_df], ignore_index=True
    ).sort_values("Timestamp")

    x_test, edge_test, _, context_size = build_context(
        test_context, account_to_id, device
    )
    test_metrics = evaluate_gcn(
        model, x_test, edge_test, test_df, account_to_id, device
    )

    print("\nGCN baseline final test metrics:")
    print(test_metrics)

    pd.DataFrame([{
        "model": "GCN",
        "seed": SEED,
        "best_validation_ap": best_val_ap,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "test_context_edges": context_size,
    }]).to_csv(metrics_path, index=False)

    print("Saved checkpoint:", checkpoint_path)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
