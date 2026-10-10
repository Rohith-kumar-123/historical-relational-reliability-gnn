
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Paid",
    "Amount Received",
    "Is Laundering",
    "Payment Currency",
    "Receiving Currency",
    "Payment Format"
]


def load_transactions(csv_path):
    df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="%Y/%m/%d %H:%M",
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Is Laundering",
        ]
    ).copy()

    df["Is Laundering"] = df["Is Laundering"].astype(int)
    df["day"] = df["Timestamp"].dt.floor("D")

    return df.sort_values("Timestamp").reset_index(drop=True)


def add_historical_persistence(df, lookback_days=7):
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    df = df.sort_values("Timestamp").reset_index(drop=True).copy()

    sources = list(zip(df["From Bank"], df["Account"]))
    destinations = list(zip(df["To Bank"], df["Account.1"]))

    history = defaultdict(set)
    persistence = np.zeros(len(df), dtype=np.float32)

    # Process one complete day at a time.
    for _, positions in df.groupby("day", sort=True).indices.items():
        day = df.iloc[positions[0]]["day"]
        window_start = day - pd.Timedelta(days=lookback_days)

        # Read past history before adding any transactions from today.
        for i in positions:
            pair = (sources[i], destinations[i])
            past_days = history[pair]

            recent_days = sum(
                window_start <= d < day for d in past_days
            )
            persistence[i] = recent_days / lookback_days

        # Update history only after every transaction for this day
        # has received its persistence feature.
        for i in positions:
            history[(sources[i], destinations[i])].add(day)

    df["persistence"] = persistence
    return df


def build_account_mapping(df):
    accounts = (
        list(zip(df["From Bank"], df["Account"]))
        + list(zip(df["To Bank"], df["Account.1"]))
    )
    unique_accounts = list(dict.fromkeys(accounts))

    return {
        account: idx
        for idx, account in enumerate(unique_accounts)
    }


def make_edge_index(df, account_to_id):
    sources = list(zip(df["From Bank"], df["Account"]))
    destinations = list(zip(df["To Bank"], df["Account.1"]))

    source_ids = np.fromiter(
        (account_to_id[a] for a in sources),
        dtype=np.int64,
        count=len(df),
    )
    destination_ids = np.fromiter(
        (account_to_id[a] for a in destinations),
        dtype=np.int64,
        count=len(df),
    )

    return torch.from_numpy(
        np.stack([source_ids, destination_ids])
    )

def add_transaction_features(df):
    """
    Create numerical features from AMLSim transaction records.

    These are transaction-level features, not account IDs.
    Categorical encodings are learned from the available dataset
    schema; temporal account-history features are handled separately.
    """
    import numpy as np

    result = df.copy()

    # Monetary values: reduce skew while preserving non-negative scale.
    result["amount_paid_log"] = np.log1p(
        pd.to_numeric(result["Amount Paid"], errors="coerce").clip(lower=0)
    )

    result["amount_received_log"] = np.log1p(
        pd.to_numeric(
            result["Amount Received"], errors="coerce"
        ).clip(lower=0)
    )

    paid = pd.to_numeric(result["Amount Paid"], errors="coerce")
    received = pd.to_numeric(result["Amount Received"], errors="coerce")

    # Relative discrepancy between the two reported amounts.
    denominator = paid.abs() + received.abs() + 1e-8
    result["amount_discrepancy"] = (
        (paid - received).abs() / denominator
    ).clip(0, 1)

    # Currency agreement is a categorical relationship signal.
    if {"Payment Currency", "Receiving Currency"}.issubset(result.columns):
        result["currency_match"] = (
            result["Payment Currency"].astype(str)
            == result["Receiving Currency"].astype(str)
        ).astype("float32")

    return result

def build_account_features(context_df, account_to_id):
    """
    Build account features using historical context transactions only.

    Features per account:
      1. log outgoing transaction count
      2. log incoming transaction count
      3. log outgoing amount total
      4. log incoming amount total
      5. log average outgoing amount
      6. log average incoming amount

    context_df must contain only transactions available before the
    prediction window. Never pass future or target transactions here.
    """
    num_accounts = len(account_to_id)

    sources = list(zip(context_df["From Bank"], context_df["Account"]))
    destinations = list(
        zip(context_df["To Bank"], context_df["Account.1"])
    )

    src = np.fromiter(
        (account_to_id[a] for a in sources),
        dtype=np.int64,
        count=len(context_df),
    )
    dst = np.fromiter(
        (account_to_id[a] for a in destinations),
        dtype=np.int64,
        count=len(context_df),
    )

    paid = pd.to_numeric(
        context_df["Amount Paid"], errors="coerce"
    ).fillna(0).clip(lower=0).to_numpy(dtype=np.float64)

    received = pd.to_numeric(
        context_df["Amount Received"], errors="coerce"
    ).fillna(0).clip(lower=0).to_numpy(dtype=np.float64)

    outgoing_count = np.bincount(src, minlength=num_accounts)
    incoming_count = np.bincount(dst, minlength=num_accounts)

    outgoing_total = np.bincount(
        src, weights=paid, minlength=num_accounts
    )
    incoming_total = np.bincount(
        dst, weights=received, minlength=num_accounts
    )

    outgoing_mean = outgoing_total / np.maximum(outgoing_count, 1)
    incoming_mean = incoming_total / np.maximum(incoming_count, 1)

    features = np.column_stack(
        [
            np.log1p(outgoing_count),
            np.log1p(incoming_count),
            np.log1p(outgoing_total),
            np.log1p(incoming_total),
            np.log1p(outgoing_mean),
            np.log1p(incoming_mean),
        ]
    ).astype(np.float32)

    return torch.from_numpy(features)

