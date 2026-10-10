
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ReliabilityWeightedConv


class ReliabilityAwareGNN(nn.Module):
    """Historical-relational reliability-aware transaction classifier."""

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        dropout=0.2,
    ):
        super().__init__()

        self.encoder = nn.Linear(in_channels, hidden_channels)

        self.reliability_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.conv1 = ReliabilityWeightedConv(
            hidden_channels, hidden_channels
        )
        self.conv2 = ReliabilityWeightedConv(
            hidden_channels, hidden_channels
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels + 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

        self.dropout = dropout

    def encode_graph(
        self,
        x,
        context_edge_index,
        context_persistence,
    ):
        """Encode accounts once using the supplied historical graph."""
        h0 = F.relu(self.encoder(x))

        source, destination = context_edge_index

        consistency = F.cosine_similarity(
            h0[source],
            h0[destination],
            dim=-1,
            eps=1e-8,
        )

        signals = torch.stack(
            [
                context_persistence.to(dtype=h0.dtype),
                consistency,
            ],
            dim=-1,
        )

        reliability = torch.sigmoid(
            self.reliability_net(signals).squeeze(-1)
        )

        h = F.relu(
            self.conv1(h0, context_edge_index, reliability)
        )
        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )
        h = F.relu(
            self.conv2(h, context_edge_index, reliability)
        )

        return h0, h

    def score_edges(
        self,
        h0,
        h,
        target_edge_index,
        target_persistence,
    ):
        """Classify a batch of target transactions using encoded accounts."""
        source, destination = target_edge_index

        consistency = F.cosine_similarity(
            h0[source],
            h0[destination],
            dim=-1,
            eps=1e-8,
        )

        persistence = target_persistence.to(dtype=h.dtype)

        signals = torch.stack(
            [persistence, consistency],
            dim=-1,
        )

        reliability = torch.sigmoid(
            self.reliability_net(signals).squeeze(-1)
        )

        transaction_features = torch.cat(
            [
                h[source],
                h[destination],
                persistence.unsqueeze(-1),
                reliability.unsqueeze(-1),
            ],
            dim=-1,
        )

        logits = self.classifier(transaction_features)

        return logits, reliability, consistency

    def forward(
        self,
        x,
        context_edge_index,
        context_persistence,
        target_edge_index,
        target_persistence,
    ):
        """Compatible convenience interface for a single target batch."""
        h0, h = self.encode_graph(
            x,
            context_edge_index,
            context_persistence,
        )

        return self.score_edges(
            h0,
            h,
            target_edge_index,
            target_persistence,
        )
