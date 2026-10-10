import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ReliabilityWeightedConv


class ReliabilityAwareGNN(nn.Module):
    """Reliability-aware GNN with configurable ablations."""

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        dropout=0.2,
        ablation_mode="full",
    ):
        super().__init__()

        if ablation_mode not in {"full", "A", "B"}:
            raise ValueError("ablation_mode must be full, A, or B")

        self.ablation_mode = ablation_mode
        self.encoder = nn.Linear(in_channels, hidden_channels)

        reliability_inputs = 1 if ablation_mode == "A" else 2
        self.reliability_net = nn.Sequential(
            nn.Linear(reliability_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.conv1 = ReliabilityWeightedConv(
            hidden_channels, hidden_channels
        )
        self.conv2 = ReliabilityWeightedConv(
            hidden_channels, hidden_channels
        )

        extra_features = 1 if ablation_mode in {"A", "B"} else 2
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels + extra_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

        self.dropout = dropout

    def encode_graph(self, x, context_edge_index, context_persistence):
        h0 = F.relu(self.encoder(x))
        source, destination = context_edge_index

        consistency = F.cosine_similarity(
            h0[source], h0[destination], dim=-1, eps=1e-8
        )

        if self.ablation_mode == "A":
            reliability = torch.sigmoid(
                self.reliability_net(consistency.unsqueeze(-1)).squeeze(-1)
            )
        elif self.ablation_mode == "B":
            reliability = torch.ones_like(context_persistence)
        else:
            signals = torch.stack(
                [context_persistence.to(h0.dtype), consistency], dim=-1
            )
            reliability = torch.sigmoid(
                self.reliability_net(signals).squeeze(-1)
            )

        h = F.relu(self.conv1(h0, context_edge_index, reliability))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, context_edge_index, reliability))

        return h0, h

    def score_edges(
        self, h0, h, target_edge_index, target_persistence
    ):
        source, destination = target_edge_index

        consistency = F.cosine_similarity(
            h0[source], h0[destination], dim=-1, eps=1e-8
        )
        persistence = target_persistence.to(dtype=h.dtype)

        if self.ablation_mode == "A":
            reliability = torch.sigmoid(
                self.reliability_net(consistency.unsqueeze(-1)).squeeze(-1)
            )
            features = torch.cat(
                [h[source], h[destination], reliability.unsqueeze(-1)],
                dim=-1,
            )
        elif self.ablation_mode == "B":
            reliability = torch.ones_like(persistence)
            features = torch.cat(
                [h[source], h[destination], persistence.unsqueeze(-1)],
                dim=-1,
            )
        else:
            signals = torch.stack([persistence, consistency], dim=-1)
            reliability = torch.sigmoid(
                self.reliability_net(signals).squeeze(-1)
            )
            features = torch.cat(
                [
                    h[source],
                    h[destination],
                    persistence.unsqueeze(-1),
                    reliability.unsqueeze(-1),
                ],
                dim=-1,
            )

        logits = self.classifier(features)
        return logits, reliability, consistency

    def forward(
        self,
        x,
        context_edge_index,
        context_persistence,
        target_edge_index,
        target_persistence,
    ):
        h0, h = self.encode_graph(
            x, context_edge_index, context_persistence
        )
        return self.score_edges(
            h0, h, target_edge_index, target_persistence
        )
