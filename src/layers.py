
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class ReliabilityWeightedConv(MessagePassing):
    """
    Reliability-weighted graph convolution.

    Incoming messages are weighted directly by learned reliability
    scores and normalized by their total reliability mass.
    Self-information is preserved through a separate linear path.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add", flow="source_to_target")

        self.self_lin = nn.Linear(in_channels, out_channels)
        self.neighbor_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_reliability):
        if x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, num_features]")

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        if edge_reliability.ndim != 1:
            raise ValueError("edge_reliability must be one-dimensional")

        if edge_index.size(1) != edge_reliability.numel():
            raise ValueError(
                "Each edge must have exactly one reliability score"
            )

        num_nodes = x.size(0)

        if edge_index.size(1) == 0:
            aggregated = x.new_zeros((num_nodes, x.size(1)))
            return self.self_lin(x) + self.neighbor_lin(aggregated)

        if not torch.isfinite(edge_reliability).all():
            raise ValueError("edge_reliability must be finite")

        if (edge_reliability < 0).any():
            raise ValueError("edge_reliability must be nonnegative")

        source, destination = edge_index
        weights = edge_reliability.to(dtype=x.dtype)

        reliability_mass = x.new_zeros(num_nodes)
        reliability_mass.index_add_(0, destination, weights)

        weighted_messages = self.propagate(
            edge_index,
            x=x,
            edge_weight=weights,
            size=(num_nodes, num_nodes),
        )

        aggregated = weighted_messages / (
            reliability_mass.clamp_min(1e-8).unsqueeze(-1)
        )

        # Avoid numerical artifacts for nodes with no reliable neighbors.
        aggregated = torch.where(
            (reliability_mass > 0).unsqueeze(-1),
            aggregated,
            torch.zeros_like(aggregated),
        )

        return self.self_lin(x) + self.neighbor_lin(aggregated)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j
