
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class ReliabilityWeightedConv(MessagePassing):
    """
    Reliability-weighted neighbor aggregation.

    Incoming edge reliabilities are normalized per destination node.
    Self-information is combined with aggregated neighbor information.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")

        self.self_lin = nn.Linear(in_channels, out_channels)
        self.neighbor_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_reliability):
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        if edge_reliability.ndim != 1:
            raise ValueError("edge_reliability must be one-dimensional")

        if edge_index.size(1) != edge_reliability.numel():
            raise ValueError(
                "Each edge must have exactly one reliability score"
            )

        # Always preserve the self-information path, even with no edges.
        if edge_index.size(1) == 0:
            aggregated = torch.zeros_like(x)
            return self.self_lin(x) + self.neighbor_lin(aggregated)

        source, destination = edge_index

        # Normalize scores across incoming edges of each destination.
        normalized_weight = softmax(
            edge_reliability,
            destination,
            num_nodes=x.size(0),
        )

        aggregated = self.propagate(
            edge_index,
            x=x,
            edge_weight=normalized_weight,
            size=(x.size(0), x.size(0)),
        )

        return self.self_lin(x) + self.neighbor_lin(aggregated)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j
