"""
Effectively just the implementation from SubgraphRAG here!

https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/src/model/retriever.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing


class PEConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="mean")

    def forward(self, edge_index: Tensor, x: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class TopicDDE(nn.Module):
    def __init__(self, num_rounds: int = 2, num_reverse_rounds: int = 2) -> None:
        super().__init__()
        self.num_rounds = num_rounds
        self.num_reverse_rounds = num_reverse_rounds
        self.layers = nn.ModuleList([PEConv() for _ in range(num_rounds)])
        self.reverse_layers = nn.ModuleList(
            [PEConv() for _ in range(num_reverse_rounds)]
        )

        self.out_dim = 2 * (self.num_rounds + self.num_reverse_rounds)

    def forward(
        self,
        topic_entity_one_hot: Tensor,
        edge_index: Tensor,
        reverse_edge_index: Tensor,
    ) -> Tensor:
        parts: list[Tensor] = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            parts.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            parts.append(h_pe_rev)

        return torch.cat(parts, dim=1)


def build_topic_one_hot(topic_entity_ids: list[int], num_entities: int) -> Tensor:
    mask = torch.zeros(num_entities, dtype=torch.long)
    for tid in topic_entity_ids:
        mask[tid] = 1
    return F.one_hot(mask, num_classes=2).float()
