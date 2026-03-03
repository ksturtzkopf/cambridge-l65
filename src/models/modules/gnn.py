import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


class RelationalGatedConv(MessagePassing):
    def __init__(
        self,
        hidden_dim: int = 256,
        rel_dim: int = 128,
        pe_dim: int = 31,
        disable_gating: bool = False,
    ) -> None:
        super().__init__(aggr="mean")
        self.hidden_dim = hidden_dim
        self.rel_dim = rel_dim
        self.pe_dim = pe_dim
        self.disable_gating = disable_gating or pe_dim == 0

        self.w_msg = nn.Linear(hidden_dim + rel_dim, hidden_dim)

        if not self.disable_gating:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * pe_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self.w_upd = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        pe: Tensor,
    ) -> Tensor:
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, pe=pe)
        out = torch.relu(self.w_upd(torch.cat([x, agg], dim=-1)))
        return out

    def message(
        self,
        x_j: Tensor,
        edge_attr: Tensor,
        pe_j: Tensor,
        pe_i: Tensor,
    ) -> Tensor:
        msg = torch.relu(self.w_msg(torch.cat([x_j, edge_attr], dim=-1)))
        if self.disable_gating:
            return msg

        gate = torch.sigmoid(self.gate_mlp(torch.cat([pe_j, pe_i], dim=-1)))
        return gate * msg


class GNNStack(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        hidden_dim: int = 256,
        rel_dim: int = 128,
        pe_dim: int = 31,
        num_layers: int = 2,
        dropout: float = 0.1,
        disable_gating: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.rel_dim = rel_dim
        self.pe_dim = pe_dim
        self.num_layers = num_layers

        self.project_in = nn.Linear(emb_dim + pe_dim, hidden_dim)
        self.rel_project = nn.Linear(emb_dim, rel_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RelationalGatedConv(hidden_dim, rel_dim, pe_dim, disable_gating)
            )

        self.project_out = nn.Linear(hidden_dim, emb_dim)

    def forward(
        self,
        entity_embs: Tensor,
        pe: Tensor,
        edge_index: Tensor,
        relation_embs_per_edge: Tensor,
    ) -> Tensor:
        edge_attr = self.rel_project(relation_embs_per_edge)

        x = torch.relu(self.project_in(torch.cat([entity_embs, pe], dim=-1)))
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, pe)
            x = self.dropout(x)

        return self.project_out(x)
