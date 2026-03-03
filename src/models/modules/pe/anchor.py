import torch
import torch.nn as nn
from torch import Tensor


class AnchorProximityPE(nn.Module):
    def __init__(self, d_pe: int = 16, max_dist: int = 5) -> None:
        super().__init__()
        self.d_pe = d_pe
        self.max_dist = max_dist
        self.dist_embed = nn.Embedding(max_dist + 1, d_pe)

    def forward(
        self,
        h_ids: Tensor,
        t_ids: Tensor,
        anchor_triple_indices: Tensor,
        num_entities: int,
    ) -> Tensor:

        anchor_entities = _collect_anchor_entities(h_ids, t_ids, anchor_triple_indices)

        embed_device = self.dist_embed.weight.device

        if anchor_entities.shape[0] == 0:
            return self.dist_embed(
                torch.full(
                    (num_entities,),
                    self.max_dist,
                    dtype=torch.long,
                    device=embed_device,
                )
            )

        dist_matrix = _batch_bfs_distances(
            h_ids, t_ids, anchor_entities, num_entities, self.max_dist
        )

        embedded = self.dist_embed(dist_matrix.to(embed_device))
        return embedded.mean(dim=0)


def _collect_anchor_entities(
    h_ids: Tensor, t_ids: Tensor, anchor_triple_indices: Tensor
) -> Tensor:
    if anchor_triple_indices.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=h_ids.device)
    anchor_h = h_ids[anchor_triple_indices]
    anchor_t = t_ids[anchor_triple_indices]
    anchor_ents = torch.cat([anchor_h, anchor_t])
    return torch.unique(anchor_ents, sorted=True)


def _batch_bfs_distances(
    h_ids: Tensor,
    t_ids: Tensor,
    sources: Tensor,
    num_entities: int,
    max_dist: int,
) -> Tensor:

    dev = h_ids.device
    n_src = sources.shape[0]

    # Build undirected sparse adjacency (binary, float for spmm)
    edge_src = torch.cat([h_ids, t_ids])
    edge_dst = torch.cat([t_ids, h_ids])
    indices = torch.stack([edge_dst, edge_src])  # [2, 2*N_triples], row=dst, col=src
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=dev)
    adj = torch.sparse_coo_tensor(indices, values, (num_entities, num_entities))
    adj = adj.coalesce()

    # Binarise: any nonzero -> 1 (handles multi-edges)
    adj = torch.sparse_coo_tensor(
        adj.indices(), torch.ones_like(adj.values()), adj.shape
    )

    dist_matrix = torch.full(
        (n_src, num_entities), max_dist, dtype=torch.long, device=dev
    )

    frontier = torch.zeros(num_entities, n_src, dtype=torch.float32, device=dev)
    frontier[sources, torch.arange(n_src, device=dev)] = 1.0

    visited = torch.zeros(num_entities, n_src, dtype=torch.bool, device=dev)
    visited[sources, torch.arange(n_src, device=dev)] = True

    dist_matrix[torch.arange(n_src, device=dev), sources] = 0

    for depth in range(1, max_dist + 1):
        new_frontier = torch.sparse.mm(adj, frontier)  # [num_entities, n_src]
        # Mask out already-visited nodes
        newly_reached = (new_frontier > 0) & ~visited  # [num_entities, n_src]
        if not newly_reached.any():
            break

        src_indices, ent_indices = newly_reached.nonzero(as_tuple=True)

        # newly_reached is [num_entities, n_src], so dim0=entity, dim1=source
        ent_ids = src_indices  # entity IDs (dim 0)
        source_ids = ent_indices  # source indices (dim 1)
        dist_matrix[source_ids, ent_ids] = depth

        # Update visited and frontier
        visited = visited | newly_reached
        frontier = newly_reached.float()

    return dist_matrix
