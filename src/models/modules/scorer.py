import torch
import torch.nn as nn
from torch import Tensor


class TripleScorer(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        num_scalar_feats: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = 4 * emb_dim + num_scalar_feats
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(
        self,
        q_emb: Tensor,
        h_final: Tensor,
        relation_embs: Tensor,
        h_ids: Tensor,
        r_ids: Tensor,
        t_ids: Tensor,
        scalar_feats: Tensor,
    ) -> Tensor:
        n_triples = h_ids.shape[0]
        h_triple = torch.cat(
            [
                q_emb.expand(n_triples, -1),
                h_final[h_ids],
                relation_embs[r_ids],
                h_final[t_ids],
                scalar_feats,
            ],
            dim=1,
        )
        return self.mlp(h_triple)


def assemble_entity_embeddings(
    original_embs: Tensor,
    gnn_refined_embs: Tensor,
    local_to_global: Tensor,
) -> Tensor:
    h_final = original_embs.clone()
    h_final[local_to_global] = gnn_refined_embs
    return h_final


def compute_scalar_features(
    h_ids: Tensor,
    t_ids: Tensor,
    anchor_triple_indices: Tensor,
    anchor_scores: Tensor,
    topic_entity_ids: list[int],
    num_entities: int,
) -> Tensor:
    device = h_ids.device
    max_dist = 5

    entity_anchor_scores_h = torch.zeros(num_entities, device=device)
    entity_anchor_scores_t = torch.zeros(num_entities, device=device)

    if anchor_triple_indices.numel() > 0:
        anchor_h = h_ids[anchor_triple_indices]
        anchor_t = t_ids[anchor_triple_indices]
        entity_anchor_scores_h = entity_anchor_scores_h.scatter_reduce(
            0, anchor_h, anchor_scores, reduce="amax", include_self=True
        )
        entity_anchor_scores_t = entity_anchor_scores_t.scatter_reduce(
            0, anchor_t, anchor_scores, reduce="amax", include_self=True
        )

    entity_anchor_scores = torch.maximum(entity_anchor_scores_h, entity_anchor_scores_t)
    anchor_score_h = entity_anchor_scores[h_ids]
    anchor_score_t = entity_anchor_scores[t_ids]

    topic_dists = _compute_topic_distances(
        h_ids, t_ids, topic_entity_ids, num_entities, max_dist
    )
    topic_dist_h = topic_dists[h_ids].float() / max_dist
    topic_dist_t = topic_dists[t_ids].float() / max_dist

    return torch.stack(
        [anchor_score_h, anchor_score_t, topic_dist_h, topic_dist_t], dim=1
    )


def _compute_topic_distances(
    h_ids: Tensor,
    t_ids: Tensor,
    topic_entity_ids: list[int],
    num_entities: int,
    max_dist: int,
) -> Tensor:
    device = h_ids.device

    if not topic_entity_ids:
        return torch.full((num_entities,), max_dist, dtype=torch.long, device=device)

    edges = torch.cat(
        [
            torch.stack([h_ids, t_ids]),
            torch.stack([t_ids, h_ids]),
        ],
        dim=1,
    )
    ones = torch.ones(edges.shape[1], device=device)
    adj = torch.sparse_coo_tensor(edges, ones, (num_entities, num_entities)).coalesce()

    dist = torch.full((num_entities,), max_dist, dtype=torch.long, device=device)
    reached = torch.zeros(num_entities, dtype=torch.bool, device=device)
    frontier = torch.zeros(num_entities, dtype=torch.float, device=device)

    for tid in topic_entity_ids:
        frontier[tid] = 1.0
        dist[tid] = 0
        reached[tid] = True

    for d in range(1, max_dist + 1):
        frontier = (torch.sparse.mm(adj, frontier.unsqueeze(1)).squeeze(1) > 0).float()
        newly_reached = (frontier > 0) & ~reached
        dist[newly_reached] = d
        reached |= frontier > 0
        frontier = newly_reached.float()

    return dist
