from collections import defaultdict, deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def select_anchor_triples(
    q_emb: Tensor,
    entity_embs: Tensor,
    relation_embs: Tensor,
    h_ids: Tensor,
    r_ids: Tensor,
    t_ids: Tensor,
    num_anchors: int = 64,
) -> tuple[Tensor, Tensor]:
    n_triples = h_ids.shape[0]
    effective_a = min(num_anchors, n_triples)

    head_embs = entity_embs[h_ids]
    rel_embs = relation_embs[r_ids]
    tail_embs = entity_embs[t_ids]
    triple_embs = (head_embs + rel_embs + tail_embs) / 3.0

    triple_embs_normed = F.normalize(triple_embs, p=2, dim=1)
    q_emb_normed = F.normalize(q_emb, p=2, dim=1)

    scores = (triple_embs_normed @ q_emb_normed.T).squeeze(1)

    anchor_scores, anchor_indices = torch.topk(scores, effective_a)
    return anchor_indices, anchor_scores


@dataclass
class AnchorSubgraph:
    entity_mask: Tensor
    triple_mask: Tensor
    local_to_global_entity: Tensor
    global_to_local_entity: Tensor
    local_edge_index: Tensor
    local_edge_relation_ids: Tensor
    num_local_entities: int
    num_local_edges: int


def build_anchor_subgraph(
    h_ids: Tensor,
    r_ids: Tensor,
    t_ids: Tensor,
    anchor_triple_indices: Tensor,
    topic_entity_ids: list[int],
    num_entities: int,
    k_hops: int = 1,
) -> AnchorSubgraph:

    h_list = h_ids.tolist()
    t_list = t_ids.tolist()
    r_list = r_ids.tolist()
    n_triples = len(h_list)

    adj: dict[int, set[int]] = defaultdict(set)
    for i in range(n_triples):
        h, t = h_list[i], t_list[i]
        adj[h].add(t)
        adj[t].add(h)

    seed_entities: set[int] = set(topic_entity_ids)
    for idx in anchor_triple_indices.tolist():
        seed_entities.add(h_list[idx])
        seed_entities.add(t_list[idx])

    expanded_entities = _bfs_expand(adj, seed_entities, k_hops)

    entity_mask = torch.zeros(num_entities, dtype=torch.bool)
    for e in expanded_entities:
        entity_mask[e] = True

    local_triple_indices: list[int] = []
    for i in range(n_triples):
        if h_list[i] in expanded_entities and t_list[i] in expanded_entities:
            local_triple_indices.append(i)

    triple_mask = torch.zeros(n_triples, dtype=torch.bool)
    for i in local_triple_indices:
        triple_mask[i] = True

    sorted_entities = sorted(expanded_entities)
    num_local = len(sorted_entities)
    local_to_global = torch.tensor(sorted_entities, dtype=torch.long)

    global_to_local = torch.full((num_entities,), -1, dtype=torch.long)
    for local_idx, global_idx in enumerate(sorted_entities):
        global_to_local[global_idx] = local_idx

    num_local_edges = len(local_triple_indices)
    if num_local_edges > 0:
        local_h = torch.empty(num_local_edges, dtype=torch.long)
        local_t = torch.empty(num_local_edges, dtype=torch.long)
        local_r = torch.empty(num_local_edges, dtype=torch.long)
        for j, tri_idx in enumerate(local_triple_indices):
            local_h[j] = global_to_local[h_list[tri_idx]]
            local_t[j] = global_to_local[t_list[tri_idx]]
            local_r[j] = r_list[tri_idx]
        local_edge_index = torch.stack([local_h, local_t], dim=0)
    else:
        local_edge_index = torch.zeros(2, 0, dtype=torch.long)
        local_r = torch.zeros(0, dtype=torch.long)

    return AnchorSubgraph(
        entity_mask=entity_mask,
        triple_mask=triple_mask,
        local_to_global_entity=local_to_global,
        global_to_local_entity=global_to_local,
        local_edge_index=local_edge_index,
        local_edge_relation_ids=local_r,
        num_local_entities=num_local,
        num_local_edges=num_local_edges,
    )


def _bfs_expand(
    adj: dict[int, set[int]],
    seeds: set[int],
    k_hops: int,
) -> set[int]:
    visited: set[int] = set(seeds)
    frontier: deque[tuple[int, int]] = deque()
    for s in seeds:
        frontier.append((s, 0))

    while frontier:
        node, depth = frontier.popleft()
        if depth >= k_hops:
            continue
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))

    return visited
