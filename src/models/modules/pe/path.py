from collections import defaultdict, deque

import torch
from torch import Tensor


def compute_path_indicators(
    h_ids: Tensor,
    t_ids: Tensor,
    topic_entity_ids: list[int],
    anchor_triple_indices: Tensor,
    num_entities: int,
    anchor_subgraph_entity_mask: Tensor | None = None,
) -> Tensor:

    h_list = h_ids.tolist()
    t_list = t_ids.tolist()
    n_triples = len(h_list)

    adj = _build_adjacency(h_list, t_list, n_triples)
    anchor_entities = _collect_anchor_entities(h_list, t_list, anchor_triple_indices)
    topic_set = set(topic_entity_ids)

    dev = h_ids.device
    out = torch.zeros(num_entities, 7, device=dev)

    # Col 0: shortest-path membership
    sp_entities = _shortest_path_membership(adj, topic_set, anchor_entities)
    if sp_entities:
        out[torch.tensor(sorted(sp_entities), dtype=torch.long, device=dev), 0] = 1.0

    # Col 1: topic neighbor (1-hop)
    topic_nbrs = _collect_neighbors(adj, topic_set)
    if topic_nbrs:
        out[torch.tensor(sorted(topic_nbrs), dtype=torch.long, device=dev), 1] = 1.0

    # Col 2: anchor neighbor (1-hop)
    anchor_nbrs = _collect_neighbors(adj, anchor_entities)
    if anchor_nbrs:
        out[torch.tensor(sorted(anchor_nbrs), dtype=torch.long, device=dev), 2] = 1.0

    # Cols 3-6: degree one-hot (4 bins)
    degree = _compute_degree(h_ids, t_ids, num_entities, anchor_subgraph_entity_mask)
    bins = _degree_to_bin(degree)
    out[torch.arange(num_entities, device=dev), 3 + bins] = 1.0

    return out


def _build_adjacency(
    h_list: list[int], t_list: list[int], n: int
) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for i in range(n):
        h, t = h_list[i], t_list[i]
        if h != t:
            adj[h].add(t)
            adj[t].add(h)
    return adj


def _collect_anchor_entities(
    h_list: list[int], t_list: list[int], anchor_triple_indices: Tensor
) -> set[int]:
    entities: set[int] = set()
    if anchor_triple_indices.numel() > 0:
        for idx in anchor_triple_indices.tolist():
            entities.add(h_list[idx])
            entities.add(t_list[idx])
    return entities


def _collect_neighbors(adj: dict[int, set[int]], sources: set[int]) -> set[int]:
    nbrs: set[int] = set()
    for s in sources:
        nbrs.update(adj.get(s, set()))
    return nbrs


def _shortest_path_membership(
    adj: dict[int, set[int]],
    topic_entities: set[int],
    anchor_entities: set[int],
) -> set[int]:
    if not topic_entities or not anchor_entities:
        return set()

    on_path: set[int] = set()

    topic_dists: dict[int, dict[int, int]] = {}
    for src in topic_entities:
        topic_dists[src] = _bfs_distances(adj, src)

    anchor_dists: dict[int, dict[int, int]] = {}
    for src in anchor_entities:
        if src not in anchor_dists:
            anchor_dists[src] = _bfs_distances(adj, src)

    for t_ent in topic_entities:
        t_dist = topic_dists[t_ent]
        for a_ent in anchor_entities:
            if t_ent == a_ent:
                on_path.add(t_ent)
                continue
            d_ta = t_dist.get(a_ent, -1)
            if d_ta < 0:
                continue
            a_dist = anchor_dists[a_ent]
            for v, d_tv in t_dist.items():
                d_va = a_dist.get(v, -1)
                if d_va >= 0 and d_tv + d_va == d_ta:
                    on_path.add(v)

    return on_path


def _bfs_distances(adj: dict[int, set[int]], source: int) -> dict[int, int]:
    dist: dict[int, int] = {source: 0}
    frontier = deque([source])
    while frontier:
        node = frontier.popleft()
        d = dist[node]
        for neighbor in adj.get(node, set()):
            if neighbor not in dist:
                dist[neighbor] = d + 1
                frontier.append(neighbor)
    return dist


def _compute_degree(
    h_ids: Tensor,
    t_ids: Tensor,
    num_entities: int,
    mask: Tensor | None,
) -> Tensor:
    dev = h_ids.device
    degree = torch.zeros(num_entities, dtype=torch.long, device=dev)
    if h_ids.numel() == 0:
        return degree
    if mask is not None:
        mask = mask.to(dev)
        valid = mask[h_ids] & mask[t_ids]
        h_valid = h_ids[valid]
        t_valid = t_ids[valid]
        if h_valid.numel() > 0:
            ones = torch.ones(h_valid.shape[0], dtype=torch.long, device=dev)
            degree.scatter_add_(0, h_valid, ones)
            degree.scatter_add_(0, t_valid, ones)
    else:
        ones = torch.ones(h_ids.shape[0], dtype=torch.long, device=dev)
        degree.scatter_add_(0, h_ids, ones)
        degree.scatter_add_(0, t_ids, ones)
    return degree


def _degree_to_bin(degree: Tensor) -> Tensor:
    bins = torch.zeros_like(degree)
    bins[degree >= 2] = 1
    bins[degree >= 4] = 2
    bins[degree >= 8] = 3
    return bins
