from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class QuestionSample:
    id: str
    question: str
    h_ids: Tensor  # [N_triples] int64, head entity ID per triple
    r_ids: Tensor  # [N_triples] int64, relation ID per triple
    t_ids: Tensor  # [N_triples] int64, tail entity ID per triple
    q_emb: Tensor  # [1, 1024] float, question embedding
    entity_embs: Tensor  # [N_text_entities, 1024] float, text entity embeddings
    relation_embs: Tensor  # [N_relations, 1024] float, relation embeddings
    num_entities: int  # total entity count (text + non-text)
    num_text_entities: int
    num_non_text_entities: int
    topic_entity_ids: list[int]
    answer_entity_ids: list[int]
    target_triple_scores: Tensor  # [N_triples] float, 0.0 or 1.0
    max_path_length: Optional[int]  # max shortest path length (hop count)


@dataclass
class SubgraphEdges:
    edge_index: Tensor  # [2, N_edges] int64, PyG-style (source, target)
    reverse_edge_index: Tensor  # [2, N_edges] int64, reversed edges
    edge_relation_ids: Tensor  # [N_edges] int64, relation ID per edge
    num_nodes: int


def build_subgraph_edges(
    h_ids: Tensor,
    r_ids: Tensor,
    t_ids: Tensor,
    num_entities: int,
) -> SubgraphEdges:

    edge_index = torch.stack([h_ids, t_ids], dim=0)  # [2, N_triples]
    reverse_edge_index = torch.stack([t_ids, h_ids], dim=0)  # [2, N_triples]
    return SubgraphEdges(
        edge_index=edge_index,
        reverse_edge_index=reverse_edge_index,
        edge_relation_ids=r_ids.clone(),
        num_nodes=num_entities,
    )
