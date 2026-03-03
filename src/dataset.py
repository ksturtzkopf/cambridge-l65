import os
import pickle
from typing import Optional

import networkx as nx
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .asr_types import QuestionSample


def collate_single(batch: list[QuestionSample]) -> QuestionSample:
    return batch[0]


class ASRDataset(Dataset[QuestionSample]):
    def __init__(
        self,
        data_dir: str,
        split: str,
        text_encoder_name: str = "gte-large-en-v1.5",
    ) -> None:
        processed_path = os.path.join(data_dir, "processed", f"{split}.pkl")
        emb_path = os.path.join(data_dir, "emb", text_encoder_name, f"{split}.pth")

        with open(processed_path, "rb") as f:
            raw_samples: list[dict] = pickle.load(f)

        emb_dict: dict = torch.load(emb_path, weights_only=False)

        triple_scores = _compute_all_triple_scores(raw_samples, data_dir, split)

        self._samples: list[QuestionSample] = []
        for raw in raw_samples:
            sample_id = raw["id"]
            scores_entry = triple_scores[sample_id]

            num_text = len(raw["text_entity_list"])
            num_non_text = len(raw["non_text_entity_list"])
            num_entities = num_text + num_non_text

            h_ids = torch.tensor(raw["h_id_list"], dtype=torch.long)
            r_ids = torch.tensor(raw["r_id_list"], dtype=torch.long)
            t_ids = torch.tensor(raw["t_id_list"], dtype=torch.long)

            if sample_id in emb_dict:
                emb = emb_dict[sample_id]
                q_emb = emb["q_emb"]
                entity_embs = emb["entity_embs"]
                relation_embs = emb["relation_embs"]
            else:
                q_emb = torch.zeros(1, 1024)
                entity_embs = torch.zeros(num_text, 1024)
                num_relations = len(raw["relation_list"])
                relation_embs = torch.zeros(num_relations, 1024)

            self._samples.append(
                QuestionSample(
                    id=sample_id,
                    question=raw["question"],
                    h_ids=h_ids,
                    r_ids=r_ids,
                    t_ids=t_ids,
                    q_emb=q_emb,
                    entity_embs=entity_embs,
                    relation_embs=relation_embs,
                    num_entities=num_entities,
                    num_text_entities=num_text,
                    num_non_text_entities=num_non_text,
                    topic_entity_ids=list(raw["q_entity_id_list"]),
                    answer_entity_ids=list(set(raw["a_entity_id_list"])),
                    target_triple_scores=scores_entry["triple_scores"],
                    max_path_length=scores_entry["max_path_length"],
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> QuestionSample:
        return self._samples[idx]


def _compute_all_triple_scores(
    raw_samples: list[dict],
    data_dir: str,
    split: str,
) -> dict[str, dict]:
    """Compute or load cached triple supervision scores for all samples.

    Uses shortest paths between topic and answer entities as weak supervision.
    Triples on any shortest path get label 1.0, others get 0.0.
    """

    cache_dir = os.path.join(data_dir, "triple_scores")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}.pth")

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    result: dict[str, dict] = {}
    for raw in raw_samples:
        scores, max_path = _extract_paths_and_score(raw)
        result[raw["id"]] = {
            "triple_scores": scores,
            "max_path_length": max_path,
        }

    torch.save(result, cache_path)
    return result


def _extract_paths_and_score(
    sample: dict,
) -> tuple[Tensor, Optional[int]]:
    """Score triples by whether they lie on a shortest path from topic to answer."""

    h_list = sample["h_id_list"]
    r_list = sample["r_id_list"]
    t_list = sample["t_id_list"]
    num_triples = len(h_list)

    g = nx.DiGraph()
    for i in range(num_triples):
        g.add_edge(h_list[i], t_list[i], triple_id=i, relation_id=r_list[i])

    path_triples: list[list[list[int]]] = []
    for q_id in sample["q_entity_id_list"]:
        for a_id in sample["a_entity_id_list"]:
            sp = _all_shortest_paths_both_dirs(g, q_id, a_id)
            if sp:
                path_triples.extend(sp)

    if not path_triples:
        return torch.zeros(num_triples), None

    max_path_length = 0
    scored_triple_ids: set[int] = set()
    for path in path_triples:
        max_path_length = max(max_path_length, len(path))
        for triple_id_list in path:
            scored_triple_ids.update(triple_id_list)

    if max_path_length == 0:
        return torch.zeros(num_triples), None

    scores = torch.zeros(num_triples)
    for tid in scored_triple_ids:
        scores[tid] = 1.0

    return scores, max_path_length


def _all_shortest_paths_both_dirs(
    g: nx.DiGraph,
    source: int,
    target: int,
) -> list[list[list[int]]]:

    try:
        forward = list(nx.all_shortest_paths(g, source, target))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        forward = []

    try:
        backward = list(nx.all_shortest_paths(g, target, source))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        backward = []

    all_paths = forward + backward

    if not all_paths:
        return []

    if forward and backward:
        min_len = min(len(p) for p in all_paths)
        all_paths = [p for p in all_paths if len(p) == min_len]

    result: list[list[list[int]]] = []
    for path in all_paths:
        triple_path: list[list[int]] = []
        for i in range(len(path) - 1):
            edge_attrs: dict = g.edges[path[i], path[i + 1]]  # type: ignore[index]
            triple_path.append([int(edge_attrs["triple_id"])])
        result.append(triple_path)

    return result
