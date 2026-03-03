import math

import torch
from torch import Tensor

from .asr_types import QuestionSample


def triple_recall_at_k(
    predicted_scores: Tensor,
    target_labels: Tensor,
    k: int,
) -> float:

    num_positives = target_labels.sum().item()
    if num_positives == 0:
        return float("nan")

    effective_k = min(k, predicted_scores.shape[0])
    _, top_indices = torch.topk(predicted_scores, effective_k)
    positives_in_top_k = target_labels[top_indices].sum().item()
    return positives_in_top_k / num_positives


def entity_recall_at_k(
    predicted_scores: Tensor,
    h_ids: Tensor,
    t_ids: Tensor,
    answer_entity_ids: list[int],
    k: int,
) -> float:

    answer_set = set(answer_entity_ids)
    if len(answer_set) == 0:
        return float("nan")

    effective_k = min(k, predicted_scores.shape[0])
    _, top_indices = torch.topk(predicted_scores, effective_k)

    top_h = h_ids[top_indices]
    top_t = t_ids[top_indices]
    retrieved_entities = set(top_h.tolist()) | set(top_t.tolist())

    found = len(answer_set & retrieved_entities)
    return found / len(answer_set)


def evaluate_retriever(
    predictions: list[Tensor],
    samples: list[QuestionSample],
    k_values: list[int] = [50, 100, 200, 400],
) -> dict[str, float]:

    accumulators: dict[str, list[float]] = {}
    for k in k_values:
        accumulators[f"triple_recall@{k}"] = []
        accumulators[f"entity_recall@{k}"] = []
        accumulators[f"triple_recall@{k}_1hop"] = []
        accumulators[f"triple_recall@{k}_multihop"] = []
        accumulators[f"entity_recall@{k}_1hop"] = []
        accumulators[f"entity_recall@{k}_multihop"] = []

    for pred, sample in zip(predictions, samples):
        is_1hop = sample.max_path_length is not None and sample.max_path_length == 1
        is_multihop = sample.max_path_length is not None and sample.max_path_length >= 2

        for k in k_values:
            tr = triple_recall_at_k(pred, sample.target_triple_scores, k)
            if not math.isnan(tr):
                accumulators[f"triple_recall@{k}"].append(tr)
                if is_1hop:
                    accumulators[f"triple_recall@{k}_1hop"].append(tr)
                elif is_multihop:
                    accumulators[f"triple_recall@{k}_multihop"].append(tr)

            er = entity_recall_at_k(
                pred, sample.h_ids, sample.t_ids, sample.answer_entity_ids, k
            )
            if not math.isnan(er):
                accumulators[f"entity_recall@{k}"].append(er)
                if is_1hop:
                    accumulators[f"entity_recall@{k}_1hop"].append(er)
                elif is_multihop:
                    accumulators[f"entity_recall@{k}_multihop"].append(er)

    results: dict[str, float] = {}
    for key, values in accumulators.items():
        if len(values) > 0:
            results[key] = sum(values) / len(values)
        else:
            results[key] = float("nan")

    return results
