import torch
from torch import Tensor

from .dataset import ASRDataset
from .metrics import evaluate_retriever
from .models.asr import AnchoredSubgraphRAG
from .asr_types import QuestionSample


def evaluate_on_test(
    model: AnchoredSubgraphRAG,
    test_dataset: ASRDataset,
) -> dict[str, float]:

    model.eval()
    predictions: list[Tensor] = []
    samples: list[QuestionSample] = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            if len(sample.h_ids) == 0:
                continue
            logits = model(sample).reshape(-1).cpu()
            predictions.append(logits)
            samples.append(sample)

    return evaluate_retriever(predictions, samples, k_values=[50, 100, 200, 400])
