import argparse
from pathlib import Path

import pandas as pd
import torch

from .config import ASRConfig
from .dataset import ASRDataset
from .metrics import evaluate_retriever
from .models.asr import AnchoredSubgraphRAG
from .train import ASRLitModule
from .asr_types import QuestionSample

K_VALUES = [50, 100, 200, 300, 400, 500]


def load_model(checkpoint_path: str, device: str = "cpu"):
    config = ASRConfig(device=device)
    model = AnchoredSubgraphRAG(config)
    lit = ASRLitModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        k_values=K_VALUES,
    )
    return lit.model


def run_inference(
    model: AnchoredSubgraphRAG,
    dataset: ASRDataset,
) -> tuple[list[torch.Tensor], list[QuestionSample]]:
    model.eval()
    predictions: list[torch.Tensor] = []
    samples: list[QuestionSample] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            if len(sample.h_ids) == 0:
                continue
            logits = model(sample).reshape(-1)
            predictions.append(logits.cpu())
            samples.append(sample)

    return predictions, samples


def metrics_to_recall_csv(
    metrics: dict[str, float],
    samples: list[QuestionSample],
    model_name,
) -> pd.DataFrame:
    n_all = len(samples)
    n_1hop = sum(
        1 for s in samples if s.max_path_length is not None and s.max_path_length == 1
    )
    n_2hop = sum(
        1 for s in samples if s.max_path_length is not None and s.max_path_length >= 2
    )

    rows = []
    for k in K_VALUES:
        for hops, suffix, n in [
            ("all", "", n_all),
            ("1-hop", "_1hop", n_1hop),
            ("2-hop", "_multihop", n_2hop),
        ]:
            for metric in ["ans_recall", "triple_recall"]:
                # ans_recall maps to entity_recall in the metrics dict
                metric_key = "entity_recall" if metric == "ans_recall" else metric
                key = f"{metric_key}@{k}{suffix}"
                value = metrics.get(key)
                rows.append(
                    {
                        "model": model_name,
                        "hops": hops,
                        "n": n,
                        "metric": metric,
                        "k": k,
                        "value": value if value is not None and value == value else "",
                    }
                )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("checkpoint", type=str, help="Path to .ckpt file")
    parser.add_argument("--model-name", type=str, default="ours")
    parser.add_argument("--out-dir", type=str, default="data/ours")
    parser.add_argument("--data-dir", type=str, default="./data/webqsp")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(args.checkpoint, device=device)

    print("Loading test dataset...")
    test_dataset = ASRDataset(args.data_dir, "test")

    print("Running inference...")
    predictions, samples = run_inference(model, test_dataset)

    preds_path = out_dir / "test_predictions.pt"
    torch.save(
        {"predictions": predictions, "sample_ids": [s.id for s in samples]},
        preds_path,
    )
    print(f"Saved raw predictions to {preds_path}")

    print("Computing metrics...")
    metrics = evaluate_retriever(predictions, samples, k_values=K_VALUES)

    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")

    csv_df = metrics_to_recall_csv(metrics, samples, model_name=args.model_name)
    csv_path = out_dir / "recall_by_hops.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved recall_by_hops.csv to {csv_path}")


if __name__ == "__main__":
    main()
