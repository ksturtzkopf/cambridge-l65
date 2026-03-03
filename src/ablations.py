import argparse
import glob

import torch

from .ablation_configs.gnn import NoGatingConfig, NoGNNConfig
from .ablation_configs.pe import (
    NoAllPEConfig,
    NoAnchorPEConfig,
    NoPathPEConfig,
    NoTopicPEConfig,
)
from .config import ASRConfig
from .dataset import ASRDataset
from .eval import evaluate_on_test
from .models.asr import AnchoredSubgraphRAG
from .train import ASRLitModule, train


def run_ablation(
    config: ASRConfig,
    train_dataset: ASRDataset,
    val_dataset: ASRDataset,
    test_dataset: ASRDataset,
    seed: int = 42,
) -> None:

    best_ablated_model = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        checkpoint_dir=f"checkpoints/ablation-{config.ablation_name}",
        seed=seed,
    )

    evaluate_on_test(best_ablated_model, test_dataset)


def run_eval_only(
    config: ASRConfig,
    test_dataset: ASRDataset,
) -> dict[str, float]:
    checkpoint_dir = f"checkpoints/ablation-{config.ablation_name}"
    ckpt_files = sorted(glob.glob(f"{checkpoint_dir}/*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    ckpt_path = ckpt_files[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    model = AnchoredSubgraphRAG(config)
    lit_module = ASRLitModule.load_from_checkpoint(
        ckpt_path,
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if torch.cuda.is_available():
        lit_module = lit_module.cuda()

    return evaluate_on_test(lit_module.model, test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        help="index of the ablation to run, if none, all will run!",
        default=None,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training, load best checkpoint and run test evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )

    args = parser.parse_args()

    ablation_configs = [
        NoTopicPEConfig(),
        NoAnchorPEConfig(),
        NoPathPEConfig(),
        NoAllPEConfig(),
        NoGatingConfig(),
        NoGNNConfig(),
    ]

    if args.eval_only:
        test_dataset = ASRDataset("./data/webqsp", "test")

        if args.index is not None:
            config = ablation_configs[args.index]
            results = run_eval_only(config, test_dataset)
            print(results)
        else:
            for config in ablation_configs:
                print(f"\n--- {config.ablation_name} ---")
                results = run_eval_only(config, test_dataset)
                print(results)
    else:
        train_dataset = ASRDataset("./data/webqsp", "train")
        val_dataset = ASRDataset("./data/webqsp", "val")
        test_dataset = ASRDataset("./data/webqsp", "test")

        if args.index is not None:
            config = ablation_configs[args.index]
            run_ablation(
                config, train_dataset, val_dataset, test_dataset, seed=args.seed
            )
        else:
            for config in ablation_configs:
                run_ablation(
                    config, train_dataset, val_dataset, test_dataset, seed=args.seed
                )
