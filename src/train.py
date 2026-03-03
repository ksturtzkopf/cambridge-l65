import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader

from .asr_types import QuestionSample
from .config import ASRConfig
from .dataset import ASRDataset, collate_single
from .eval import evaluate_on_test
from .metrics import evaluate_retriever
from .models.asr import AnchoredSubgraphRAG
from .utils import set_seed


class ASRLitModule(pl.LightningModule):
    def __init__(
        self,
        model: AnchoredSubgraphRAG,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        k_values: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.k_values = k_values or [50, 100, 200, 400]
        self._val_predictions: list[Tensor] = []
        self._val_samples: list[QuestionSample] = []

    def forward(self, sample: QuestionSample) -> Tensor:
        return self.model(sample)

    def training_step(self, batch: QuestionSample, _batch_idx: int) -> Tensor | None:
        sample = batch
        if len(sample.h_ids) == 0:
            return None

        if sample.target_triple_scores.sum().item() == 0:
            return None

        logits = self.model(sample)
        target = sample.target_triple_scores.to(logits.device).unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_predictions = []
        self._val_samples = []

    def validation_step(self, batch: QuestionSample, _batch_idx: int) -> None:
        sample = batch
        if len(sample.h_ids) == 0:
            return

        logits = self.model(sample).reshape(-1)
        self._val_predictions.append(logits.cpu())
        self._val_samples.append(sample)

    def on_validation_epoch_end(self) -> None:
        if not self._val_predictions:
            return

        metrics = evaluate_retriever(
            self._val_predictions,
            self._val_samples,
            k_values=self.k_values,
        )

        for key, value in metrics.items():
            if not (isinstance(value, float) and value != value):
                self.log(f"val_{key}", value, prog_bar=("@100" in key), batch_size=1)

        self._val_predictions = []
        self._val_samples = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def train(
    train_dataset: ASRDataset,
    val_dataset: ASRDataset,
    config: ASRConfig | None = None,
    max_epochs: int = 10000,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/asr",
) -> AnchoredSubgraphRAG:

    set_seed(seed)

    if config is None:
        config = ASRConfig()

    model = AnchoredSubgraphRAG(config)
    k_values = [50, 100, 200, 400]
    lit_module = ASRLitModule(
        model, lr=config.lr, weight_decay=config.weight_decay, k_values=k_values
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stop_callback = EarlyStopping(
        monitor="val_triple_recall@100",
        mode="max",
        patience=config.patience,
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="asr-{epoch:03d}-{val_triple_recall@100:.4f}",
        monitor="val_triple_recall@100",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    try:
        wandb_logger = WandbLogger(
            project="asr-retrieval",
            name=f"asr-{config.ablation_name}",
            config={
                "ablation_name": config.ablation_name,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "num_anchors": config.num_anchors,
                "gnn_layers": config.gnn_layers,
                "hidden_dim": config.hidden_dim,
                "rel_dim": config.rel_dim,
                "scorer_dropout": config.scorer_dropout,
                "gnn_dropout": config.gnn_dropout,
                "accumulate_grad_batches": config.accumulate_grad_batches,
                "subgraph_hops": config.subgraph_hops,
                "disable_gating": config.disable_gating,
                "disable_topic_pe": config.disable_topic_pe,
                "disable_anchor_pe": config.disable_anchor_pe,
                "disable_path_pe": config.disable_path_pe,
                "pe_total_dim": config.pe_total_dim,
                "seed": seed,
            },
        )
    except Exception:
        wandb_logger = None

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=50,
        default_root_dir=checkpoint_dir,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=wandb_logger,
    )

    trainer.fit(lit_module, train_loader, val_loader)

    best_path = checkpoint_callback.best_model_path
    if best_path:
        best_lit = ASRLitModule.load_from_checkpoint(
            best_path,
            model=model,
            lr=config.lr,
            weight_decay=config.weight_decay,
            k_values=k_values,
        )
        return best_lit.model

    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="webqsp",
        choices=["webqsp", "cwq"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    args = parser.parse_args()

    data_dir = f"./data/{args.dataset}"

    train_dataset = ASRDataset(data_dir, "train")
    val_dataset = ASRDataset(data_dir, "val")

    best_model = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=f"checkpoints/asr-{args.dataset}-seed{args.seed}",
        seed=args.seed,
    )

    test_set = ASRDataset(data_dir, "test")
    evaluate_on_test(best_model, test_set)
