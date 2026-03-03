from dataclasses import dataclass

from ..config import ASRConfig


@dataclass
class NoGNNConfig(ASRConfig):
    gnn_layers: int = 0
    ablation_name: str = "no-gnn"


@dataclass
class NoGatingConfig(ASRConfig):
    disable_gating: bool = True
    ablation_name: str = "no-gating"
