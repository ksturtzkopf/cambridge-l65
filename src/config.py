from dataclasses import dataclass


@dataclass
class ASRConfig:
    emb_dim: int = 1024
    hidden_dim: int = 256
    rel_dim: int = 128
    pe_anchor_dim: int = 16
    pe_max_dist: int = 5
    num_anchors: int = 64
    subgraph_hops: int = 1
    gnn_layers: int = 2
    dde_rounds: int = 2
    dde_reverse_rounds: int = 2
    scorer_dropout: float = 0.1
    gnn_dropout: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 1e-2
    accumulate_grad_batches: int = 16
    patience: int = 10
    device: str = "cpu"
    disable_gating: bool = False
    disable_topic_pe: bool = False
    disable_anchor_pe: bool = False
    disable_path_pe: bool = False
    ablation_name: str = "baseline"

    @property
    def pe_topic_dim(self) -> int:
        return 2 * (self.dde_rounds + self.dde_reverse_rounds)

    @property
    def pe_path_dim(self) -> int:
        return 7

    @property
    def pe_total_dim(self) -> int:
        total = 0
        if not self.disable_topic_pe:
            total += self.pe_topic_dim
        if not self.disable_anchor_pe:
            total += self.pe_anchor_dim
        if not self.disable_path_pe:
            total += self.pe_path_dim
        return total
