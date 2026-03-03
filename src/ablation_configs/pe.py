from dataclasses import dataclass

from ..config import ASRConfig


@dataclass
class NoTopicPEConfig(ASRConfig):
    disable_topic_pe: bool = True
    ablation_name: str = "no-topic-pe"


@dataclass
class NoAnchorPEConfig(ASRConfig):
    disable_anchor_pe: bool = True
    ablation_name: str = "no-anchor-pe"


@dataclass
class NoPathPEConfig(ASRConfig):
    disable_path_pe: bool = True
    ablation_name: str = "no-path-pe"


@dataclass
class NoAllPEConfig(ASRConfig):
    disable_topic_pe: bool = True
    disable_anchor_pe: bool = True
    disable_path_pe: bool = True
    ablation_name: str = "no-all-pe"
