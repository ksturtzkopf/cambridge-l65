"""
Adapted from SubgraphRAG (https://github.com/Graph-COM/SubgraphRAG/tree/main/retrieve/configs/emb/gte-large-en-v1.5).
"""

from dataclasses import dataclass


@dataclass
class CWQEmbeddingConfig:
    num_threads: int = 16
    seed: int = 42

    entity_identifier_file: str = "data/cwq/entity_identifiers.txt"


@dataclass
class WebQSPEmbeddingConfig:
    num_threads: int = 16
    seed: int = 42

    entity_identifier_file: str = "data/webqsp/entity_identifiers.txt"
