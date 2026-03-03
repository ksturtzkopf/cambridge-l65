import torch
import torch.nn as nn
from torch import Tensor

from ..asr_types import QuestionSample, build_subgraph_edges
from ..config import ASRConfig
from .modules.anchor import build_anchor_subgraph, select_anchor_triples
from .modules.gnn import GNNStack
from .modules.pe.anchor import AnchorProximityPE
from .modules.pe.path import compute_path_indicators
from .modules.pe.topic import TopicDDE, build_topic_one_hot
from .modules.scorer import (
    TripleScorer,
    assemble_entity_embeddings,
    compute_scalar_features,
)


class AnchoredSubgraphRAG(nn.Module):
    def __init__(self, config: ASRConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ASRConfig()
        self.config = config

        self.non_text_entity_emb = nn.Embedding(1, config.emb_dim)

        self.topic_dde = TopicDDE(
            num_rounds=config.dde_rounds,
            num_reverse_rounds=config.dde_reverse_rounds,
        )

        self.anchor_proximity_pe = AnchorProximityPE(
            d_pe=config.pe_anchor_dim,
            max_dist=config.pe_max_dist,
        )

        self.gnn: GNNStack | None

        if config.gnn_layers > 0:
            self.gnn = GNNStack(
                emb_dim=config.emb_dim,
                hidden_dim=config.hidden_dim,
                rel_dim=config.rel_dim,
                pe_dim=config.pe_total_dim,
                num_layers=config.gnn_layers,
                dropout=config.gnn_dropout,
                disable_gating=config.disable_gating,
            )
        else:
            self.gnn = None

        self.scorer = TripleScorer(
            emb_dim=config.emb_dim,
            num_scalar_feats=4,
            dropout=config.scorer_dropout,
        )

    def forward(self, sample: QuestionSample) -> Tensor:
        config = self.config
        device = next(self.parameters()).device

        h_ids = sample.h_ids.to(device)
        r_ids = sample.r_ids.to(device)
        t_ids = sample.t_ids.to(device)
        q_emb = sample.q_emb.to(device)
        entity_embs = sample.entity_embs.to(device)
        relation_embs = sample.relation_embs.to(device)

        # 1. Assemble full entity embeddings
        all_entity_embs = self._assemble_all_entity_embs(
            entity_embs, sample.num_non_text_entities, device
        )

        # 2. Anchor selection and subgraph construction
        anchor_indices, anchor_scores = select_anchor_triples(
            q_emb,
            all_entity_embs,
            relation_embs,
            h_ids,
            r_ids,
            t_ids,
            num_anchors=config.num_anchors,
        )

        anchor_subgraph = build_anchor_subgraph(
            h_ids,
            r_ids,
            t_ids,
            anchor_indices,
            sample.topic_entity_ids,
            sample.num_entities,
            k_hops=config.subgraph_hops,
        )

        # 3. Positional encodings
        edges = build_subgraph_edges(h_ids, r_ids, t_ids, sample.num_entities)
        num_entities = sample.num_entities

        pe_parts: list[Tensor] = []

        if not config.disable_topic_pe:
            topic_one_hot = build_topic_one_hot(
                sample.topic_entity_ids, num_entities
            ).to(device)
            pe_topic = self.topic_dde(
                topic_one_hot,
                edges.edge_index.to(device),
                edges.reverse_edge_index.to(device),
            )
            pe_parts.append(pe_topic)

        if not config.disable_anchor_pe:
            pe_anchor = self.anchor_proximity_pe(
                h_ids, t_ids, anchor_indices, num_entities
            ).to(device)
            pe_parts.append(pe_anchor)

        if not config.disable_path_pe:
            pe_path = compute_path_indicators(
                h_ids,
                t_ids,
                sample.topic_entity_ids,
                anchor_indices,
                num_entities,
                anchor_subgraph_entity_mask=anchor_subgraph.entity_mask,
            ).to(device)
            pe_parts.append(pe_path)

        if pe_parts:
            pe_full = torch.cat(pe_parts, dim=1)
        else:
            pe_full = torch.zeros(num_entities, 0, device=device)

        if self.gnn is not None and anchor_subgraph.num_local_entities > 0:
            local_entity_embs = all_entity_embs[anchor_subgraph.local_to_global_entity]
            local_pe = pe_full[anchor_subgraph.local_to_global_entity]

            local_edge_index = anchor_subgraph.local_edge_index.to(device)
            local_rel_ids = anchor_subgraph.local_edge_relation_ids.to(device)
            local_rel_embs = relation_embs[local_rel_ids]

            gnn_refined = self.gnn(
                local_entity_embs,
                local_pe,
                local_edge_index,
                local_rel_embs,
            )

            h_final = assemble_entity_embeddings(
                all_entity_embs,
                gnn_refined,
                anchor_subgraph.local_to_global_entity.to(device),
            )
        else:
            h_final = all_entity_embs

        # 5. Scalar features and scoring
        scalar_feats = compute_scalar_features(
            h_ids,
            t_ids,
            anchor_indices,
            anchor_scores.detach(),
            sample.topic_entity_ids,
            sample.num_entities,
        ).to(device)

        logits = self.scorer(
            q_emb, h_final, relation_embs, h_ids, r_ids, t_ids, scalar_feats
        )

        return logits

    def _assemble_all_entity_embs(
        self,
        text_entity_embs: Tensor,
        num_non_text: int,
        device: torch.device,
    ) -> Tensor:

        if num_non_text > 0:
            non_text_emb = self.non_text_entity_emb(
                torch.zeros(1, dtype=torch.long, device=device)
            ).expand(num_non_text, -1)

            return torch.cat([text_entity_embs, non_text_emb], dim=0)

        return text_entity_embs
