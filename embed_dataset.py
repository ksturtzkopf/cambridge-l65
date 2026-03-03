"""
Taken & adapted from SubgraphRAG (https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/emb.py).

Config loading is changed to use pythonic classes.
"""

import os
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.embeddings.config import CWQEmbeddingConfig, WebQSPEmbeddingConfig
from src.embeddings.dataset import EmbInferDataset
from src.models.text_encoders import GTELargeEN


def get_emb(subset, text_encoder, save_file):
    emb_dict = dict()
    for i in tqdm(range(len(subset))):
        id, q_text, text_entity_list, relation_list = subset[i]

        q_emb, entity_embs, relation_embs = text_encoder(
            q_text, text_entity_list, relation_list
        )
        emb_dict_i = {
            "q_emb": q_emb,
            "entity_embs": entity_embs,
            "relation_embs": relation_embs,
        }
        emb_dict[id] = emb_dict_i

    torch.save(emb_dict, save_file)


def load_config(dataset: str):
    if dataset == "cwq":
        return CWQEmbeddingConfig

    return WebQSPEmbeddingConfig


def main(args):
    config = load_config(args.dataset)

    torch.set_num_threads(config.num_threads)

    if args.dataset == "cwq":
        input_file = os.path.join("rmanluo", "RoG-cwq")
    else:
        input_file = os.path.join("ml1996", "webqsp")

    train_set = load_dataset(input_file, split="train")
    val_set = load_dataset(input_file, split="validation")
    test_set = load_dataset(input_file, split="test")

    entity_identifiers = []
    with open(config.entity_identifier_file, "r") as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)

    save_dir = f"data/{args.dataset}/processed"
    os.makedirs(save_dir, exist_ok=True)

    train_set = EmbInferDataset(
        train_set, entity_identifiers, os.path.join(save_dir, "train.pkl")
    )

    val_set = EmbInferDataset(
        val_set, entity_identifiers, os.path.join(save_dir, "val.pkl")
    )

    test_set = EmbInferDataset(
        test_set,
        entity_identifiers,
        os.path.join(save_dir, "test.pkl"),
        skip_no_topic=False,
        skip_no_ans=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    text_encoder = GTELargeEN(device)

    emb_save_dir = f"data/{args.dataset}/emb/gte-large-en-v1.5"
    os.makedirs(emb_save_dir, exist_ok=True)

    get_emb(train_set, text_encoder, os.path.join(emb_save_dir, "train.pth"))
    get_emb(val_set, text_encoder, os.path.join(emb_save_dir, "val.pth"))
    get_emb(test_set, text_encoder, os.path.join(emb_save_dir, "test.pth"))


if __name__ == "__main__":
    parser = ArgumentParser("Text Embedding Pre-Computation for Retrieval")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["webqsp", "cwq"],
        help="Dataset name",
    )
    args = parser.parse_args()

    main(args)
