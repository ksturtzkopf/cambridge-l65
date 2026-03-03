# Companion Repository to L65 Mini Project: Anchored SubgraphRAG

This project implements Anchored SubgraphRAG, an extension of [SubgraphRAG](https://github.com/Graph-COM/SubgraphRAG). Anchored SubgraphRAG implements triple retrieval using a gated message passing GNN to incorporate structural features into triple embeddings. 

## Prerequisites

- Python >= 3.12
- A CUDA-capable GPU (training and inference should be run on GPU)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

Please run `uv sync` first to install all required dependencies.

We have preprocessed data available to download on hugging face.
You can download the pre-processed data using `uv run python download_data.py`, but will need to be granted access to the HuggingFace repositories.
Please contact `kcs41@cam.ac.uk` with your hugging face username to be invited.

Otherwise, you can run the preprocessing from [https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/](https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/).

There are some utility methods in our repository to run this, but please refer to [https://github.com/Graph-COM/SubgraphRAG/tree/main/retrieve#stage-1-retrieval](https://github.com/Graph-COM/SubgraphRAG/tree/main/retrieve#stage-1-retrieval) for detailed instructions on how to retrieve and prepare the data.

After running their preparation scripts, please run `mv data_files data`, as our setup assumes all data to be stored in a `data` directory.

## Training

You can train our model by running `uv run python -m train`, which trains our model described in the report.

Checkpoints are saved to `checkpoints/asr-<dataset>-seed<seed>/` (default: `checkpoints/asr-webqsp-seed42/`). The best checkpoint by validation loss is kept automatically by PyTorch lightning.

## Ablations

You can run ablations by running `uv run python -m src.ablations --index=<ablationIndex>` note that if you do not supply an `ablationIndex`, all ablations are run in sequence.
The ablation index refers to the index in the array in `ablations.py`:

```py
ablation_configs = [
        NoTopicPEConfig(),
        NoAnchorPEConfig(),
        NoPathPEConfig(),
        NoAllPEConfig(),
        NoGatingConfig(),
        NoGNNConfig(),
]
```

## Inference

To get a file with inference results on the test set as well as results for `recall@k`, please run `uv run python -m src.inference <checkpoint>`, where `<checkpoint>` is the path to the `.ckpt` file from the training step, e.g.:

```bash
uv run python -m src.inference checkpoints/asr-webqsp-seed42/epoch=XX-step=XXXX.ckpt
```

