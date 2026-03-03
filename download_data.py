from huggingface_hub import login, snapshot_download

if __name__ == "__main__":
    login()

    # Trained baseline model
    snapshot_download(
        repo_id="ksturtzkopf/l65-baseline",
        local_dir="./data",
    )

    # Precomputed WebQSP dataset
    snapshot_download(
        repo_id="ksturtzkopf/l65-know-precomputed",
        local_dir="./data",
        repo_type="dataset",
    )

    # Baseline retriever triples scored
    snapshot_download(
        repo_id="ksturtzkopf/l65-retrieved",
        local_dir="./data",
        repo_type="dataset",
    )
