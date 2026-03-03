"""Merge per-seed recall_by_hops CSVs into one file and compute averages."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval_dir", type=str, help="Directory containing seed*/recall_by_hops.csv"
    )
    args = parser.parse_args()

    out_dir = Path(args.eval_dir)
    frames = []
    for csv_path in sorted(out_dir.glob("seed*/recall_by_hops.csv")):
        frames.append(pd.read_csv(csv_path))
        print(f"Loaded {csv_path} ({len(frames[-1])} rows)")

    if not frames:
        print("ERROR: No CSVs found to merge")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged_path = out_dir / "recall_by_hops_all_seeds.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged CSV to {merged_path} ({len(merged)} rows)")

    avg = (
        merged.groupby(["hops", "metric", "k", "n"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    avg["model"] = "ours-avg"
    avg_path = out_dir / "recall_by_hops_averaged.csv"
    avg.to_csv(avg_path, index=False)
    print(f"Saved averaged CSV to {avg_path}")


if __name__ == "__main__":
    main()
