"""Freeze the graph encoder output used by a trained MEDDC-DTI checkpoint."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MODEL_CONFIG  # noqa: E402
from main import build_inputs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Davis", choices=["Davis", "KIBA", "DrugBank"])
    parser.add_argument("--setting", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--source_data_dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    source_data_dir = Path(args.source_data_dir).resolve()
    os.environ["MEDDC_DTI_DATA_DIR"] = str(source_data_dir)
    runtime_args = SimpleNamespace(
        seed=args.seed,
        deterministic=1,
        deterministic_warn_only=1,
        dataset=args.dataset,
        dataset_dir=str(ROOT / "dataset"),
        data_seed=args.seed,
        setting=args.setting,
        split_seed=args.seed,
    )
    data, features, *_ = build_inputs(torch.device("cpu"), runtime_args, MODEL_CONFIG)
    output = Path(args.output) if args.output else ROOT / "data" / args.dataset.lower() / "node_features.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.cpu(),
            "dataset": args.dataset,
            "n_drugs": data["nb_drugs"],
            "n_proteins": data["nb_proteins"],
            "feature_dim": features.shape[1],
            "seed": args.seed,
        },
        output,
    )
    print(f"Saved {tuple(features.shape)} node features to {output}")


if __name__ == "__main__":
    main()
