#!/usr/bin/env python3
"""
Run the same end-to-end pipeline as generate_1000_mlp_zoo.ipynb:
1) generate a 1000-model MLP checkpoint zoo
2) convert checkpoints to a graph zoo
3) print sanity checks and one graph preview
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_HIDDEN_ARCHS = [
    "32-32",
    "64-64",
    "128-64",
    "128-128",
    "256-128-64",
    "256-256-128-64",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Notebook-equivalent pipeline: model zoo generation + graph conversion."
    )

    # Model-zoo settings (from notebook CONFIG)
    parser.add_argument("--output_dir", type=str, default="model_zoo_poly_1000")
    parser.add_argument("--num_models", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--coeff_scale", type=float, default=1.5)
    parser.add_argument("--x_min", type=float, default=-2.0)
    parser.add_argument("--x_max", type=float, default=2.0)
    parser.add_argument("--train_size", type=int, default=2048)
    parser.add_argument("--val_size", type=int, default=512)
    parser.add_argument("--test_size", type=int, default=1024)
    parser.add_argument("--train_noise_std", type=float, default=0.03)
    parser.add_argument("--hidden_archs", nargs="+", default=DEFAULT_HIDDEN_ARCHS)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--optimizers", nargs="+", default=["adam", "adamw"])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--lr_max", type=float, default=3e-3)
    parser.add_argument("--weight_decay_min", type=float, default=1e-8)
    parser.add_argument("--weight_decay_max", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print_every", type=int, default=20)

    # Graph-zoo settings (from notebook GRAPH_CONFIG)
    parser.add_argument("--graph_output_subdir", type=str, default="graph_zoo")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include_layer_position_in_edge_attr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require_fixed_architecture", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--graph_print_every", type=int, default=25)

    # Optional stage controls
    parser.add_argument("--skip_model_generation", action="store_true")
    parser.add_argument("--skip_graph_generation", action="store_true")

    return parser.parse_args()


def run_step(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}\n")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_model_generation(args: argparse.Namespace, script_dir: Path) -> None:
    gen_script = script_dir / "generate_mlp_zoo.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"Missing script: {gen_script}")

    cmd = [
        sys.executable,
        str(gen_script),
        "--output_dir",
        args.output_dir,
        "--num_models",
        str(args.num_models),
        "--degree",
        str(args.degree),
        "--coeff_scale",
        str(args.coeff_scale),
        "--x_min",
        str(args.x_min),
        "--x_max",
        str(args.x_max),
        "--train_size",
        str(args.train_size),
        "--val_size",
        str(args.val_size),
        "--test_size",
        str(args.test_size),
        "--train_noise_std",
        str(args.train_noise_std),
        "--hidden_archs",
        *args.hidden_archs,
        "--activations",
        args.activation,
        "--optimizers",
        *args.optimizers,
        "--batch_sizes",
        *[str(x) for x in args.batch_sizes],
        "--lr_min",
        str(args.lr_min),
        "--lr_max",
        str(args.lr_max),
        "--weight_decay_min",
        str(args.weight_decay_min),
        "--weight_decay_max",
        str(args.weight_decay_max),
        "--max_epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--print_every",
        str(args.print_every),
    ]
    run_step(cmd, cwd=script_dir)


def run_graph_conversion(args: argparse.Namespace, script_dir: Path) -> None:
    graph_script = script_dir / "build_graph_zoo.py"
    if not graph_script.exists():
        raise FileNotFoundError(f"Missing script: {graph_script}")

    cmd = [
        sys.executable,
        str(graph_script),
        "--zoo_dir",
        args.output_dir,
        "--output_subdir",
        args.graph_output_subdir,
        "--val_ratio",
        str(args.val_ratio),
        "--test_ratio",
        str(args.test_ratio),
        "--split_seed",
        str(args.split_seed),
        "--print_every",
        str(args.graph_print_every),
    ]

    if not args.bidirectional:
        cmd.append("--no-bidirectional")
    if not args.include_layer_position_in_edge_attr:
        cmd.append("--no-include_layer_position_in_edge_attr")
    if args.require_fixed_architecture:
        cmd.append("--require_fixed_architecture")

    run_step(cmd, cwd=script_dir)


def print_model_sanity(output_dir: Path) -> None:
    models_dir = output_dir / "models"
    jsonl_path = output_dir / "zoo_index.jsonl"

    num_ckpts = len(list(models_dir.glob("*.pt")))
    print("Checkpoints:", num_ckpts)
    print("Index exists:", jsonl_path.exists())

    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line:
            first = json.loads(first_line)
            keys = ["model_id", "hidden_dims", "activation", "test_mse", "test_r2"]
            print({k: first.get(k) for k in keys})


def print_graph_preview(output_dir: Path, graph_output_subdir: str) -> None:
    graph_path = output_dir / graph_output_subdir / "graphs" / "graph_00000.pt"
    if not graph_path.exists():
        print(f"Graph preview skipped; file not found: {graph_path}")
        return

    graph = torch.load(graph_path, map_location="cpu")
    print("Graph file:", graph_path)
    print("x shape:", tuple(graph["x"].shape))
    print("edge_index shape:", tuple(graph["edge_index"].shape))
    print("edge_attr shape:", tuple(graph["edge_attr"].shape))
    print("architecture:", graph["architecture"])


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    if not args.skip_model_generation:
        run_model_generation(args, script_dir)
    else:
        print("Skipping model generation stage.")

    print_model_sanity(output_dir)

    if not args.skip_graph_generation:
        run_graph_conversion(args, script_dir)
        print_graph_preview(output_dir, args.graph_output_subdir)
    else:
        print("Skipping graph conversion stage.")


if __name__ == "__main__":
    main()
