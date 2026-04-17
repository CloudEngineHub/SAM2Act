#!/usr/bin/env python3

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


DEFAULT_REMOVE_KEYS = (
    "optimizer_state",
    "optimizer",
    "optim_state",
    "optimizer_states",
    "optim",
)


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_optimizer_state(
    input_path: Path,
    output_path: Path,
    remove_keys=DEFAULT_REMOVE_KEYS,
):
    checkpoint = load_checkpoint(input_path)

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected checkpoint at {input_path} to be a dict-like object, "
            f"got {type(checkpoint).__name__}."
        )

    cleaned = checkpoint.copy() if hasattr(checkpoint, "copy") else dict(checkpoint)
    removed_keys = [key for key in remove_keys if key in cleaned]

    for key in removed_keys:
        cleaned.pop(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cleaned, output_path)

    return removed_keys


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Remove optimizer state entries from a PyTorch checkpoint and save "
            "the stripped checkpoint to a new path."
        )
    )
    parser.add_argument("input_path", type=Path, help="Path to the input .pt checkpoint.")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path where the stripped checkpoint should be written.",
    )
    parser.add_argument(
        "--remove-key",
        action="append",
        default=None,
        help=(
            "Extra checkpoint key to remove. Can be passed multiple times. "
            "Defaults to the common optimizer keys."
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not input_path.is_file():
        parser.error(f"Input checkpoint does not exist: {input_path}")

    if input_path == output_path:
        parser.error("Input and output paths must be different.")

    remove_keys = tuple(dict.fromkeys(DEFAULT_REMOVE_KEYS + tuple(args.remove_key or ())))
    removed_keys = strip_optimizer_state(input_path, output_path, remove_keys=remove_keys)

    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    size_delta_mb = (input_size - output_size) / (1024 * 1024)

    if removed_keys:
        print(f"Removed keys: {', '.join(removed_keys)}")
    else:
        print("No optimizer keys were found. Saved a copy of the checkpoint unchanged.")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Size:   {input_size} -> {output_size} bytes ({size_delta_mb:.2f} MB smaller)")


if __name__ == "__main__":
    main()
