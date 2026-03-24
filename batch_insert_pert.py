from __future__ import annotations

import argparse
import time
from pathlib import Path

from insert_pert import generate_perturbed_structures, insert_random_cr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "random_cr_single_test" / "perturb_abacustest.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "batch_1500_results")
    parser.add_argument("--repeat", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefix", default="STRU_")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for i in range(1, args.repeat + 1):
        case_dir = output_dir / f"case_{i:04d}"
        insert_dir = case_dir / "insert"
        pert_dir = case_dir / "pert"
        case_seed = args.seed + i
        selected = insert_random_cr(input_dir, insert_dir, case_seed)
        outputs = generate_perturbed_structures(selected, config, pert_dir, case_seed, args.prefix)
        print(
            f"\rprogress {i}/{args.repeat} selected={selected.name} perturbed={len(outputs)} case={case_dir.name}",
            end="",
            flush=True,
        )

    elapsed = time.time() - started
    print()
    print(f"done repeat={args.repeat} output={output_dir} elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
