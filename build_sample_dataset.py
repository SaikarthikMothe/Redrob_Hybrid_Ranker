"""Rebuild data/sample_candidates.jsonl for sandbox and validator demos."""

import argparse
import json
import random

from rank_candidates import evaluate_stage1_boolean


def build_sample_dataset(input_path, output_path, total=2969, survivor_target=100, seed=204):
    survivors = []
    non_survivors = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if evaluate_stage1_boolean(record):
                survivors.append(record)
            else:
                non_survivors.append(record)

    if len(survivors) < survivor_target:
        raise SystemExit(
            f"Need at least {survivor_target} Stage 1 survivors; found {len(survivors)} in {input_path}."
        )

    filler_target = total - survivor_target
    if len(non_survivors) < filler_target:
        raise SystemExit(
            f"Need at least {filler_target} non-survivors; found {len(non_survivors)} in {input_path}."
        )

    rng = random.Random(seed)
    chosen_survivors = sorted(survivors, key=lambda r: r.get("candidate_id", ""))[:survivor_target]
    chosen_filler = rng.sample(non_survivors, filler_target)
    sample = chosen_survivors + chosen_filler
    rng.shuffle(sample)

    passed = sum(1 for record in sample if evaluate_stage1_boolean(record))
    if passed != survivor_target:
        raise SystemExit(
            f"Sample build sanity check failed: expected {survivor_target} survivors, got {passed}."
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in sample:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(sample)} candidates to {output_path} "
        f"({passed} Stage 1 survivors, {len(sample) - passed} filtered out at runtime)."
    )


def main():
    parser = argparse.ArgumentParser(description="Build a sample candidate dataset for demos.")
    parser.add_argument("--input", default="data/candidates.jsonl")
    parser.add_argument("--output", default="data/sample_candidates.jsonl")
    parser.add_argument("--total", type=int, default=2969)
    parser.add_argument("--survivors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=204)
    args = parser.parse_args()

    build_sample_dataset(
        args.input,
        args.output,
        total=args.total,
        survivor_target=args.survivors,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
