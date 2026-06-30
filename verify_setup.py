"""Verify local technical prerequisites for the ranking pipeline."""

import gzip
import json
import os
import sys
from pathlib import Path

from rank_candidates import evaluate_stage1_boolean


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
MODEL_WEIGHTS = MODEL_DIR / "model.safetensors"
FULL_CANDIDATES = ROOT / "data" / "candidates.jsonl"
SAMPLE_CANDIDATES = ROOT / "data" / "sample_candidates.jsonl"
SEMANTIC_SCORES = ROOT / "data" / "semantic_scores.json.gz"
SUBMISSION = ROOT / "team_204.csv"


def count_survivors(path):
    passed = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if evaluate_stage1_boolean(json.loads(line)):
                passed += 1
    return total, passed


def check_dependencies():
    required = {
        "numpy": "2.2.3",
        "lightgbm": "4.6.0",
        "torch": "2.7.1+cpu",
        "sentence_transformers": "5.6.0"
    }
    ok = True
    checks = []
    errors = []
    for pkg, exp_ver in required.items():
        try:
            mod = __import__(pkg)
            # handle subpackage naming conventions if any
            if pkg == "sentence_transformers":
                import sentence_transformers as st
                ver = st.__version__
            else:
                ver = getattr(mod, "__version__", "unknown")
            checks.append(f"Library {pkg} is installed (version={ver}, expected={exp_ver})")
        except ImportError:
            ok = False
            errors.append(f"Missing dependency: {pkg}. Please run: pip install {pkg}=={exp_ver}")
    return ok, checks, errors


def check_model():
    if not MODEL_DIR.is_dir():
        return False, f"Missing model directory: {MODEL_DIR}"
    if not MODEL_WEIGHTS.is_file() or MODEL_WEIGHTS.stat().st_size < 1_000_000:
        return False, (
            f"Missing or incomplete model weights at {MODEL_WEIGHTS}. "
            "Run: python prepare_embedding_model.py --out models/all-MiniLM-L6-v2"
        )
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(MODEL_DIR))
        dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
        return True, f"Embedding model loads successfully (dimension={dim})."
    except Exception as exc:
        return False, f"Embedding model failed to load: {exc}"


def check_semantic_scores(expected_min=679):
    if not SEMANTIC_SCORES.is_file():
        return False, (
            f"Missing {SEMANTIC_SCORES}. "
            "Run: python generate_precomputed_scores.py"
        )
    with gzip.open(SEMANTIC_SCORES, "rt", encoding="utf-8") as f:
        scores = json.load(f)
    if len(scores) < expected_min:
        return False, f"Only {len(scores)} precomputed semantic scores found; expected at least {expected_min}."
    return True, f"Precomputed semantic scores present ({len(scores)} entries)."


def main():
    errors = []
    checks = []

    # Check dependencies first
    dep_ok, dep_checks, dep_errors = check_dependencies()
    checks.extend(dep_checks)
    errors.extend(dep_errors)

    for path, label in (
        (FULL_CANDIDATES, "full candidate dataset"),
        (SAMPLE_CANDIDATES, "sample candidate dataset"),
    ):
        if not path.is_file():
            errors.append(f"Missing {label}: {path}")
            continue
        total, passed = count_survivors(path)
        checks.append(f"{label}: {total} records, {passed} Stage 1 survivors")
        if path == SAMPLE_CANDIDATES and passed != 100:
            errors.append(
                f"Sample dataset must have exactly 100 Stage 1 survivors for validator demos; found {passed}. "
                "Run: python build_sample_dataset.py"
            )
        if path == FULL_CANDIDATES and passed < 100:
            errors.append(f"Full dataset has only {passed} Stage 1 survivors; expected at least 100.")

    ok, message = check_model()
    checks.append(message)
    if not ok:
        errors.append(message)

    ok, message = check_semantic_scores(expected_min=679)
    checks.append(message)
    if not ok:
        errors.append(message)

    if SUBMISSION.is_file():
        checks.append(f"Submission file present: {SUBMISSION.name}")
    else:
        errors.append(f"Missing submission file: {SUBMISSION}")

    print("Redrob Hybrid Ranker setup verification")
    print("=" * 50)
    for line in checks:
        print(f"  [OK] {line}")

    if errors:
        print("\nIssues found:")
        for issue in errors:
            print(f"  [FAIL] {issue}")
        return 1

    print("\nAll technical prerequisites are satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
