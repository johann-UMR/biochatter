from __future__ import annotations

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path

from benchmark.medication_safety.scripts.conservative_term_matching import (
    MedicationSafetyScorer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "benchmark_medication_safety_data.yaml"
DEFAULT_REPLACEMENTS = ROOT / "results" / "term_matching" / "normalization_replacements.csv"
DEFAULT_SYNONYMS = ROOT / "results" / "term_matching" / "canonical_synonym_groups.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the conservative medication safety term matcher to response text.",
    )
    parser.add_argument("input", type=Path, help="CSV with case_id and response columns.")
    parser.add_argument("--output", type=Path, help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--replacements", type=Path, default=DEFAULT_REPLACEMENTS)
    parser.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input CSV is empty.")
    required = {"case_id", "response"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    scorer = MedicationSafetyScorer.from_files(
        args.data,
        args.replacements,
        args.synonyms,
    )
    scored_rows = [
        {**row, **scorer.score(row["case_id"], row["response"])}
        for row in rows
    ]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_context = args.output.open("w", encoding="utf-8", newline="")
    else:
        output_context = nullcontext(sys.stdout)
    with output_context as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)

    if args.output:
        print(f"Wrote {len(scored_rows)} scored responses to {args.output}")


if __name__ == "__main__":
    main()
