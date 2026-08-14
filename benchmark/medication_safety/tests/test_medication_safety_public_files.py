from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from benchmark.medication_safety.scripts.analyze_system_prompt_tradeoff import (
    load_and_center,
)
from benchmark.medication_safety.scripts.build_benchmark_instances import (
    iter_benchmark_instances,
)
from benchmark.medication_safety.scripts.conservative_term_matching import (
    MedicationSafetyScorer,
)
from benchmark.medication_safety.scripts.medication_safety_utils import (
    build_user_prompt,
    load_benchmark_cases,
    load_patient_attitudes,
    load_replacements,
    load_synonym_groups,
    load_system_prompts,
    synonym_aware_match,
)
from benchmark.medication_safety.scripts.update_file_manifest import manifest_payload


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_yaml_loads_with_20_cases() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    cases = load_benchmark_cases(data_path)
    assert len(cases) == 20
    assert all(case["expected"]["contraindications"] is not None for case in cases)


def test_prompt_matrix_definitions_load() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    prompts = load_system_prompts(data_path)
    attitudes = load_patient_attitudes(data_path)
    assert set(prompts) == {"none", "minimal", "role_encouraging", "role_attitude_sensitive"}
    assert set(attitudes) == {"very_anxious", "anxious", "neutral", "confident"}
    first_prompt = build_user_prompt(load_benchmark_cases(data_path)[0], "anxious", attitudes)
    assert "What are the adverse reactions and contraindications?" in first_prompt


def test_benchmark_instance_builder_creates_320_instances() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    rows = list(iter_benchmark_instances(data_path))
    assert len(rows) == 320
    assert rows[0]["system_prompt_label"] in {
        "none",
        "minimal",
        "role_encouraging",
        "patient_attitude_sensitive",
    }


def test_public_term_matching_rules_detect_common_variants() -> None:
    replacements = load_replacements(
        ROOT / "results" / "term_matching" / "normalization_replacements.csv"
    )
    groups = load_synonym_groups(
        ROOT / "results" / "term_matching" / "canonical_synonym_groups.csv"
    )
    response = "The patient reported nosebleeds, joint pain, low blood sugar, and dyspnoea."
    assert synonym_aware_match("epistaxis", response, replacements, groups)
    assert synonym_aware_match("arthralgia", response, replacements, groups)
    assert synonym_aware_match("hypoglycemia", response, replacements, groups)
    assert synonym_aware_match("dyspnea", response, replacements, groups)


def test_conservative_scorer_uses_local_token_windows() -> None:
    scorer = MedicationSafetyScorer.from_files(
        ROOT / "data" / "benchmark_medication_safety_data.yaml",
        ROOT / "results" / "term_matching" / "normalization_replacements.csv",
        ROOT / "results" / "term_matching" / "canonical_synonym_groups.csv",
    )
    reordered = scorer.score(
        1,
        "Common adverse reactions:\n- Insulin combination was associated with heart failure.",
    )
    assert reordered["common_adverse_effects_coverage"] > 0

    separated = scorer.score(1, "Common adverse reactions:\n- Weight\n- gain")
    assert separated["common_adverse_effects_coverage"] == 0


def test_response_model_settings_include_primary_and_additional_models() -> None:
    with (ROOT / "model_settings" / "response_model_settings.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {row["Analysis role"] for row in rows} == {
        "Initial response-model set",
        "Additional response-model set",
    }


def test_manual_validation_summaries_use_512_response_subset() -> None:
    summary_path = ROOT / "results" / "manual_validation" / "manual_validation_512_by_metric.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {
        "understandability",
        "usefulness",
        "patient_attitude_responsiveness",
    }
    assert {int(row["n"]) for row in rows} == {512}

    calibration_path = (
        ROOT
        / "results"
        / "manual_validation"
        / "deepseek_subcriteria_calibration_512_by_metric.csv"
    )
    with calibration_path.open("r", encoding="utf-8", newline="") as handle:
        calibration_rows = list(csv.DictReader(handle))
    assert {int(row["n_responses"]) for row in calibration_rows} == {512}


def test_combined_system_prompt_source_data_reproduces_correlation() -> None:
    result_dir = ROOT / "results" / "additional_response_models"
    frame = load_and_center(
        result_dir / "combined_system_prompt_trajectory_source_data.csv"
    )
    summary = next(
        csv.DictReader(
            (result_dir / "combined_system_prompt_tradeoff_summary.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            )
        )
    )
    correlation = float(
        np.corrcoef(
            frame["structured_centered"],
            frame["communication_centered"],
        )[0, 1]
    )
    assert len(frame) == 32
    assert np.isclose(
        correlation,
        float(summary["pearson_r_model_centered"]),
        rtol=0,
        atol=1e-15,
    )


def test_file_manifest_is_complete_and_current() -> None:
    manifest_path = ROOT / "file_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["relative_path"]: row for row in csv.DictReader(handle)}

    expected_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert set(manifest) == expected_paths

    for relative_path, row in manifest.items():
        content = manifest_payload(ROOT / relative_path)
        assert int(row["bytes"]) == len(content)
        assert row["sha256"] == hashlib.sha256(content).hexdigest()
