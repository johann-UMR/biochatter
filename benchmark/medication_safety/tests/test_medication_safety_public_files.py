from __future__ import annotations

from pathlib import Path

from benchmark.medication_safety.scripts.medication_safety_utils import (
    build_user_prompt,
    load_benchmark_cases,
    load_patient_attitudes,
    load_replacements,
    load_synonym_groups,
    load_system_prompts,
    synonym_aware_match,
)
from benchmark.medication_safety.scripts.build_benchmark_instances import iter_benchmark_instances


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
    replacements = load_replacements(ROOT / "results" / "term_matching" / "normalization_replacements.csv")
    groups = load_synonym_groups(ROOT / "results" / "term_matching" / "canonical_synonym_groups.csv")
    response = "The patient reported nosebleeds, joint pain, low blood sugar, and dyspnoea."
    assert synonym_aware_match("epistaxis", response, replacements, groups)
    assert synonym_aware_match("arthralgia", response, replacements, groups)
    assert synonym_aware_match("hypoglycemia", response, replacements, groups)
    assert synonym_aware_match("dyspnea", response, replacements, groups)
