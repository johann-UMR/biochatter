from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmark.medication_safety.scripts.analyze_system_prompt_tradeoff import (
    load_and_center,
)
from benchmark.medication_safety.scripts.build_benchmark_instances import (
    iter_benchmark_instances,
)
from benchmark.medication_safety.scripts.conservative_term_matching import (
    MedicationSafetyScorer,
)
from benchmark.medication_safety.scripts.judge_subcriteria import (
    parse_subcriteria,
    render_prompt,
    threshold_label,
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
    assert set(prompts) == {
        "none",
        "minimal",
        "role_encouraging",
        "role_attitude_sensitive",
    }
    assert set(attitudes) == {"very_anxious", "anxious", "neutral", "confident"}
    first_prompt = build_user_prompt(
        load_benchmark_cases(data_path)[0], "anxious", attitudes
    )
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
    response = (
        "The patient reported nosebleeds, joint pain, low blood sugar, and dyspnoea."
    )
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
        "Common adverse reactions:\n"
        "- Insulin combination was associated with heart failure.",
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
    assert {row["Cohort"] for row in rows} == {
        "Original response-model cohort",
        "Expanded response-model cohort",
    }
    gpt = next(row for row in rows if row["Response model"] == "GPT-5.4")
    assert float(gpt["Temperature"]) == 0.0


def test_final_judge_prompt_thresholds_match_manual_validation() -> None:
    with (ROOT / "prompts" / "llm_judge_prompts.yaml").open(
        "r",
        encoding="utf-8",
    ) as handle:
        prompt_data = yaml.safe_load(handle)
    prompts = prompt_data["prompts"]
    assert prompts["understandability"]["threshold"] == 4
    assert prompts["usefulness"]["threshold"] == 5
    assert prompts["patient_attitude_responsiveness"]["threshold"] == 4
    assert prompts["patient_attitude_responsiveness"]["applicable_subcriteria"] == 4

    rendered = render_prompt(
        "usefulness",
        system_messages="No explicit system instructions.",
        prompt="Medication question",
        patient_attitude="neutral",
        response="Generated response",
    )
    assert "Medication question" in rendered
    assert '"q1":"yes"' in rendered

    understandable = parse_subcriteria(
        '{"q1":"yes","q2":"yes","q3":"yes","q4":"yes","q5":"no"}'
    )
    assert threshold_label(understandable, "understandability", "neutral") == 1

    responsive = parse_subcriteria(
        '{"q1":"yes","q2":"yes","q3":"not_applicable",'
        '"q4":"yes","q5":"yes"}'
    )
    assert threshold_label(
        responsive,
        "patient_attitude_responsiveness",
        "very_anxious",
    ) == 1


def test_manual_validation_summaries_use_512_response_subset() -> None:
    summary_path = (
        ROOT / "results" / "manual_validation" / "manual_validation_by_metric.csv"
    )
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {
        "understandability",
        "usefulness",
        "patient_attitude_responsiveness",
    }
    assert {int(row["n"]) for row in rows} == {512}

    report_path = (
        ROOT / "results" / "manual_validation" / "manual_validation_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["unique_validated_responses_with_at_least_one_complete_metric"] == 512
    assert report["incomplete_replacement_rows"] == 0


def test_final_score_corpus_is_complete_and_unique() -> None:
    score_path = (
        ROOT / "results" / "final_analysis" / "final_response_metric_scores.csv"
    )
    scores = pd.read_csv(score_path)
    key = [
        "model_label",
        "metric",
        "case_id",
        "system_prompt",
        "patient_attitude",
        "response_index",
    ]
    assert len(scores) == 102_400
    assert scores["model_label"].nunique() == 8
    assert scores["metric"].nunique() == 10
    assert not scores.duplicated(key).any()
    assert scores["score"].dropna().between(0, 1).all()

    report = json.loads(
        (ROOT / "results" / "final_analysis" / "pairwise_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["structured_tests"] == 1764
    assert report["communication_tests"] == 756

    structured_pairwise = pd.read_csv(
        ROOT
        / "results"
        / "structured_metrics"
        / "final_structured_pairwise_wilcoxon.csv"
    )
    communication_pairwise = pd.read_csv(
        ROOT
        / "results"
        / "llm_judge_metrics"
        / "final_communication_pairwise_mcnemar.csv"
    )
    assert len(structured_pairwise) == 1764
    assert len(communication_pairwise) == 756

    structured_within = pd.read_csv(
        ROOT
        / "results"
        / "structured_metrics"
        / "final_structured_within_friedman.csv"
    )
    communication_within = pd.read_csv(
        ROOT
        / "results"
        / "llm_judge_metrics"
        / "final_communication_within_cochran_q.csv"
    )
    assert len(structured_within) == 112
    assert len(communication_within) == 48

    model_summary = pd.read_csv(
        ROOT
        / "results"
        / "final_analysis"
        / "summary_model_structured_communication.csv"
    ).set_index("model_label")
    assert np.isclose(
        model_summary.loc["GPT-5.4", "structured_score"],
        0.6646336028711517,
    )
    assert np.isclose(
        model_summary.loc["Gemini 3.5 Flash", "communication_score"],
        0.9516927083333334,
    )


def test_temperature_sensitivity_uses_complete_paired_gpt54_runs() -> None:
    report = json.loads(
        (
            ROOT
            / "results"
            / "temperature_sensitivity"
            / "temperature0_vs_temperature1_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["temperature_0_rows"] == 1280
    assert report["temperature_1_rows"] == 1280
    structured = next(
        row
        for row in report["paired_metric_summary"]
        if row["metric"] == "structured_score"
    )
    assert np.isclose(structured["temperature_0_mean"], 0.6646336028711517)


def test_combined_system_prompt_source_data_reproduces_correlation() -> None:
    result_dir = ROOT / "results" / "final_analysis"
    frame = load_and_center(
        result_dir / "summary_model_system_prompt_trajectory.csv"
    )
    correlation = float(
        np.corrcoef(
            frame["structured_centered"],
            frame["communication_centered"],
        )[0, 1]
    )
    assert len(frame) == 32
    assert np.isclose(correlation, -0.6272343655209744, rtol=0, atol=1e-12)


def test_public_release_excludes_private_and_raw_outputs() -> None:
    forbidden_name_parts = {
        "answer_key",
        "manual_validation_response_level",
        "private",
        "raw_output",
        "responses_long",
    }
    relative_paths = [
        path.relative_to(ROOT).as_posix().lower() for path in ROOT.rglob("*")
    ]
    assert not [
        path
        for path in relative_paths
        if any(part in path for part in forbidden_name_parts)
    ]

    text_extensions = {".csv", ".json", ".md", ".py", ".txt", ".yaml"}
    for path in ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in text_extensions
            and path.resolve() != Path(__file__).resolve()
        ):
            content = path.read_text(encoding="utf-8", errors="ignore")
            assert "C:\\Users\\johan" not in content
            assert "OPENAI_API_KEY=" not in content
            assert "DEEPSEEK_API_KEY=" not in content


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
