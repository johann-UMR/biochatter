# Medication safety benchmark

This folder contains the public release of a medication safety benchmark
evaluating large language model responses about adverse reactions and
contraindications.

The benchmark includes 20 synthetic adult medication indication cases based on
European Medicines Agency Product Information. Each medication indication case is
combined with four system prompts and four patient attitude conditions. The
initial four response models were evaluated across four repeated response
iterations, yielding 5,120 responses. Four additional response models were
evaluated with the same design, yielding another 5,120 responses. The combined
eight-model aggregate analyses therefore cover 10,240 generated responses.

## Contents

- `data/benchmark_medication_safety_data.yaml`: benchmark cases, EMA source
  metadata, system prompts, patient attitude statements, curated adverse
  reactions, and contraindications.
- `prompts/response_generation_prompts.yaml`: response generation prompt setup.
- `prompts/llm_judge_prompts.yaml`: LLM-judge rubrics for understandability,
  usefulness, and patient attitude responsiveness.
- `model_settings/`: curated model and judge metadata. These files contain model
  IDs and inference settings, not API keys or provider account data.
- `scripts/`: utilities for loading cases, recreating the 320 benchmark prompts,
  and applying the conservative term matching procedure to response text.
- `results/structured_metrics/`: final aggregate structured medication safety
  results from the conservative canonical term matching run.
- `results/llm_judge_metrics/`: primary DeepSeek V4 Flash judge inferential
  summaries.
- `results/manual_validation/`: aggregate manual validation summaries for the
  512-response validation subset, discordance analyses, and a subcriterion-level
  DeepSeek calibration sensitivity summary.
- `results/judge_sensitivity/`: aggregate multi judge sensitivity summaries.
- `results/figure_source_data/`: source data tables used to generate manuscript
  figures. Final figure image files are not included in this initial release.
- `results/term_matching/`: normalization replacements and canonical synonym
  groups used for term matching.
- `results/additional_response_models/`: aggregate structured and DeepSeek judge
  results for the four additional response models, including source data for the
  combined system prompt analysis.
- `tests/`: minimal smoke tests for the public benchmark files.
- `environment/python_pip_freeze.txt`: package snapshot from the local analysis
  environment.

## Not included

The public release intentionally excludes full generated model
responses, raw LLM-judge outputs, row level manual validation labels, provider
request logs, API keys, and local runtime configuration. Aggregate outputs and
figure source data are included to support reproducibility of the reported
analyses without disclosing raw generated text.

## Basic usage

Run the smoke tests from the repository root:

```bash
python -m pytest benchmark/medication_safety/tests \
  --confcutdir=benchmark/medication_safety
```

Build the 320 benchmark instances:

```bash
python -m benchmark.medication_safety.scripts.build_benchmark_instances
```

To write them to a file, pass `--output path/to/benchmark_instances.csv`.

Score a CSV containing one generated response per row with `case_id` and
`response` columns:

```bash
python -m benchmark.medication_safety.scripts.score_responses responses.csv \
  --output scored_responses.csv
```

Reproduce the combined system prompt correlation and permutation test:

```bash
python -m benchmark.medication_safety.scripts.analyze_system_prompt_tradeoff
```

The smoke tests verify that the benchmark YAML loads, that the expected 20
medication indication cases are present, and that the public matcher applies
normalization, canonical synonym groups, local text segmentation, and the
12-token fallback window.
The YAML keeps the original technical prompt key `role_attitude_sensitive`;
manuscript text refers to the same prompt as the patient attitude sensitive
prompt.

## Notes on reproducibility

The aggregate result files in `results/` are derived from the final DeepSeek
primary judge analysis, the conservative canonical term matching run, and the
512-response manual validation subset. The four additional response models were
included in the DeepSeek and manual validation aggregates, but not in the
alternative-judge sensitivity analysis. Some full corpus recomputation steps
require raw model responses and raw judge outputs, which are not part of this
public release. The released files therefore support audit of benchmark
definitions, prompt design, scoring rules, aggregate outputs, and figure source
data, while raw generated text remains excluded.
