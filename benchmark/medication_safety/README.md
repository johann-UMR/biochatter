# Medication safety benchmark

This folder contains the public release candidate for a medication safety benchmark
evaluating large language model responses about adverse reactions and
contraindications.

The benchmark includes 20 synthetic adult medication indication cases based on
European Medicines Agency Product Information. Each medication indication case is
combined with four system prompts and four patient attitude conditions. Four
response models were evaluated across four repeated response iterations, yielding
5,120 generated responses in the study.

## Contents

- `data/benchmark_medication_safety_data.yaml`: benchmark cases, EMA source
  metadata, system prompts, patient attitude statements, curated adverse
  reactions, and contraindications.
- `prompts/response_generation_prompts.yaml`: response generation prompt setup.
- `prompts/llm_judge_prompts.yaml`: LLM-judge rubrics for understandability,
  usefulness, and patient attitude responsiveness.
- `model_settings/`: curated model and judge metadata. These files contain model
  IDs and inference settings, not API keys or provider account data.
- `scripts/`: lightweight utilities for loading cases and applying the public
  term normalization and synonym rules. The instance builder can recreate the
  320 benchmark prompts from the released YAML file.
- `results/structured_metrics/`: final aggregate structured medication safety
  results from the conservative canonical term matching run.
- `results/llm_judge_metrics/`: primary DeepSeek V4 Flash judge inferential
  summaries.
- `results/manual_validation/`: aggregate manual validation summaries and
  discordance analyses.
- `results/judge_sensitivity/`: aggregate multi judge sensitivity summaries.
- `results/figure_source_data/`: source data tables used to generate manuscript
  figures. Final figure image files are not included in this no figure candidate.
- `results/term_matching/`: normalization replacements and canonical synonym
  groups used for term matching.
- `tests/`: minimal smoke tests for the public benchmark files.
- `environment/python_pip_freeze.txt`: package snapshot from the local analysis
  environment.

## Not included

The public release candidate intentionally excludes full generated model
responses, raw LLM-judge outputs, row level manual validation labels, provider
request logs, API keys, and local runtime configuration. Aggregate outputs and
figure source data are included to support reproducibility of the reported
analyses without disclosing raw generated text.

## Basic usage

Run the smoke tests from the repository root:

```bash
python -m pytest benchmark/medication_safety/tests
```

Build the 320 benchmark instances:

```bash
python -m benchmark.medication_safety.scripts.build_benchmark_instances
```

To write them to a file, pass `--output path/to/benchmark_instances.csv`.

The smoke tests verify that the benchmark YAML loads, that the expected 20
medication indication cases are present, and that the public normalization and
synonym matching utilities detect common lexical variants in a toy response.
The YAML keeps the original technical prompt key `role_attitude_sensitive`;
manuscript text refers to the same prompt as the patient attitude sensitive
prompt.

## Notes on reproducibility

The aggregate result files in `results/` are derived from the final DeepSeek
primary judge analysis and the conservative canonical term matching run. Some
full corpus recomputation steps require raw model responses and raw judge
outputs, which are not part of this public release candidate. The released files
therefore support audit of benchmark definitions, prompt design, scoring rules,
aggregate outputs, and figure source data, while raw generated text remains
excluded.
