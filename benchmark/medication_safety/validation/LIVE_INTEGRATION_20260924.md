# Live BioChatter integration check

Completed 24 September 2026 at 21:20:57 UTC. This is a software execution check,
not an additional study result or a replication of the eight-model experiment.

## Execution

The actual BioChatter provider, response-generation CLI, response CSV writer,
structured scorer and subcriterion judge CLI were used with live DeepSeek calls.
The requested and provider-reported model identifier was `deepseek-v4-flash`.
The case was pioglitazone in adult type 2 diabetes, with the Minimal system
prompt and very-anxious patient-attitude condition.

```bash
python -m benchmark.medication_safety.scripts.live_smoke \
  --model deepseek-v4-flash \
  --system-prompt minimal --patient-attitude very_anxious \
  --generation-max-tokens 16384 \
  --output-dir benchmark/results/medication_safety_smoke
```

Credentials are supplied only through an environment variable. The command
makes paid calls; it is not part of CI. Use a fresh output directory.

| Check | Observed result |
| --- | --- |
| Response generation | Four complete responses, saved in BioChatter format |
| Generation settings | Temperature 0; thinking disabled; token limit 16384 |
| Structured scoring | Four rows with seven metric columns; applicable scores in [0, 1] |
| Judge settings | Temperature 0; thinking enabled; reasoning effort high; token limit 4096 |
| Criterion-level judgements | 24: four responses x three criteria x two rounds |
| Communication summaries | 12 response-criterion rows, each with two distinct rounds |
| Independent label check | All 24 labels reproduced from their subcriteria |
| Independent aggregation check | All 12 mean scores and strict labels reproduced |
| Completion status | All 28 inference completions ended with `stop`, not `length` |
| Resume | No additional inference calls; response, judgement and score files unchanged |

The successful run reported 188586 total tokens across the 28 completions.
This excludes earlier failed attempts. Provider authentication/model-list
requests are not counted as inference calls.

## Failed attempts and limits

Earlier attempts at generation budgets of 4096 and 8192 tokens failed because
the provider returned a length-limited response. No judge calls were started in
those attempts. The test deliberately rejected incomplete responses. In the
8192-token attempt, the four completion lengths were 6398, 5437, 6234 and 8192;
the last had finish reason `length`. These failures are not hidden by the
successful rerun and do not imply the lower limits always fail.

The successful run's response lengths were 6811, 684, 7892 and 3206 tokens.
The higher ceiling was a test-specific choice, not a change to study settings.
No new scientific comparisons were calculated or added to the paper.

This check covers one case, one condition and the DeepSeek provider. It does
not establish operation of every provider, full historical reproducibility,
clinical validity, or successful execution of the remote GitHub CI. Local
dependency deprecation warnings were present but did not prevent completion.

Raw responses, per-item judge outputs, credentials and local logs are not
included here. The local test retains those outputs for diagnosis. This report
records an observed run; external users can rerun the command with their own
credentials, but are not guaranteed identical generated content.
