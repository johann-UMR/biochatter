# Additional response models

These files contain aggregate results for Gemini 3.5 Flash, Llama 4 Maverick,
Llama 3.1 8B Instruct, and GLM 4.5. Each model was evaluated on the same 320
benchmark instances and four response iterations as the primary response models,
yielding 1,280 responses per model and 5,120 responses in total.

Structured scores were calculated with the conservative term matching procedure
released in `../../scripts/conservative_term_matching.py`. Communication scores
were assigned by the primary DeepSeek V4 Flash judge using two judge iterations,
thinking enabled, `reasoning_effort=high`, temperature 0, and a maximum output of
2,048 tokens.

The `new_model_*` files contain aggregate results for the four additional models.
The `combined_*` files combine these aggregates with the four primary response
models for the system prompt analysis. Full generated responses and raw judge
outputs are intentionally excluded.
