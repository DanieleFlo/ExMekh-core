# Numerical and performance investigation

When a numerical mismatch or a performance regression is found, investigate
the corresponding upstream implementations and previous repository research
before changing the kernel or its dispatch. This is a standing user instruction.

- Consult the relevant Transformers implementation for model semantics,
  dtypes, masks, operation order and rounding boundaries. Include Diffusers
  for the Qwen Image pipeline, DiT and VAE.
- Inspect the corresponding Strata and llama.cpp/ggml kernels, wrappers,
  capability checks and tests. Check actual model and operation coverage.
- Search this repository's architecture notes, performance reports,
  diagnostics, CUDA/Metal implementations and similar investigated cases.
  Reuse established findings where their assumptions match the current case.
- Record the exact source revisions and paths, the observed difference,
  transferable findings, incompatible assumptions and the chosen approach
  in the task's diagnostic or performance report before implementation.
- Verify the change on identical isolated inputs, then repeat the independent
  full-model gate. Preserve frozen numerical criteria and distinguish static
  comparisons from measured performance results.

Apply this investigation again when a subsequent difference is found; an
earlier source audit does not explain a new discrepancy automatically.

# Metal inference optimization

Follow `docs/roadmaps/metal-inference-performance-20261102.md` and update its
progress ledger after each experiment. Work on one model at a time, starting
with Flash Next Q2 streamed, then Qwen3.8-27B streamed.

- Profile before choosing an optimization and profile again after improvements.
- For every kernel, dispatch, scheduling or inference-workflow optimization,
  revisit the relevant upstream code and local research before implementation;
  record exact revisions, paths, applicability and alternatives in its report.
  An earlier audit is not a substitute for this per-intervention comparison.
- Measure prefill, decode and request latency separately. Allow phase/shape/
  capability-specific dispatch when it preserves the model's semantics.
- Track small gains and evaluate coherent bundles against a frozen baseline.
  Do not reject a component solely because it gains less than 1–2%; distinguish
  measured gains from noise and validate bundle correctness and total speedup.
