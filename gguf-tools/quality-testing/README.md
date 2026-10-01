# Official-Continuation Quality Testing

This directory contains the prompts, tracked official fixtures, and scripts used
to compare local GGUF variants against hosted-model continuations.

The main metric is target-token negative log likelihood: collect an
official continuation, then ask each local GGUF how much probability it assigns
to that exact continuation token by token.  This avoids judging quality from one
sampled answer.

## 1. Tracked Fixture Sets

Curated fixtures are kept in the repository so release QA can run without
calling hosted APIs:

- `data/glm52-openrouter-100`: 100 GLM 5.2 continuations collected through
  OpenRouter `z-ai/glm-5.2` with `top_logprobs=20`.
- `data/glm53-flash-openrouter-zai-fp8-100`: 100 GLM 5.3 Flash continuations
  from OpenRouter's pinned Z.AI FP8 endpoint. The endpoint does not expose
  logprobs, so these are deterministic continuation fixtures.
- `data/glm53-flash-openrouter-zai-fp8-long`: eight code-context continuations
  from the same FP8 endpoint, with 3K-7K prompts that exercise sparse attention.
  Use the rendered prompts below to preserve the API's reasoning prefix.
- `data/flash`: 100 DeepSeek V4 Flash 0731 continuations collected from the
  official DeepSeek API with `top_logprobs=20`.
- `data/pro`: 100 DeepSeek V4 PRO preview continuations collected from the
  official DeepSeek API with `top_logprobs=20`.
- `data/pro-0813`: 100 DeepSeek V4 PRO 0813 continuations collected from the
  official DeepSeek API with `top_logprobs=20`.
- `data/qwen38-flash-alibaba-100`: 100 non-thinking Qwen3.8 Flash
  continuations from Alibaba through OpenRouter, with top-five logprobs.
- `data/qwen38-flash-alibaba-long`: 12 archive/code continuations from the
  same endpoint, with prompts from 2K to 24K tokens.
- [deepseek-v4.1-flash-20260919-router](deepseek-v4.1-flash-20260919-router/README.md):
  112 fresh official V4.1 Flash continuations, including 12 longer prompts up
  to 23K tokens. Temperature-1 samples with top-20 logprobs, intended for
  paired numerical comparisons. These are not V4 Flash or Vision-Exp vectors.

DeepSeek V4 Flash also has tracked official smoke vectors in
`tests/test-vectors/`.  Those vectors drive `./ds4_test --logprob-vectors` and
include short prompts plus long-prompt attention cases.

The hosted APIs expose output-token logprobs and top-logprob alternatives, not
full vocabulary logits.

The scorer verifies each API token's bytes against the local token boundaries,
not just the number of tokens. If they differ, it still scores the continuation
text but skips API logprob comparisons for that case. Alternatives containing
Unicode replacement characters are excluded because the original token bytes
may have been lost by the provider.

## 2. Collect Official Continuations

The Flash 0731 fixture is historical. The official API now redirects the old
Flash names to V4.1; do not overwrite `data/flash` with new calls. Collect
current V4.1 references in a separate directory:

```sh
export DEEPSEEK_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-flash \
  --endpoint https://api.deepseek.com/chat/completions \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out /tmp/deepseek-v4.1-flash-new \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --thinking disabled \
  --reasoning-effort omit
```

For GLM 5.2 through OpenRouter:

```sh
export OPENROUTER_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model z-ai/glm-5.2 \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/glm52-openrouter \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --token-limit-field max_tokens \
  --provider-order parasail/fp8 \
  --require-parameters \
  --thinking omit \
  --reasoning-effort none
```

For GLM 5.3 Flash through the pinned Z.AI FP8 endpoint:

```sh
export OPENROUTER_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model z-ai/glm-5.3-flash \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100 \
  --count 100 \
  --max-tokens 128 \
  --top-logprobs 0 \
  --token-limit-field max_tokens \
  --provider-order z-ai/fp8 \
  --thinking omit \
  --reasoning-effort low
```

Use one output directory per checkpoint. For PRO 0813 through the official
DeepSeek API:

```sh
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-pro \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/pro-0813 \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20
```

The script writes:

- `data/<model>/prompts/case_*.txt`
- `data/<model>/continuations/case_*.txt`
- `data/<model>/responses/case_*.json`
- `data/<model>/manifest.tsv`

The prompt list is tracked in `prompts.jsonl`.  Curated fixture directories are
also tracked after review; ad-hoc API collection directories should stay
untracked until they are intentionally promoted into the release QA set.

Use `--system TEXT` when the reference needs an explicit system message.
`--system ''` sends an empty message, which is different from omitting it:
some providers insert a default system prompt when none is supplied. The
collector records this setting and rejects a changed system message on resume.
For a nonempty system message, render the matching local chat prompt and use
`--rendered-prompt`; the scorer's ordinary prompt mode adds no system message.

For Qwen, use the following settings in a new output directory:

```sh
python3 gguf-tools/quality-testing/collect_official.py \
  --out /tmp/qwen-official \
  --model qwen/qwen3.8-flash \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --provider-order alibaba --require-parameters \
  --require-response-provider Alibaba --require-response-model qwen/qwen3.8-flash \
  --system '' --thinking omit --reasoning-effort none \
  --count 100 --max-tokens 64 --top-logprobs 5
```

This reads `OPENROUTER_API_KEY` from the environment and disables provider
fallbacks. Omitting `--system ''` caused Alibaba to add 13 prompt tokens in
the September 2026 checks. The empty message matches Qwen's published
no-thinking template and the scorer's ordinary prompt mode.

Alibaba does not disclose this endpoint's precision. The
[official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) describes
hosted Flash as based on Flash Next, not as a byte-identical checkpoint.
Agreement with this service is an external quality check, not proof of exact
native-weight inference.

## 3. Build The Local Scorer

```sh
make -C gguf-tools quality-score
```

The scorer links against the DS4 runtime, using Metal on macOS and CUDA on
Linux. On a DGX Spark, pass `CUDA_ARCH=sm_121` to the build command.

Build the optional llama.cpp control scorer with:

```sh
make -C gguf-tools quality-llama-score
```

## 4. Score GGUF Variants

```sh
gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/OLD.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/old.tsv \
  4096

gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/NEW.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/new.tsv \
  4096
```

Use `data/flash/manifest.tsv` for Flash GGUFs and
`data/glm52-openrouter-100/manifest.tsv` for GLM 5.2 GGUFs. Use
`data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv` for GLM 5.3 Flash. Use
`data/pro/manifest.tsv` for the PRO preview checkpoint and
`data/pro-0813/manifest.tsv` for PRO 0813. The scorer and comparator do not
care which model produced the manifest; the manifest path selects the
continuation set.

Add `--continued-prefill N` to process each prompt in two calls, leaving its
last `N` tokens for the second call. This checks continued-prefill quality
against the same official answers. Every prompt must contain more than `N`
tokens; otherwise the scorer fails instead of silently skipping the test.

Add `--session-batch N` (2 to 8) to score an official continuation alongside
unrelated, independently advancing sessions. The scorer rotates their row
order on every step. Compare against a run without this option using the same
manifest, model and context; budget memory for all `N` sessions.

Check session isolation separately on a dedicated Metal host:

```sh
DS4_TEST_MODEL=MODEL.gguf DS4_TEST_BATCH_ISOLATION=1 \
  DS4_TEST_SESSION_COUNT=4 DS4_TEST_DECODE_STEPS=32 \
  MTL_DEBUG_LAYER=1 ./tests/test_metal_session_batch
```

This requires exact logits when companion prompts and row order change, then
checks a continued prefill and resumed serial decoding. It also checks that
invalid batches leave the target unchanged. It does not replace serial/batch
quality comparisons: different arithmetic can round differently without
sessions contaminating one another. Use `DS4_TEST_PROMPT_FILE` and
`DS4_TEST_CONTEXT_SIZE` to repeat at longer prefixes. For physical TP, the test
also accepts `DS4_TEST_TP_RDMA_DEVICE` and `DS4_TEST_TP_GID_INDEX` alongside
its coordinator/worker settings. Throughput measurements must run separately
without Metal API validation.

For Metal V4.1 prefill scheduling changes, also run:

```sh
make tests/test_deepseek41_prefill
./tests/test_deepseek41_prefill --dispatch
MTL_DEBUG_LAYER=1 ./tests/test_deepseek41_prefill MODEL.gguf speed-bench/promessi_sposi.txt
```

The model test uses SSD streaming and two 128K sessions. Run it alone on a
dedicated host with at least 128 GiB RAM. It alternates small and large appends
through 113K context, checking dispatch, progress, unchanged-prefix reuse,
saved state and subsequent decoding against a control without decoder deferral.
Keep the official continuation checks too: this scheduling test does not judge
quality across different floating-point operation orders. Measure speed without
Metal API validation, separately for initial and continued prefills.

Run the same mixed-prefix test over physical RDMA on two dedicated Metal hosts.
Start the worker with the same model and a 131072-token context, then the test
on the coordinator (replace the host and RDMA device names):

```sh
# Worker
MTL_DEBUG_LAYER=1 ./ds4 -m MODEL.gguf --ctx 131072 \
  --tensor-parallel --role worker --coordinator COORDINATOR 19455 \
  --transport rdma --rdma-device WORKER_DEVICE --rdma-gid-index 1

# Coordinator
MTL_DEBUG_LAYER=1 ./tests/test_deepseek41_prefill --tensor-parallel \
  MODEL.gguf speed-bench/promessi_sposi.txt COORDINATOR 19455 COORDINATOR_DEVICE 1
```

This compares equal TP prefill partitions and queued versus synchronous decode,
including full logits and cache state. TP snapshots rebuild both ranks from
tokens, so restoration is compared with an equivalent fresh replay. Also run
the official scorer in TP mode for initial and continued prompts; scheduling
parity alone does not establish model quality.

Add `--quality` to disable DS4's speed-oriented numerical shortcuts. For an
independent llama.cpp comparison of a DeepSeek V4 GGUF, use the same manifest
and the token-identical DS4 prompt renderer:

```sh
gguf-tools/quality-testing/score_llama \
  /path/to/model.gguf \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/llama.tsv \
  4096 \
  deepseek-ds4
```

GLM's hosted endpoint requires reasoning. A no-thinking local prompt does not
match a reply generated after reasoning, even at temperature zero. For the long
GLM fixtures, render the official `Reasoning Effort: Low` template and include
the returned reasoning before scoring the answer:

```sh
python3 gguf-tools/quality-testing/render_glm_references.py \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-long
gguf-tools/quality-testing/score_official /path/to/GLM-5.3-Flash-Q2.gguf \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-long/manifest-rendered.tsv \
  /tmp/glm-long.tsv 8192 --rendered-prompt
```

`--rendered-prompt` reads the prompt file with its existing chat markers; it
does not add another chat template. The renderer can also be applied to the
older 100-case GLM 5.3 Flash fixture. Keep rendered and legacy no-thinking
scores separate: their prefixes differ. Z.AI supplies no token logprobs, so
judge these runs by continuation NLL and greedy agreement, not logprob parity.

For a full-residency vs SSD-streaming comparison, score the same model twice and
add the streaming flags to one run:

```sh
gguf-tools/quality-testing/score_official \
  /path/to/model.gguf \
  gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv \
  /tmp/streaming.tsv \
  4096 \
  --ssd-streaming
```

## 5. Compare

```sh
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Output fields:

- `avg_nll`: average negative log likelihood; lower is better.
- `delta_new_minus_old`: negative means the new GGUF fits the official
  continuation better.
- `case_wins_new_old_ties`: per-prompt NLL wins.
- `first_token_matches`: how often the local greedy first token matches the
  official first token.
- `avg_greedy_lcp`: average greedy longest common prefix against the official
  continuation. This measures exact agreement, not general answer quality;
  one early mismatch discards all later agreement. Do not use it alone as a
  quality gate, especially for sampled references. Compare paired NLL and
  API probability agreement under matching execution settings as well.
- `api_target_mae`: when the manifest includes `response_file`, absolute
  local-vs-API logprob delta for aligned official output tokens.
- `api_top_coverage`: fraction of API top-logprob alternatives that map exactly
  to one local tokenizer token.
- `api_top1_rate`: how often the API top alternative equals the local greedy
  token.
- `api_topn_recall`: fraction of mapped API top-N alternatives found in the
  local top-N for the same position.
- `api_top_mae`: local-vs-API logprob MAE over mapped API top alternatives.
- `api_pair_rate`: pairwise ordering agreement among mapped API alternatives.

## 6. Qwen3.6 Fixture Staging

Qwen3.6 uses the same data-set layout as the existing quality fixtures.  The
tracked manifests live under `data/qwen36-27b` and `data/qwen36-27b-mtp`, with
`prompts`, `continuations`, and `responses` directories.  Shared scripts stay
in this directory alongside `collect_official.py` and the existing scorers.

Validate both manifests without loading a model:

```sh
python3 gguf-tools/quality-testing/validate_qwen36.py \
  gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  gguf-tools/quality-testing/data/qwen36-27b-mtp/manifest.json

make -C gguf-tools quality-qwen36-test
```

Inspect local GGUF headers and regenerate the tracked, deterministic metadata
snapshots.  Local model paths are command-line inputs and are never stored in
the manifests:

```sh
python3 gguf-tools/quality-testing/inspect_qwen36_gguf.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --artifact target=gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --snapshot-dir gguf-tools/quality-testing/data/qwen36-metadata

python3 gguf-tools/quality-testing/inspect_qwen36_gguf.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b-mtp/manifest.json \
  --artifact mtp=gguf/mtp-Qwen3.6-27B-Q4_0.gguf \
  --snapshot-dir gguf-tools/quality-testing/data/qwen36-metadata
```

The inspector validates the complete artifact SHA-256 and size, GGUF v3
structure, semantic metadata, tokenizer/chat-template hashes, tensor names,
shapes, offsets and quantization types.  Duplicate or truncated directory
entries are rejected.

Materialize the deterministic long prompt into an ignored staging directory:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_prompt.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --case long_canary_4096 \
  --staging-dir gguf-tools/quality-testing/staging/local
```

Generate an oracle candidate by choosing `llama.cpp`, `transformers`, or
`vllm`.  Every run requires complete provenance and writes the familiar
`prompts/`, `continuations/`, `responses/`, and `manifest.tsv` structure:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_oracle.py \
  --manifest /path/to/local-manifest.json \
  --oracle llama.cpp \
  --staging-dir /path/to/qwen36-staging \
  --run-id llama-q4-cuda-001 \
  --engine-commit LLAMA_CPP_FULL_COMMIT \
  --build-flags '-DGGML_CUDA=ON' \
  --backend CUDA \
  --hardware 'NVIDIA GPU model and driver' \
  --dtype 'GGUF Q4_K_M'
```

For the pinned local target, pass the model without editing the manifest and
size the runtime context for the selected corpus:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_oracle.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --oracle llama.cpp \
  --artifact-path target=gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --staging-dir gguf-tools/quality-testing/staging/oracles \
  --run-id llama-q4-cuda-001 \
  --steps 32 --top-k 20 --context 24576 --n-gpu-layers -1 \
  --engine-commit LLAMA_CPP_FULL_COMMIT \
  --build-flags 'EXACT BUILD FLAGS' \
  --backend CUDA --hardware 'EXACT GPU AND DRIVER' --dtype 'GGUF Q4_K_M'
```

llama.cpp and Transformers save separate greedy and teacher-forced
full-vocabulary float32 streams.  Every stream records shape, byte size and
SHA-256.  After producing a second run, validate both candidates and their
render/token determinism:

```sh
python3 gguf-tools/quality-testing/verify_qwen36_run.py \
  gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --repeat gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-002 \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json
```

The llama.cpp adapter records full-vocabulary logits against the exact pinned
GGUF.  Transformers and vLLM are labelled semantic upstream oracles because
their weights and precision differ.  vLLM records top-logprob slices because
its public generation API does not expose full logits.

Generators reject `golden` and `goldens` paths, reject non-empty unmarked
staging directories, and provide no promotion or acceptance command.  Review
and copy approved fixture data manually.  Missing models, runtimes, or hardware
remain `not_verified`; they are never treated as passing.

The Q4 target was verified on an RTX 3090 with llama-cpp-python 0.3.23,
llama.cpp commit `7d442abf5c6244117fd5a1dc51a5d19f00792491`, CUDA full offload and two
independent 32-step runs over all nine corpus categories.  The generated runs
remain ignored, unreviewed staging candidates.  Transformers, vLLM and MTP
execution are still `not_verified` on this host.

## 7. Qwen3.6 Numerical Equivalence

The structured scorer mode bypasses templates and tokenization during the
numeric pass. It consumes canonical prompt and teacher-forced token IDs from
an existing, validated oracle run and keeps the model resident while writing
greedy and teacher-forced `float32-le` logits. Python owns the same directory,
checksum and provenance format introduced in section 6:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_score.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --source-run gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --scorer gguf-tools/quality-testing/score_official \
  --model gguf/Qwen3.6-27B-Q4_K_M.gguf --engine ds4 \
  --staging-dir gguf-tools/quality-testing/staging/oracles \
  --run-id ds4-q4-cuda-001 --context 24576 --top-k 20 \
  --engine-commit DS4_FULL_COMMIT --build-flags 'EXACT BUILD FLAGS' \
  --backend CUDA --hardware 'EXACT GPU AND DRIVER' --dtype 'GGUF Q4_K_M' \
  --prefill-chunk 2048
```

For the temporary 24 GiB GPU cap, use the versioned
`data/qwen36-27b/context-profile-16k.json` together with
`manifest-16k.json`, pass `--context 16384 --context-profile ...` explicitly,
and use `run_qwen36_context_matrix.py` to expand the 4096/16384 frontiers.
The context limit is configuration data, not a compiled engine constant.
On CUDA, weight arenas default to a 256 MiB reserve granularity so a Q4_K_M
model does not strand multiple GiB of VRAM in partially used arenas; the
`DS4_CUDA_WEIGHT_ARENA_CHUNK_MB` override remains available for diagnostics.

Use `score_llama` and `--engine llama.cpp` to collect the same token-driven
layout from the C++ scorer. Existing positional TSV invocations of both
scorers remain unchanged. Until native Qwen chat rendering exists, a DS4 run
records `native_rendering_status=tokenizer_only`; this prevents a report from
passing while still allowing numeric diagnostics.

`score_official --quality` enables the runtime quality mode. For an output-head
localization test, `--output-head-hidden FILE --output-head-logits FILE` reads
one exact 5120-element F32 hidden state and writes the corresponding
248320-element DS4 output-head logits without running the transformer body.
For the patched local llama scorer, `LLAMA_QWEN_CPU=1` forces zero GPU layers
and disables KQV/operation offload; verify the startup buffers before labeling
the result a CPU oracle.

Compare two complete runs with one command:

```sh
python3 gguf-tools/quality-testing/compare_qwen36_equivalence.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --mode ds4-vs-llama \
  --left-run gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --right-run gguf-tools/quality-testing/staging/oracles/ds4-q4-cuda-001 \
  --report gguf-tools/quality-testing/staging/oracles/ds4-vs-llama.json
```

Task 4 uses the same comparator and run layout with the fixed short-message
profile declared in the manifest:

```sh
python3 gguf-tools/quality-testing/compare_qwen36_equivalence.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --mode ds4-vs-llama --suite short --top-k 20 \
  --left-run gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --right-run gguf-tools/quality-testing/staging/oracles/ds4-q4-cuda-001 \
  --report gguf-tools/quality-testing/staging/oracles/ds4-vs-llama-short.json
```

The short suite selects every corpus category except `long-canary`, requires
32 greedy and teacher-forced positions, and blocks prompt/token/decoded-byte
differences and non-finite logits per position.  The calibrated v2 profile
also requires aggregate top-20 overlap and rank agreement of at least 0.95,
and aggregate teacher-forced oracle-logprob MAE no greater than 0.05.
Per-position cross-engine metric envelopes remain versioned separately in
`equivalence_thresholds.cross_engine`. Top-1/top-2 margins at or below 0.10
are reported but do not fail the run. Native rendered bytes and greedy decoded
bytes are stored as compact hex in the response JSON; full logits remain
checksummed `float32-le` files.

`ds4-vs-ds4` additionally requires every float32 bit pattern to be identical.
For a long run whose reference contains more cases than the candidate,
`--case CASE_ID` selects an explicit common case and may be repeated; it cannot
be combined with `--suite short` and never relaxes provenance or metric gates.
Exit statuses are `0` for `PASS`, `1` for a failed gate, `2` for invalid input,
and `3` for `NOT_VERIFIED`. `--diagnostic` permits nonstandard engine roles
for tooling checks but can never produce `PASS`. Unreviewed runs or missing
cross-engine calibration likewise remain `NOT_VERIFIED`; neither collection
nor comparison can promote golden files or update thresholds automatically.

### Qwen layer trace and long-run speed gate

The short-prefix diagnostic can capture the final token at every layer without
changing model semantics. First dump only `layer_out` (DS4) and `l_out`
(llama.cpp) with `DS4_QWEN_TRACE_*` and `LLAMA_QWEN_TRACE_*`; use layer `all`
for the boundary bisection, then repeat without a stage filter on the first bad
layer. `compare_qwen36_trace.py` associates embedding, attention/GDN,
recurrent state, FFN and residual stages and writes the first float32
divergence. Trace directories must already exist. These reports are diagnostic
and do not calibrate or update cross-engine thresholds.

`DS4_CUDA_QWEN_Q8_ACT_DIAG=1` is an intentionally slow diagnostic path. It
tests CPU-like Q8_K activation quantization and Q4_K integer grouping.
`DS4_CUDA_QWEN_Q8_0_CPU_SCALE_DIAG=1` separately rounds Q8_0 activation scales
to FP16. Neither changes the default path; do not combine them unless the
experiment explicitly calls for two variables, and do not promote either on
the basis of lower MAE alone. The 2026-08-05 experiments did not fix the greedy
top-1. The decision log is `docs/performance/qwen36-drift-hypotheses.md`.

`run_qwen36_context_matrix.py --execute` is blocked unless
`--performance-report` names a `ds4-qwen36-speed-v1` JSON report for the same
model SHA-256, backend and hardware. The report records `engine_commit`,
`build_flags`, exact `command` argv, `context`, `prefill_tokens`,
`decode_tokens`, both measured token/s values and `peak_memory_mib`. The runner
refuses missing/non-comparable measurements, prefill below 500 token/s, or
decode below 20 token/s before starting a long-context model process.

## 8. Qwen3.8 Generation-Isolated Quality Path

Qwen3.8 reuses the audited Qwen input corpus, never Qwen3.6 continuations or
logits. Its manifest, staging marker, run format, thresholds, snapshots and
same-GGUF oracle namespace are all `ds4-qwen38-*`:

```sh
python3 gguf-tools/quality-testing/validate_qwen38.py \
  gguf-tools/quality-testing/data/qwen38-27b/manifest.json
make -C gguf-tools quality-qwen38-test
```

The generation-aware tools have explicit Qwen3.8 entrypoints:

```sh
python3 gguf-tools/quality-testing/generate_qwen38_oracle.py \
  --manifest gguf-tools/quality-testing/data/qwen38-27b/manifest.json \
  --oracle llama.cpp --artifact-path target=gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --staging-dir /path/to/qwen38-staging --run-id llama-qwen38-001 \
  --steps 32 --top-k 20 --context 8192 --n-gpu-layers -1 \
  --engine-commit COMMIT --build-flags FLAGS --backend CUDA \
  --hardware HARDWARE --dtype "GGUF UD-Q4_K_S"

python3 gguf-tools/quality-testing/verify_qwen38_run.py \
  /path/to/qwen38-staging/llama-qwen38-001 \
  --manifest gguf-tools/quality-testing/data/qwen38-27b/manifest.json

python3 gguf-tools/quality-testing/compare_qwen38_equivalence.py \
  --manifest gguf-tools/quality-testing/data/qwen38-27b/manifest.json \
  --mode ds4-vs-llama --left-run /path/to/llama-run \
  --right-run /path/to/ds4-run --report /path/to/report.json --top-k 20
```

`generate_qwen38_score.py` and `generate_qwen38_prompt.py` package DS4 scorer
output and long inputs using the same Qwen3.8-only staging contract. The
Qwen3.8 speed-report formats are `ds4-qwen38-speed-v1/v2`; they retain the same
production minimums of 500 prefill and 15 decode tokens/s. Runtime numerical
and speed execution is enabled because every quant format in the pinned UD
artifact now has a CUDA path. Use `DS4_TEST_QWEN38_LIVE=1 make test-qwen38` for
the target/MTP CLI gate and `DS4_TEST_QWEN38_SHA=1` for the separate 16.7 GB
checksum gate. A generated oracle remains `NOT_VERIFIED` until its full corpus
and provenance are reviewed; a one-case diagnostic is not a promotion.
## 6. Validate regression artifacts

Two standard-library Python tools check saved artifacts without running inference. For matched `score_official` runs with API logprob coverage, require the complete intended manifest and finite, consistent counts and scores:

```sh
python3 gguf-tools/quality-testing/validate_scores.py \
  --manifest gguf-tools/quality-testing/data/flash/manifest.tsv \
  --baseline /tmp/old.tsv --candidate /tmp/new.tsv --strict-identical > /tmp/scores.json
```

Exit 0 with `--strict-identical` requires every per-case value to match; omit it to report valid differences without deciding their acceptability. Unlike `compare_scores.py`, validation rejects missing/duplicate cases, malformed fields and changed coverage denominators. This stricter mode requires the current four-column manifest and scorer TSV format with nonzero API coverage; use the existing comparator for continuation-only references without API logprobs.

Compare complete vocabulary dumps from `ds4-bench --dump-frontier-logits-dir`:

```sh
python3 gguf-tools/quality-testing/compare_frontier_logits.py \
  /tmp/old-logits /tmp/new-logits --frontiers 2048 4096 --ctx 8192 \
  --model /models/model.gguf --backend rocm --quality false --quant-bits 2 --vocab 129280 \
  --output /tmp/frontiers.json
```

Supply the exact expected metadata: `ctx` is allocated capacity; each frontier is total context, and prefill is its increase from the previous frontier (initially zero). Each directory must contain exactly the requested dumps. Exit 0 requires complete finite float32 bit identity; exit 1 reports drift, invalid artifacts or an operational error. JSON distinguishes `identical`, `drift` and `invalid`, with full-vector hashes, differing counts and maximum absolute differences. The output path must be new. The parser follows the writer's nine-significant-digit float format and preserves signed zero.

Matching argmax is insufficient: a measured 8K–64K comparison retained all four argmax IDs while full vectors differed, with maximum absolute difference 7.45228. This is numerical drift, not by itself proof of worse generation quality. Record prompt/weight hashes, source/build identity, settings and successful run completion separately; these tools cannot recover that provenance from score tables or dumps, and do not replace model-quality checks.

Run the host-only positive and negative controls:

```sh
python3 -m unittest discover -s gguf-tools/quality-testing/tests -v
```

## 7. Qwen3.8 Spark Baseline

Measured on 2026-09-15, using one DGX Spark per quant. The short set has 100
cases and 5,696 target tokens; the long set has 12 cases and 766 target tokens,
with prefixes from 1,971 to 23,985 tokens. These are ordinary decoding and
prefill checks, without MTP or vision.

| Suite | GGUF | API top-token agreement | NLL, default | NLL, `--quality` |
| --- | --- | ---: | ---: | ---: |
| Short | Q2 | 88.90% | 0.352980 | 0.352915 |
| Short | Q4 | 92.08% | 0.290297 | 0.290538 |
| Long | Q2 | 93.99% | 0.154199 | 0.154106 |
| Long | Q4 | 97.26% | 0.124611 | 0.125546 |

Agreement is measured after feeding the same official continuation prefix,
not as a percentage of correct answers. The two short cases with damaged
emoji metadata contribute to NLL but not API agreement, which covers 5,568
short-set tokens. All 766 long-set tokens align. Every local prompt and target
token ID was also checked against Qwen's published tokenizer.

The default fast path shows no material loss relative to `--quality` in these
sets. The paired case-bootstrap intervals include zero for both quants and
both suites. Q4 is closer to the hosted reference than Q2 on average.
These measurements do not prove identical native weights or correctness at
every context length.

Leaving the last 1 or 256 prompt tokens for a separate continued prefill
changed aggregate NLL by less than 0.0008 for either quant. Independent CLI
answers at about 2K and 24K context also retrieved the correct archive owner
and revision, and gave the correct clamp outputs and boundary tests, for
both Q2 and Q4.

Raw per-case scores, weight/source hashes and comparison details are retained
in [short results](data/qwen38-flash-alibaba-100/cuda-spark-20260915/results.json)
and [long results](data/qwen38-flash-alibaba-long/cuda-spark-20260915/results.json).
