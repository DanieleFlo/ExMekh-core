Contributing to ExMekh Core

Thank you for your interest in contributing to ExMekh Core.

ExMekh is an independently maintained fork of DwarfStar (DS4). It preserves DS4’s preference for small, carefully engineered, model-specific inference paths while extending the system toward capable, multimodal local agents on consumer hardware.

The primary development target is an NVIDIA RTX 3090 with 24 GB of VRAM. Contributions for other hardware are welcome, but their validation status must be documented separately.

Project principles

ExMekh has two connected objectives:

1. Make long-running agent workflows practical with a limited active context. Hierarchical skills, isolated histories, and inference checkpoints should reduce unnecessary context accumulation without silently losing information needed by the parent agent.
2. Extend agent capabilities across modalities. The goal is a small number of well-engineered models for language, vision, image generation, embeddings, and eventually additional media capabilities—not superficial support for every available model.

Contributions should preserve these principles:

* Correctness before performance. Do not accept unexplained numerical drift, altered model behavior, or broken checkpoint semantics in exchange for higher throughput.
* Measure complete workflows. Tokens per second matter, but so do task success, peak VRAM, latency, context pressure, and recovery after agent or native-media operations.
* Respect component boundaries. Agent policy belongs to ExMekh-Lib; frontend interaction belongs to ExMekh Agent; inference state and model execution belong to ExMekh Core.
* Prefer explicit resource ownership. GPU memory, model residency, temporary buffers, sessions, and cancellation paths must have clear lifecycles.
* Keep implementation focused. Prefer readable, specialized implementations over large compatibility layers or unqualified generic abstractions.
* Distinguish evidence from expectations. A build, unit test, live inference test, quality evaluation, and release qualification are different results.

Areas of contribution

Inference engine and accelerator backends

Relevant areas include C inference, model-specific execution, quantized kernels, CUDA, Metal, ROCm, tokenization, speculative decoding, GGUF handling, KV caches, and server APIs.

Before changing a backend, identify the model layouts and platforms affected. Preserve unchanged paths, including those that cannot be exercised on your local hardware.

Do not assume that an optimization for one quantization, GPU architecture, or model family transfers to another.

Agentic execution and context management

Contributions may improve skill checkpointing, nested execution, allowed-tool decoding, Responses API integration, session isolation, context restoration, or error recovery.

Tests should cover both normal and adversarial lifecycle transitions:

* Parent-to-child checkpoint and exact parent restoration.
* Nested skill execution and correct return values.
* Cancellation, timeout, malformed input, and rejected tool calls.
* Session isolation and corrupted or incompatible checkpoints.
* Long workflows with compact results and unfavorable workflows requiring substantial shared context.

An agent-context optimization should be evaluated for information retained and task completion, not only tokens removed.

ExMekh Agent and ExMekh-Lib integration

The frontend lives under agent/ and uses the separately developed ExMekh-Lib package.

Contributions should preserve project and chat persistence, the execution archive, tool approvals, streaming, message editing, attachments, and recovery after interruption.

ExMekh-Lib is distributed here through a versioned wheel. Changes to its behavior must be coordinated with its source repository and the wheel synchronization process. Do not replace a packaged wheel without recording the corresponding source revision.

Native multimodal suite

The native suite covers specialized media and embedding operations alongside language inference.

Changes should explicitly test model admission, VRAM allocation, cleanup, cancellation, and restoration of the language-model session. Model loading and functional HTTP responses alone do not establish output quality.

Changes involving image, audio, video, or embedding preprocessing must preserve model-specific numerical and media-processing conventions and cite their reference implementations.

Documentation and reproducibility

Documentation improvements, minimal bug reproductions, portability reports, benchmark harnesses, and negative experimental results are valuable contributions.

The technical entry points are:

* Project documentation
* Agentic API contract
* Frontend guide
* Native suite status
* Performance references

Before opening a pull request

Describe:

1. Problem: What fails, under which conditions, or which capability is missing?
2. Scope: Which files, model families, platforms, and API contracts change?
3. Approach: Why is the proposed solution appropriate for ExMekh’s architecture?
4. Correctness: Which tests and numerical comparisons passed?
5. Performance: What changes in speed, memory, or end-to-end workflow behavior?
6. Provenance: Which external implementations, papers, or repositories informed the work?
7. Limitations: What remains untested, unsupported, slower, or potentially incompatible?

Provide the commit revision, GPU, driver, runtime/toolchain, model artifact and quantization, context size, and relevant configuration.

A result measured on one configuration must not be presented as universal.

Build and regression checks

Start with the checks relevant to the changed component.

For the native engine:

make clean
make
make test

For the locally developed CUDA path:

make cuda CUDA_ARCH=sm_86
make cuda-regression

For relevant model-specific changes:

make test-qwen38

For documentation:

make docs-check

For frontend tests:

cd agent
uv run --frozen pytest

Some tests require large model artifacts, specific accelerators, live services, or explicit environment configuration. Report skipped checks as not verified, not passed.

For focused DS4 regressions, the inherited test runner provides checks including:

./ds4_test --server
./ds4_test --logprob-vectors
./ds4_test --long-context
./ds4_test --tool-call-quality

Backend-specific tests, such as Metal kernel checks, must run on suitable hardware.

Quantization and model-quality changes

A change that affects quantization, attention, logits, tokenization, or model execution needs a correctness comparison against an appropriate baseline.

For supported official-continuation evaluations, the repository retains the scorer under gguf-tools/quality-testing:

make -C gguf-tools quality-score

Use the same manifest, model inputs, token limits, and evaluation settings for old and new implementations.

Do not weaken numerical tolerances or update reference outputs merely to make a regression disappear. Document intentional numerical differences and why they are acceptable.

Performance changes

Performance comparisons should use the same machine and workload, with consistent:

* Model artifact, quantization, and context length.
* GPU memory budget, backend, and kernel configuration.
* Warm-up, sampling method, and number of repetitions.
* CPU/GPU utilization, thermal state, and competing workloads.
* Exact baseline and candidate revisions.

For inference throughput, use the repository’s ds4-bench and relevant model-specific performance harnesses.

For agentic or multimodal changes, also report end-to-end latency, task success, peak VRAM, model loading and restoration costs, and context-management overhead.

A faster isolated kernel is not automatically a faster agent workflow.

Negative results and experiments that fail numerical gates should be documented rather than silently omitted.

External sources, inspiration, and attribution

ExMekh builds on upstream implementations and studies other projects to inform engineering decisions.

Significant external influences must be credited where they affect the implementation, not only in a general acknowledgment list.

For any external idea or implementation reference, document:

* Project name and canonical repository URL.
* Exact source commit or release, where available.
* Relevant source files, documentation, or paper.
* Whether the contribution is a conceptual inspiration, independently reimplemented technique, adapted implementation, copied component, or runtime dependency.
* Applicable copyright and license obligations.
* Differences between the external reference and ExMekh’s implementation.
* Local correctness and performance validation.

In particular, Strata has informed Qwen Flash engineering work involving expert residency, CPU/GPU scheduling, PCIe staging, prefill, and CUDA execution. The documented experiments describe independently implemented approaches, not wholesale importation of Strata kernels. See the Strata performance reports under docs/performance/.

Other important source and reference projects include DwarfStar, llama.cpp/GGML, Hugging Face Transformers and Diffusers, PyTorch, NumPy, Pillow, FFmpeg, and libsoxr. Their relationships to this repository differ; consult third_party/ for component-specific notices.

Research references such as vLLM, Marlin, and NVIDIA GatedDeltaNet should be cited when a particular contribution derives an idea from them. Merely studying a repository does not imply its code has been incorporated.

Contributors must have the right to submit any external material. Preserve required copyright statements, licenses, source references, and redistribution conditions. Never remove notices from inherited or adapted files.

Issues, traces, and security

A useful bug report includes a minimal reproducer, expected and actual behavior, exact revision, model and backend information, and a relevant trace.

For server-side inference debugging, the inherited trace facility can be useful:

./ds4-server --trace /tmp/ds4-trace.txt ...

Review traces and attached files before publishing them. Remove secrets, private prompts, local paths, credentials, access tokens, and confidential user data.

Report potentially exploitable vulnerabilities privately rather than publishing a working exploit in a public issue.

Upstream collaboration

When a bug or optimization also applies to DwarfStar without relying on ExMekh-specific agent or multimodal behavior, consider contributing the general fix upstream.

ExMekh is intended to remain technically compatible with useful upstream developments wherever practical. Fork-specific changes should have a clear reason to exist.

Licensing

Contributions to the repository must comply with its LICENSE and any additional notices applying to inherited, adapted, or third-party components.

AI-assisted development is welcome, but contributors remain responsible for correctness, provenance, licensing, reproducibility, and the final submitted changes.
