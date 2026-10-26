<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="180">
</p>

# ExMekh Core

**Consumer-hardware inference for capable, multimodal local AI agents.**

ExMekh Core is an independently maintained fork of [DwarfStar (DS4)](https://github.com/antirez/ds4). It does **not** aim to move away from DS4's philosophy: a compact native inference engine, deliberately optimized for a small set of well-understood open-weight models. It asks what that philosophy can enable when the target is **an entire agentic experience on widely available consumer hardware**, particularly an **NVIDIA RTX 3090 with 24 GB of VRAM**.

The project has two connected goals:

1. **Make long-running agent workflows practical with a relatively small active context.** Hierarchical skills and subagents compartmentalize execution history. A parent can delegate a task, suspend its inference state, and resume with the relevant result instead of permanently absorbing every intermediate tool call and token from the child. This can reduce KV-cache pressure and leave more of a fixed VRAM budget available for a higher-quality, less aggressively quantized LLM.
2. **Make local agents genuinely multimodal.** Rather than trying to support every model of one type, ExMekh extends DS4's model-specific engineering to **selected models of different types**: language, vision, image generation/editing, and multimodal embeddings today; video and audio generation are longer-term directions, **not current shipping capabilities**.

The objective is not the largest context window, the most models, or the highest isolated tokens-per-second score. It is **the most capable and reliable end-to-end agent workflow that can fit within a consumer machine's actual compute, memory, and latency budget**.

> **Status:** Research and active development. Implementation, functional tests, output quality, performance, and release certification are distinct milestones. Supported code paths are not necessarily validated on every platform.

## Why context structure matters on a 24 GB GPU

For local agents, memory is spent on more than model weights: the KV cache grows with active context, and other models or media operations need weights and working buffers of their own. A straightforward response is to quantize the LLM more aggressively or increase hardware capacity. ExMekh explores a complementary approach: **reduce how much history must be active at once**.

A conventional linear agent transcript can accumulate the entire sequence of planning, tool calls, searches, intermediate failures, and results. Yet many of those details matter only while a particular subtask is being executed.

ExMekh uses the **workflow boundary as a context boundary**:

```text
Main conversation (small, valuable active context)
  |
  +-- Delegate to skill A
  |     +-- local prompts, tool calls, intermediate attempts
  |     +-- optional nested skills
  |     '-- return the result the parent needs
  |
  +-- Continue with the result, not A's entire private transcript
  |
  +-- Delegate to skill B
  |     '-- separate execution history and result
  |
  '-- Continue the same larger task
```

At a skill boundary, the engine can checkpoint the caller's inference frontier; the child executes in its own context; the caller is restored and receives the child's result. **This is compartmentalization, not an assertion that information can be discarded without consequences.** Persistent project/chat history, execution archives, and model-visible active context have different roles.

### The trade-off, explicitly

This design can be effective when a task decomposes into relatively independent subtasks with compact, sufficient outputs. It can be unfavorable when a task requires global, simultaneous access to large amounts of information, when skill outputs become long, or when information discarded at a boundary later proves essential. Skill execution, checkpointing, summarization, and reloading also carry costs.

**ExMekh does not claim to solve long-context reasoning or guarantee that a smaller window is always better.** Its hypothesis is testable: *for suitable real agent workloads, can a better-preserved model with a smaller active context and hierarchical history outperform a more heavily quantized model or a flatter workflow under the same 24 GB budget?*

Relevant comparisons should include task success, VRAM peak, model quantization, active tokens, compaction frequency, information loss, end-to-end latency, and recovery after a child task—not just tokens per second.

## A model-specific engine, extended across modalities

DS4's core design principle remains valuable: **engineer a limited number of model paths deeply, rather than advertise broad compatibility with shallow guarantees**. ExMekh applies that principle to the capabilities an agent needs.

| Capability | Direction | Status |
| --- | --- | --- |
| Language, reasoning, coding, tool use | Qwen3.8-27B as the principal RTX 3090 development profile; other DS4 families retain separate paths | Implemented; verification depends on model and hardware |
| Vision input | Matching model-specific visual encoder integrated into the chat workflow | Implemented on documented paths |
| Image generation and editing | Native Qwen-Image-2.1-Turbo suite, accessible to the agent/chat | Functional path implemented; final output-quality and release gates remain open |
| Semantic and multimodal retrieval | Native EmbeddingGemma-2 embeddings, including supported media inputs | Functional path implemented; qualification is workload-specific |
| Video generation | Future specialized model and agent workflow | **Roadmap; not implemented as a shipping generator** |
| Audio generation | Future specialized model and agent workflow | **Roadmap; not implemented as a shipping generator** |

This is not a claim that a single 24 GB GPU can keep all these models resident simultaneously. **Model admission, GPU memory ownership, parking/unloading, and restoration of the language model are first-class runtime concerns.** Multimodal quality and interactive latency must be measured across the complete user workflow.

## Relationship to upstream DwarfStar

ExMekh is a **specialization of DS4's philosophy**, not an attempt to replace its inference core or become a general-purpose GGUF runtime. The most substantial divergence is at the **agent boundary**: upstream's integrated native coding agent gives way to a separate hierarchical agent runtime, browser frontend, and an explicit API contract for stateful execution.

| Area | DwarfStar upstream | ExMekh Core |
| --- | --- | --- |
| Engineering philosophy | Fast, small, model-specific native inference | Preserved; applied to consumer agent workflows and selected modalities |
| Primary optimization target | Efficient model inference and native coding-agent execution | End-to-end agent capability under a constrained VRAM budget |
| Agent architecture | Integrated native agent | **ExMekh Agent** frontend + **ExMekh-Lib** orchestration + DS4-derived inference backend |
| Long workflows | Upstream session/KV mechanisms | Skill-scoped history, context identities, checkpoints and returns; favorable and unfavorable cases acknowledged |
| Hardware emphasis | Multiple accelerator/model configurations | Locally audited RTX 3090 (`sm_86`) path, without claiming exclusive hardware support |
| Modalities | Upstream model-specific paths | Additional native image-generation/editing and embedding suite, with future audio/video ambitions |
| Evidence | Upstream tests and benchmarks | Explicit local compatibility matrices, numerical gates, checkpoint audits and open qualification items |

The separation adds API and orchestration overhead relative to an integrated native agent. It is a deliberate trade-off for clearer state ownership, inspectability, extensibility, and frontend independence—not a promise of universally faster inference. Valuable upstream model and kernel improvements continue to be incorporated; imported code is **not automatically locally qualified**. See the [upstream merge audit](docs/architecture/ds4-upstream-merge-20261001.md).

## Architecture

```text
                      ExMekh Agent
                 NiceGUI · projects · chat
              graph · approvals · execution log
                            |
                      ExMekh-Lib
           DeepAgent · HDS · tools · skills
           subagents · history · compaction
                            |
              Responses API + agentic contract
             context IDs · tool/skill allowlists
                            |
                       ExMekh Core
              DS4 server / inference engine
       tokenizer · grammar · KV · MTP · checkpoints
                            |
                  Native compute/runtime
                  CUDA · Metal · ROCm*
                            |
        Qwen3.8-27B / Qwen Flash Next / DS4 families

        Optional native suite on qualified CUDA path:
          image generation / editing / embeddings
          resource planning · park/restore · HTTP

* Backend availability and local verification differ.
```

Three boundaries matter:

1. **Agent policy belongs above the engine.** ExMekh-Lib decides which skill or subagent runs, which tools it can see, and what its result means. The inference engine receives resolved permissions and an opaque context identity; it does not invent the hierarchy.
2. **Inference state belongs below the API.** Token history, KV state, recurrent state, speculative-decoding state, and model-specific prompt rendering have to remain internally consistent. Skill returns restore the caller's frontier and append the skill result, rather than treating the child's private reasoning as the caller's history.
3. **Resource ownership must be explicit.** A single consumer GPU has finite memory. Native image and embedding operations have their own weights and temporary buffers; the runtime plans admission, releases resources, and restores the LLM before continuing a suspended session.

### Hierarchical execution is a state transition

A nested skill is not just more text in a chat:

```text
Parent context
  ├─ assistant calls a skill     ← checkpoint the parent frontier
  ├─ child skill executes        ← child-local history and tools
  │    └─ may call another skill
  └─ skill returns a result      ← restore parent + append result

Parent continues without inheriting the child's private transcript.
```

The server's `agentic` extension carries `agent_context_id`, `allowed_tools`, and `allowed_skills`. Tool names are constrained **during decoding**, not merely rejected after the model has already generated an unauthorized name. Schema validation and fail-closed handling remain separate parts of the contract. Checkpoints are tied to their session and model layout and are validated before restore.

This is a controlled execution protocol, **not a claim that language-model outputs are inherently safe or always correct**. Private suspended skill frames are process/session-scoped; restarting the server does not magically restore an unfinished private call stack. Durable chat history and compatible disk KV caches have different lifecycles.

Read the [agentic API contract](docs/agentic/agent-api.md) and [frontend integration](docs/agentic/frontend.md).

## What you can use

**ExMekh Agent** is a browser-based local workspace built with NiceGUI. It provides persistent projects and chats, Quick Chats, streaming text and reasoning, execution graphs, tool/skill result inspection, approvals, message editing and regeneration, steering during an active turn, image/PDF inputs, and a visual editor for agent/skill/subagent capabilities. Long histories can be compacted for subsequent model calls while keeping a separately inspectable execution archive.

**ExMekh-Lib** supplies the agent behavior and typed client integration. The frontend installs it from the versioned wheel under `agent/wheels/`; the library's development sources are not part of this repository's tracked source tree. See [the frontend README](agent/README.md) for package synchronization and the full UI feature set.

**The DS4-derived engine** provides model-specific GGUF validation, CPU reference paths, accelerator backends, HTTP serving, speculative decoding where supported, structured output/tool-call constraints, KV reuse, and session checkpoints. Qwen3.8-27B (`qwen35`) and Qwen3.8 Flash Next (`qwen4exp`) have **different loaders, graphs, tokenization and tool dialects**. They are not interchangeable implementations under one name.

**The native suite** adds an explicit `ds4-server-suite` target for image generation/editing and text/image/video/audio embeddings, served alongside chat through a single `/v1` base URL. The implemented local profiles use Qwen-Image-2.1-Turbo and EmbeddingGemma-2. The C/CUDA runtime does not launch a separate Python inference server. Offline preparation of weights and assets is still required; image quality/release gates remain open. See [suite setup and status](docs/architecture/native-suite-release.md).

## Models, hardware, and verification

| Path | What this repository establishes |
| --- | --- |
| **Qwen3.8-27B UD-Q4_K_S**, CUDA `sm_86` | Primary, pinned and locally audited path; external MTP sidecar, tool/JSON constraints, checkpoints, and matching vision encoder have dedicated validation |
| **Qwen3.8 Flash Next**, `qwen4exp` | Separate upstream-derived architecture; local functional CUDA checks exist, but hardware cost and speed differ considerably from the 27B path |
| **DeepSeek V4/V4.1, GLM 5.2/5.3, other imported DS4 families** | Integrated upstream functionality; do **not** infer complete local inference qualification from a successful build |
| **Native image / embedding suite**, RTX 3090 | Implemented and functionally exercised on single-GPU CUDA; output quality, optimization, and production release qualification must be assessed separately |
| **Metal, ROCm, distributed and multi-GPU paths** | Upstream-derived support exists; this fork's principal local verification environment does not cover all these paths |

The audited local system uses an **RTX 3090 24 GB with CUDA `sm_86`, Linux/WSL2, and host RAM/SSD to complement VRAM**. That is a development target, **not a universal minimum specification**. The supported model formats are deliberate: this project is **not a drop-in runner for arbitrary GGUF files**.

For exact artifact hashes and constraints, consult [Qwen3.8 compatibility](docs/architecture/qwen38-compatibility.md), [the upstream integration report](docs/architecture/ds4-upstream-merge-20261001.md), and [native suite qualification](docs/performance/native-suite-t19-t22-20261022/README.md).

## Quick start: Qwen3.8-27B + ExMekh Agent

The simplest path starts the **LLM-only server** first. This avoids assuming that the separate native image/embedding bundles have already been prepared.

Prerequisites: a supported Linux/WSL CUDA environment, enough free disk space for the pinned GGUF and MTP artifacts, a working C/CUDA toolchain, Python 3.13, and [`uv`](https://docs.astral.sh/uv/). The commands below are intended for the locally developed RTX 3090 `sm_86` profile; other platforms use their documented build paths.

```bash
git clone https://github.com/DanieleFlo/ExMekh-core.git
cd ExMekh-core

# Audited Qwen3.8-27B GGUF plus matching MTP sidecar.
./download_model.sh qwen38-q4-mtp

# Build the CUDA path used in local validation.
make cuda CUDA_ARCH=sm_86
```

Start the LLM-only server in terminal 1:

```bash
./start-ds4-server.sh
```

Start the frontend in terminal 2:

```bash
cp agent/.env.example agent/.env
./agent/run.sh
```

The LLM API is served at `http://127.0.0.1:8080/v1`; the frontend opens at **http://127.0.0.1:8000**. `agent/.env.example` selects the Responses API and the Qwen3.8-27B model by default. The root launcher uses the 27B profile unless configured otherwise.

For local vision input, obtain the **matching** encoder:

```bash
./download_model.sh qwen38-27b-vision
```

The launcher selects that encoder when the expected file is present. Do not pair it with the different Flash Next vision encoder. See [vision verification](docs/architecture/qwen27-vision-checkpoint-20261001.md).

### Enabling native image generation and embeddings

This is an **additional setup**, not a property of the basic LLM server. Prepare the pinned image and embedding bundles, native media/CUDA dependencies, and their configuration by following [Native suite: build, streaming and acceptance](docs/architecture/native-suite-release.md). The example suite configuration contains **local development paths** and must be adapted to your machine.

Once those prerequisites are satisfied:

```bash
cp config/native-suite.env.example config/native-suite.env
# Configure native library paths and the suite's model assets.
./agent/start-ds4-server.sh
./agent/run.sh
```

The agent launcher selects `ds4-server-suite`, with Qwen27, MTP, a 26K context, and the configured native operations on port 8080. Use [`DS4_SUITE_CONFIG`](docs/architecture/native-suite-release.md) for a custom bundle configuration. An LLM-only server **does not** expose `/v1/images/generations`.

For an alternate Flash Next configuration, see `agent/start-ds4-server-flash.sh` and the dedicated Flash Next documentation; it is not a drop-in replacement for the 27B configuration.

## Correctness before performance

This project uses a deliberately narrow standard of evidence:

- **A build is not a model validation.** Loader/graph availability and live inference qualification are separate.
- **Identical behavior matters.** Kernel, MTP, checkpoint, and tool changes are checked against numerical or deterministic baselines appropriate to the operation; thresholds are not silently widened to make a regression pass.
- **The boundary matters.** Tests examine cancellation, corrupted checkpoints, file/tool grammar, nested skill returns, resource cleanup, and recovery after native operations, not only isolated generated text.
- **Missing evidence stays missing.** Reports use results such as `PASS`, `FAIL`, `NOT_VERIFIED`, and `NEED_MORE_DATA`. A successful functional image test is not a final quality or release certification.

Selected entry points:

```bash
make test-qwen38
make docs-check

cd agent
uv run --frozen pytest
```

Some tests need downloaded weights, CUDA hardware, configured external dependencies, or explicit live-test environment variables. Consult the relevant report before interpreting skipped tests. Performance figures belong to their recorded hardware, workloads, revisions, and dates — **not** to every checkout.

## Documentation map

- [Technical documentation index](docs/INDEX.md) — architecture, agentic execution, hardware, experiments, and research provenance.
- [ExMekh Agent](agent/README.md) — frontend capabilities, projects, configuration, and packaged ExMekh-Lib dependency.
- [Agentic protocol](docs/agentic/agent-api.md) — context identities, scoped tool calls, and skill return semantics.
- [Qwen3.8-27B compatibility](docs/architecture/qwen38-compatibility.md) — pinned artifacts, model layout, MTP, tools, and vision.
- [Upstream merge and verification](docs/architecture/ds4-upstream-merge-20261001.md) — source provenance and evidence matrix.
- [Native suite setup](docs/architecture/native-suite-release.md) — image/embedding bundles, shared CUDA runtime, supported operations, and open gates.
- [Original DS4 guides](docs/upstream/_INDEX.md) — inherited model, platform, serving, and performance documentation.

The [technical wiki](docs/INDEX.md) is indexed by subject and section. `make docs-index` regenerates its indexes; `make docs-check` validates coverage and relative links.

## Scope and non-goals

ExMekh Core is not an arbitrary GGUF backend, a guarantee that all upstream hardware paths have been retested here, or a production-certified multimodal server. It does not make model-generated file edits correct merely because their syntax is constrained, and it does not persist unfinished private skill stacks across server restarts.

The aim is a **high-capability local agent on affordable consumer hardware**: preserve model quality where possible by managing active context intelligently, then extend the agent with carefully engineered specialized modalities. Correctness, resource ownership, and honest end-to-end measurements matter more than an isolated tokens-per-second number.

## Upstream, authorship, and license

ExMekh Core builds on **[DwarfStar / DS4](https://github.com/antirez/ds4)** by **Salvatore Sanfilippo (antirez)** and its contributors. The specialized inference philosophy, substantial engine implementation, and many platform/model paths come from upstream. This derivative integrates and modifies them for the ExMekh agent architecture; it is **not an official DwarfStar release**.

DwarfStar in turn acknowledges **[llama.cpp](https://github.com/ggml-org/llama.cpp)** and **GGML** for foundational formats, kernels, ideas, implementation references, and retained/adapted source components. Existing copyright and license notices are preserved; see [`LICENSE`](LICENSE) and copyright notices in the relevant source files. The DwarfStar logo is upstream artwork, not a claim of independent logo authorship.

Development uses AI coding agents extensively for implementation, exploration, and review. That does not replace human design decisions, source provenance, integration testing, or explicit acceptance criteria. The testing notes document **what was actually checked**, rather than treating AI-generated code or an automated pass as proof of universal correctness.
