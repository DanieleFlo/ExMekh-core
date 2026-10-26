<p align="center">
  <img src="logo.svg" alt="DwarfStar (DS4) logo, retained from upstream" width="160">
</p>

# ExMekh Core

**A DwarfStar-based inference engine for capable, multimodal local agents on consumer hardware.**

ExMekh Core is an independently maintained fork of **[DwarfStar (DS4)](https://github.com/antirez/ds4)**. It follows DS4's central idea: **support a small number of models and execution paths exceptionally well**, rather than providing shallow compatibility with every model.

ExMekh extends that approach from *model inference* to *the entire local agent experience*. The reference machine is an **NVIDIA RTX 3090 with 24 GB of VRAM**. The question driving the project is simple:

> **How capable can a local AI agent become when its context, model quality, modalities, and GPU memory are engineered as one system?**

Two design choices shape the answer:

1. **Spend less memory on irrelevant history, not automatically on a smaller or more aggressively quantized model.** Hierarchical skills isolate their intermediate execution histories. The parent agent resumes from a checkpoint and receives the result it needs, instead of accumulating every step of every subtask in its active context.
2. **Choose specialized models for different capabilities.** Language, vision, image generation/editing, and multimodal embeddings are integrated around a common agent workflow. Future audio and video *generation* are goals, not shipping features.

The aim is not a maximum-context benchmark or a model zoo. It is **the strongest reliable end-to-end agent workflow possible within a consumer machine's actual VRAM, RAM, storage, compute, and latency budget**.

> [!IMPORTANT]
> **Research project / active development.** Implemented functionality, passing functional tests, numerical equivalence, output quality, benchmark qualification, and release readiness are separate claims. Read the [validation notes](#validation-performance-and-limitations) before treating a code path as production-ready.

## Why this fork exists

### 1. Longer workflows without keeping every token active

A local agent's context is not just the user's conversation. It can also contain tool calls, file contents, failed attempts, search results, planning, and the private steps of delegated work. A flat transcript makes that context grow with the *process*, even when the parent only needs the *outcome*.

ExMekh treats a **skill boundary as a history boundary**:

```text
Conversation / parent agent
   |
   |-- call skill: inspect the project
   |     |-- read files, search, test hypotheses
   |     |-- call a nested skill if needed
   |     '-- return a concise, useful finding
   |
   |-- continue from the restored parent state + finding
   |
   |-- call skill: implement the change
   |     |-- edit, run checks, handle failures
   |     '-- return changes + verification
   |
   '-- respond to the user
```

The child has its own history. The parent does **not** inherit the child's full private transcript. The inference engine can checkpoint and restore the parent's active state without forcing the full conversation to be prefetched again at every skill transition. Separately, the frontend retains a durable, inspectable record of execution.

This matters on a 24 GB card. Model weights, KV cache, speculative-decoding state, and transient buffers all compete for finite memory. Reducing the *active* context can reduce KV pressure and make room for a **less aggressively quantized language model** or other operations, without claiming that model quality is determined by quantization alone.

**It is not a free long-context solution.** A task that requires simultaneous access to many documents, a long shared reasoning history, or detailed child outputs can defeat the advantage. Information omitted at a skill boundary might be needed later, while skill dispatch, checkpointing, and compaction have their own costs. The approach is most promising when work decomposes into subtasks with sufficient, relatively compact results.

The hypothesis to test is therefore:

> **Under the same hardware budget, can a higher-fidelity model with a smaller active context and compartmentalized agent history complete real tasks better than a more heavily quantized model or a flatter, longer-context agent?**

That requires measuring task success and information loss alongside peak VRAM, token usage, prefill, end-to-end latency, and recovery—not just tokens per second.

### 2. A complete local agent, not just a language-model endpoint

An agent increasingly needs to understand images, search across media, generate or edit images, and operate in a persistent workspace. ExMekh applies DS4's **model-specific engineering** to those distinct capabilities rather than expecting one model to do everything.

| Capability | Current direction | Status |
| --- | --- | --- |
| **Language, reasoning, coding and tools** | Qwen3.8-27B as the main locally audited consumer-GPU profile; other DS4 families have separate paths | Implemented; qualification depends on model/backend |
| **Visual understanding** | Matching vision encoder and image/PDF inputs through the agent | Implemented on documented paths |
| **Image creation and editing** | Native Qwen-Image-2.1-Turbo | Functional path implemented; final image-quality and release gates **open** |
| **Text and multimodal embeddings** | Native EmbeddingGemma-2 for supported text, image, video and audio inputs | Functional path implemented; workload-specific qualification required |
| **Video generation** | Dedicated model and workflow | **Future direction, not shipping** |
| **Audio generation** | Dedicated model and workflow | **Future direction, not shipping** |

Embedding a video or audio input **does not mean ExMekh can generate video or audio**. Similarly, a working image-generation endpoint is not proof that its visual quality is already release-qualified.

On a single RTX 3090, not every model can remain resident simultaneously. Consequently, **model admission, memory ownership, loading, parking, cleanup, and LLM restoration** are core parts of the design, not implementation details hidden behind an API.

## Relationship to DwarfStar

ExMekh is **not trying to replace DS4 with a generic inference framework**. The DS4-derived native engine remains the foundation; upstream model support and improvements can continue to be integrated.

The main departure is the agent architecture:

| | DwarfStar upstream | ExMekh Core |
| --- | --- | --- |
| **Core philosophy** | Compact, fast, model-specific native inference | Preserved and applied to complete local agent workflows |
| **Agent boundary** | Integrated native coding agent | Separate ExMekh Agent UI and ExMekh-Lib runtime, with an explicit inference API |
| **Agent history** | Upstream session and caching mechanisms | Skill-scoped histories, explicit context identities, checkpoint/return semantics |
| **Resource target** | Multiple model and accelerator configurations | Consumer-GPU workflows, especially the locally audited RTX 3090 path |
| **Modalities** | Upstream model-specific support | Additional native image-generation/editing and embedding suite |
| **Evidence** | Upstream benchmarks and tests | Explicit local compatibility audits, numerical checks, and recorded open gates |

A separate agent runtime introduces orchestration and API overhead compared with an integrated native agent. ExMekh accepts that trade-off for **clearer state ownership, composable skills, a persistent UI, and independent development of agent behavior and inference**. It does not promise to be universally faster than upstream DS4.

See the [upstream merge and compatibility audit](docs/architecture/ds4-upstream-merge-20261001.md).

## Architecture

```text
                       ExMekh Agent
                 NiceGUI browser workspace
           projects · chat · files · approvals
              execution graph · history
                             |
                        ExMekh-Lib
              DeepAgent · skills · subagents
             tools · history · compaction
                             |
             OpenAI-compatible Responses API
             + optional `agentic` extension
                             |
                         ExMekh Core
                    DS4-derived C engine
        model graphs · constrained decoding · KV
            MTP · checkpoints · HTTP serving
                             |
                 Native model/runtime paths
           CUDA · Metal · ROCm (verification varies)
                    |                 |
                 Language        Native suite
                 + vision        image · embeddings
                                 GPU park / restore
```

### Three ownership boundaries

**ExMekh Agent — user experience.** The browser frontend provides projects and persistent chats, streaming output, visual tool/skill graphs, approval requests, editable messages, steering, image/PDF attachments, generated-image previews, capability editing, and a separate inspectable execution archive. It uses [NiceGUI](https://github.com/zauberzeug/nicegui).

**ExMekh-Lib — agent semantics.** The separately maintained Python library determines which agent or skill runs, its instructions, which tools and subskills are available, how histories are compartmentalized, and what result returns to the caller. This repository distributes a versioned wheel under `agent/wheels/`; its complete development source is not tracked here.

**ExMekh Core — inference and state.** The native server executes the model, constrains allowed tool names at decoding time, maintains active KV and model-specific state, and checkpoints/restores inference contexts. The engine receives resolved capabilities; it does not infer the skill hierarchy.

The optional **native multimodal suite**, built as `ds4-server-suite`, integrates image generation/editing and embeddings into the same local serving environment. It uses C/CUDA for inference and does not start a separate Python model-serving process. Offline model preparation and native media libraries remain necessary.

### How the agentic API works

ExMekh extends an OpenAI-compatible request with an optional `agentic` object:

```json
{
  "agentic": {
    "agent_context_id": "run-42:root",
    "allowed_tools": ["read-file", "execute"],
    "allowed_skills": ["inspect-project"]
  }
}
```

The example shows the **extension fragment**, not a complete request. In the full request, the tool registry is declared separately and remains stable for the context; allowed tools and skills describe the capabilities currently enabled by the agent runtime. The server treats the context ID as opaque.

Allowed names are enforced during constrained decoding, not only rejected after generation. A nested skill can suspend the parent, work in its own context, and return a result to the restored parent. Checkpoints must preserve the relevant model state, including architecture-specific KV, recurrent, and speculative-decoding state.

Tool masking is **not a guarantee of semantically safe outputs**. Corrupted checkpoints, cancellation, invalid schemas, unsupported model layouts, and restart behavior still require explicit validation. Private suspended skill frames are session/process scoped; durable chat archives are not a serialized private call stack.

Read the [agentic contract](docs/agentic/agent-api.md) and [frontend integration notes](docs/agentic/frontend.md).

## Supported paths and hardware

**The primary development machine** is a single RTX 3090 (24 GB, CUDA `sm_86`) on Linux/WSL2, with system RAM and SSD used where a model or cache cannot fit entirely in VRAM. This is a **reference environment, not a claim that every feature runs on every 24 GB GPU**.

| Model or backend | What has been established |
| --- | --- |
| **Qwen3.8-27B UD-Q4_K_S** (`qwen35`) | Pinned GGUF and external MTP sidecar; local CUDA validation of model-specific inference, tool/API constraints, checkpointing and a matching vision path |
| **Qwen3.8 Flash Next** (`qwen4exp`) | Separate architecture and protocol, with locally tested CUDA paths and specialized expert/weight/KV handling; resource and performance characteristics differ substantially from 27B |
| **DeepSeek V4/V4.1, GLM 5.2/5.3 and other DS4 families** | Upstream-derived implementations are integrated; successful compilation does **not** certify local full-model inference |
| **Qwen-Image-2.1-Turbo / EmbeddingGemma-2** | Native single-GPU CUDA functional and resource-lifecycle work; image output quality and production release qualification remain open |
| **Metal, ROCm, distributed and multi-GPU** | Upstream-derived paths; not all have been independently validated on the fork's development machine |

Qwen3.8-27B and Flash Next use **different model loaders, tensor graphs, tokenization/tool dialects, and memory behavior**. Do not interchange their GGUF files, MTP configurations, or vision encoders. ExMekh is **not** a universal runner for arbitrary GGUF files.

See [Qwen3.8 compatibility and artifact hashes](docs/architecture/qwen38-compatibility.md), [Flash Next guide](docs/upstream/QWEN38_FLASH_NEXT.md), and the [native suite status](docs/architecture/native-suite-release.md).

## Getting started

The steps below target the audited **Qwen3.8-27B / CUDA `sm_86`** development profile. Other backends or operating systems require their own build and validation procedures.

### Requirements

- A compatible Linux or WSL2 environment, NVIDIA drivers, CUDA development tools, and sufficient system RAM and disk space for model artifacts.
- A supported native C/CUDA build toolchain.
- **Python 3.13** and [`uv`](https://docs.astral.sh/uv/) for ExMekh Agent.
- The selected, verified model files; these are downloaded separately from the source repository.

### 1. Build and download the language model

```bash
git clone https://github.com/DanieleFlo/ExMekh-core.git
cd ExMekh-core

# Pinned 27B GGUF and matching MTP sidecar.
./download_model.sh qwen38-q4-mtp

# CUDA target used for the RTX 3090 validation.
make cuda CUDA_ARCH=sm_86
```

The downloader verifies pinned artifacts as documented in the [model compatibility audit](docs/architecture/qwen38-compatibility.md).

### 2. Start the LLM-only server

In the first terminal:

```bash
./start-ds4-server.sh
```

The default `qwen27` profile serves the model on **`http://127.0.0.1:8080/v1`**. This launcher supports additional configuration through environment variables; it does **not** enable the native image/embedding suite unless configured for that purpose.

### 3. Start ExMekh Agent

In another terminal:

```bash
cp agent/.env.example agent/.env
./agent/run.sh
```

Open **http://127.0.0.1:8000**. The example environment uses the Responses API and points to the local DS4 server. The launcher installs/runs the pinned frontend environment through `uv`.

For image inputs to Qwen3.8-27B, download its **matching** vision encoder:

```bash
./download_model.sh qwen38-27b-vision
```

The root launcher automatically selects it when the expected file exists. See [vision and checkpoint qualification](docs/architecture/qwen27-vision-checkpoint-20261001.md).

### 4. Add native image generation and embeddings (optional)

This is an **additional build and model-preparation step**. Prepare the pinned Qwen-Image-2.1-Turbo and EmbeddingGemma-2 bundles, tokenizer/processor assets, native libraries, and suite configuration as described in the [native suite setup and release guide](docs/architecture/native-suite-release.md).

```bash
cp config/native-suite.env.example config/native-suite.env
# Edit the native dependency paths and prepare the suite bundles/config.

# Starts the Qwen27 profile with ds4-server-suite.
./agent/start-ds4-server.sh

# In a second terminal:
./agent/run.sh
```

The suite uses the same `/v1` base URL for chat and native media operations, but requires its own model assets and configuration. **The LLM-only server does not provide image-generation endpoints.** The example suite paths must be adapted to the local machine. The agent launcher uses a **26K** context default, different from the root LLM-only launcher's default.

For Qwen Flash Next, use the separately documented [`agent/start-ds4-server-flash.sh`](agent/start-ds4-server-flash.sh) and its pinned artifacts; it is **not** a drop-in switch from the 27B setup.

## Validation, performance, and limitations

ExMekh's acceptance standard is deliberately evidence-driven:

- **Build ≠ correctness.** Compiling a kernel or loading a graph does not establish numerical fidelity or live model behavior.
- **Functional ≠ release-qualified.** Successful image/embedding requests and CUDA resource tests do not close open visual-quality or release gates.
- **Reference behavior matters.** Model-specific logits, attention/KV state, checkpoint returns, speculative decoding, and media preprocessing are checked against appropriate baselines rather than silently relaxing tolerances.
- **Whole workflows matter.** Benchmark task success, context loss, model loading/parking, peak VRAM, cancellation and restoration, as well as prefill and generation throughput.
- **Unverified remains unverified.** Missing hardware, weights, tests, or reproducible measurements are not presented as passing results.

Useful local entry points include:

```bash
make test
make test-qwen38
make docs-check

# Frontend checks:
cd agent
uv run --frozen pytest
```

Some checks require hardware, large model downloads, native library paths, live-test opt-ins, or other prerequisites. Consult the specific report before interpreting skips or failures. Do not compare throughput across different model artifacts, contexts, GPU power states, or unrecorded revisions.

The [native suite T19–T22 report](docs/performance/native-suite-t19-t22-20261022/README.md) documents functional and lifecycle checks, along with **open image-quality/release gates and performance measurements that are not yet qualified for general claims**. The [Strata-inspired Qwen Flash studies](docs/performance/qwen-flash-strata-20261006.md) record both passing gates and experiments that were rejected or left disabled.

### Known non-goals

ExMekh does not claim to eliminate the need for long context, guarantee perfect information preservation across skill boundaries, or keep every multimodal model resident on a single GPU. It is not a production-certified general-purpose multimodal server, and constrained tool names do not make model-produced file changes automatically correct or safe.

## Documentation

| Resource | Contents |
| --- | --- |
| [Technical index](docs/INDEX.md) | Architecture, experiments, agentic protocol, hardware, performance and research |
| [ExMekh Agent](agent/README.md) | UI, projects, chat, tools, image outputs and packaged ExMekh-Lib integration |
| [Agentic API](docs/agentic/agent-api.md) | Context identities, tool/skill allowlists, checkpoint and return semantics |
| [Qwen3.8-27B](docs/architecture/qwen38-compatibility.md) | Model artifact hashes, MTP, tool protocol and CUDA verification |
| [Qwen Flash investigations](docs/performance/qwen-flash-prefill-strata-20261008.md) | Expert residency, prefill, streaming, correctness and experimental limits |
| [Native suite](docs/architecture/native-suite-release.md) | Image/embedding preparation, resource ownership and release requirements |
| [Upstream merge audit](docs/architecture/ds4-upstream-merge-20261001.md) | Integrated DS4 changes and locally tested versus untested paths |
| [Original DS4 guides](docs/upstream/_INDEX.md) | Preserved upstream serving, backend and model documentation |
| [Research references](performance/references.md) | External implementations, papers and their relevance to this hardware |

## Contributing

Contributions are welcome in model-specific kernels and inference, hierarchical execution, frontend/runtime integration, multimodal operations, reproducible evaluation, and documentation.

Please start with [CONTRIBUTING.md](CONTRIBUTING.md) for the inherited native-engine correctness and performance checks, and [AGENT.md](AGENT.md) for the project's engineering conventions. Contributions to the agent/frontend or native suite should additionally exercise the relevant tests and report **what was verified versus not verified**.

For a pull request, include the affected model/backend, exact test commands, hardware and model artifacts, correctness results, performance impact where applicable, and any external source that informed the change. If an optimization also applies to upstream DS4 independently of ExMekh's agent-specific work, consider contributing the general fix upstream.

## Credits, inspiration, and provenance

ExMekh builds on the work of others. **Code inheritance, adapted reference behavior, architectural inspiration, and runtime dependencies are different relationships**; the credits below deliberately distinguish them.

### Foundation: DwarfStar, llama.cpp and GGML

- **[DwarfStar / DS4](https://github.com/antirez/ds4)** — created by **Salvatore Sanfilippo (antirez)** and contributors. The DS4 engine, many model and backend paths, and the model-specific engineering philosophy are the direct upstream foundation of this fork. **ExMekh is not an official DwarfStar release.**
- **[llama.cpp](https://github.com/ggml-org/llama.cpp) / GGML** — foundational GGUF/quantization ecosystem, source and kernel references, and components acknowledged by upstream DS4. Exact obligations remain governed by the inherited files and their notices.

### Architectural inspiration: Strata

**[Strata](https://github.com/Niko1221/Strata)**, by **Niko1221 and contributors**, was an important **technical reference** for the Qwen Flash work on consumer GPUs. ExMekh studied its handling of expert residency, CPU/GPU and PCIe staging, prefill ordering, CUDA execution, and NextN/MTP scheduling. Those investigations influenced independently implemented DS4-side approaches, adapted to its different GGUF layouts, kernels, numerical contracts and RTX 3090 constraints.

The repository's documented investigations identify pinned Strata revisions under its MIT license and explicitly distinguish **ideas studied from code copied**. No wholesale Strata kernel import is claimed, and Strata's public performance figures are not used as ExMekh benchmark results. Not every explored technique was retained or enabled.

Detailed source-level attribution and experimental outcomes:

- [Initial Strata-inspired Qwen Flash pipeline](docs/performance/qwen-flash-strata-20261006.md)
- [Architecture and kernel audit](docs/performance/qwen-flash-strata-audit-20261007.md)
- [Prefill, ring, scheduling and MTP follow-up](docs/performance/qwen-flash-prefill-strata-20261008.md)

### Native suite: reference implementations and media libraries

The native suite has source-level provenance and licensing notices under [`third_party/`](third_party/):

- **[Hugging Face Diffusers](https://github.com/huggingface/diffusers)** — Qwen-Image-2.1 model, VAE, pipeline and scheduler reference behavior.
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** — EmbeddingGemma-2, Gemma model and multimodal preprocessing references.
- **[PyTorch](https://github.com/pytorch/pytorch)** — selected numeric reduction and preprocessing conventions reproduced in native paths.
- **[NumPy](https://github.com/numpy/numpy)** and **[Pillow](https://github.com/python-pillow/Pillow)** — selected audio/math and image-resampling reference behavior.
- **[FFmpeg](https://ffmpeg.org/)** and **[libsoxr](https://sourceforge.net/projects/soxr/)** — dynamically linked native media decoding and resampling dependencies, with their own redistribution obligations.
- **[Iris](third_party/iris/LICENSE)** and other inherited small components — see their own file-level licenses and notices.

These references do **not** imply that the native inference server depends on Python inference runtimes. Python tooling may still be used for offline asset preparation and numerical oracles. Model weights and architectures retain their respective creators' distribution terms; ExMekh does not relicense them.

### Frontend and further research

**[NiceGUI](https://github.com/zauberzeug/nicegui)** provides the ExMekh Agent interface framework. Its design documentation also names **[LM Studio](https://lmstudio.ai/)**, **[Unsloth](https://github.com/unslothai/unsloth)** and **[BionicGPT](https://github.com/bionic-gpt/bionic-gpt)** as user-experience inspirations, **not as imported frontend code**.

Additional sources studied for specific engineering questions include **[vLLM](https://github.com/vllm-project/vllm)**, **[Marlin](https://github.com/IST-DASLab/marlin)** and **[NVIDIA GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)**. They are **research references, not blanket claims of code reuse**. See the [research index](docs/research/INDEX.md) and [performance references](performance/references.md) for the context of these citations.

### Licensing and development practice

The repository's top-level [MIT license](LICENSE) applies subject to applicable source-file and third-party terms. Copyright notices and license obligations for inherited, adapted, and dynamically linked components must be preserved; see `third_party/`, source headers and relevant model licenses. In particular, redistributors of builds containing LGPL-linked media libraries should review the accompanying notices.

AI coding agents are used extensively for exploration, implementation, review, and test generation. Maintainers remain responsible for design choices, original-source attribution, reproducible validation, and truthful qualification claims.

---

**ExMekh's guiding idea:** *keep the model good, keep the active context purposeful, and give the local agent the right specialized capabilities—without pretending that consumer hardware has unlimited memory.*
