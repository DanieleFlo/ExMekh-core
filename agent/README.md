# ExMekh Agent — Local AI Studio Frontend for ExMekh-Lib

A Python frontend inspired by modern local LLM interfaces such as **LM Studio**, **Unsloth**, and **BionicGPT**. It is designed to integrate directly with the hierarchical agentic framework `exmekh_lib` and its distributed wheel package.

---

## Key Features

- **Dark Theme Inspired by LM Studio & Unsloth**: High-contrast obsidian/charcoal palette with support for markdown syntax, code blocks, and modern aesthetics.
- **Real-Time Streaming**:
  - **Collapsible Thinking Block**: visualization of reasoning tokens with an expandable animated view, identical to DeepSeek-R1 / LM Studio.
  - **Tool & Skill Execution Cards**: monitoring of hierarchical tools invoked by `DeepAgent` with depth tracking and skill paths.
  - **Turn Steering**: send a queued message as guidance to the active DeepAgent run. Accepted guidance and its delivery position remain visible after a page refresh and in saved conversation history. Guidance delivered inside a skill retains its canonical output ID in `get_execution_history` and `read_tool_output` across later turns, including its attached images and rendered PDF pages.
- **Images and PDFs**:
  - Pick, drop, or paste several images in one message, including queued guidance.
  - Click thumbnails in the composer or saved chat for a large preview with a blurred backdrop; close it with the button or Escape.
  - PDFs render locally into images: at most the first **10 pages per document**, in page order. The composer shows the selected/total page count and warns when pages are omitted.
  - Images keep their orientation and are resized to at most **2048 pixels per side**. Animated images use their first frame.
  - Original files stay in scoped temporary storage; normalized PNG content is also saved in chat JSON so vision history and previews survive refreshes and file cleanup. Chat messages show only previews for these visual attachments.
  - Chat Completions and Responses both use ExMekh-Lib mixed user inputs. The endpoint/model must support vision; DS4 needs its matching encoder loaded with `--vision`.

- **Editable messages**: change text, remove original attachments, and upload new files before regenerating a response. Cancel keeps the original message and attachments. Saving discards subsequent conversation history and restarts the turn.
- **Agent profiles and visual customization**:
  - Choose Generic, Programmer, Creative, or Custom through four icon buttons in Project settings. The Custom icon opens the visual editor; tooltips and accessible names explain each profile. Chat agent settings override the project profile only for that conversation; new chats inherit the project default.
  - Customize a profile on a draggable canvas. Purple connections represent HDS sub-agents, while blue and green connections assign skills and tools to an HDS or skill. Dashed connections show skill dependencies, including nested skills and shared tools. Select cards to connect, disconnect, remove, choose the root, or configure sub-agent history. Connect to lists only valid new connections. Lines follow nodes during dragging; arrow controls pan smoothly in small steps. Undo or Ctrl+Z / Cmd+Z restores every board edit while text fields keep their native undo.
  - The palette includes bundled resources and capabilities discovered in the configured project paths. Create HDS instructions or edit a capability as a named copy; library originals remain available. Skill and HDS editors separate the description from the instructions and hide the YAML front matter. Sub-skills and sub-tools are read-only in the editor and generated from the canvas connections when saving and running the agent. Copies live under `data/agent-capabilities/`, and generated runtime configurations under `data/agent-designs/`.
  - Deep Thinking retains provider thinking in direct tool output headers, up to 200 tokens before summarizing. A renamed copy follows ordinary skill behavior; the editor explains this distinction.
- **Readable execution results**: tool rows remain compact until clicked or activated with Enter/Space. Expanded results scroll within a bounded panel; completed skills offer their result in the last row of the expanded skill. Raw History preserves the XML header at the top of content with readable indentation while copying the original JSON. Archived read_tool_output results are matched by output ID and summary mode, including older saved chats.
- **Responsive tabs**: chat, execution graph, history, dialogs, and the agent editor adapt to desktop, tablet, and phone widths.

- **Local Provider Control**:
  - Dynamic health check of the local server endpoint (LM Studio, Ollama, llama.cpp, vLLM).
  - Dynamic discovery and selection of installed models.
  - Project conversations automatically use **DeepAgent**; standalone **Quick Chats** use the typed client without skills or tools.
- **Projects and Quick Chats**:
  - Projects are named after their selected work folder. The folder is fixed after creation.
  - Project settings prioritize agent selection, then default bash approvals. The less frequently changed capabilities folders stay in a collapsed section. Sub-agent history is configured on the agent canvas.
  - The default capabilities folder is always included and read-only. Add extra folders with the native folder picker (select a `capabilities` folder or its containing project), remove them independently, and reuse them across project chats. Duplicate references are ignored; saved legacy single-folder overrides are retained as additional folders.
  - Bash approval changes inside a chat stay local; the project default applies to newly created conversations.
  - Persistent chat history lives under `data/chats/`, and project settings under `data/projects/`.
  - Temporary uploads and pasted files use `<work_root>/tmp/projects/<project_id>/`; Quick Chats use the frontend code folder and `tmp/quickchats/`.
  - Input drafts and uploads stay with their original conversation when switching chats. Quick Chats include text attachments directly in the prompt and send images and rendered PDF pages as native vision inputs.
  - Deleting a project deletes its conversations and stops their active turns. The selected work folder remains available.
  - Existing agent conversations are automatically grouped by work folder. When their previous settings differ, the most recently updated chat supplies the shared project settings; each chat keeps its bash approval mode and approval history.
- **Automatic `exmekh_lib` Wheel Synchronization**:
  - WSL shell script to build the sibling ExMekh-Lib sources, update the frontend wheel and lockfile, and reinstall ExMekh-Lib.

---

## Wheel Synchronization from `exmekh_lib`

From the repository root in WSL:

```sh
./agent/sync_exmekh_lib.sh
```

The script builds the current sources in the sibling `exmekh_lib/` directory, copies the
resulting wheel into `agent/wheels/`, updates its path in `pyproject.toml` and
`uv.lock`, and reinstalls ExMekh-Lib. After a successful sync, it removes older ExMekh-Lib wheels.
No previous build or frontend virtual environment is required. Wheel cleanup runs
only after a successful installation.

The installed sync command is `uv run sync-exmekh-lib` from `agent/`.
To start the framework's interactive CLI with the frontend's locked wheel:

```sh
cd agent
uv run --frozen --env-file .env exmekh-lib
```

The Python namespace is `exmekh_lib`; the distribution and console command are
`exmekh-lib`. The frontend integration module is `exmekh_agent.exmekh_bridge`. The CLI
loads `BASE_URL` and `API_KEY` from `.env` with the command above. Set `EXMEKH_LIB_MODEL`
and `EXMEKH_LIB_API` to override its model and API; these are the environment variable
names exposed by the renamed library. After updating the library sources, sync the
wheel so the installed configuration matches them.

Both shell launchers resolve paths from their own location, so they can be called
from any working directory. They use `uv` from `PATH` or `~/.local/bin/uv`;
set `UV_BIN=/path/to/uv` to choose another executable.

---

## Starting the Frontend

For Qwen Flash CUDA, start `./agent/start-ds4-server-flash.sh` from the repository
root. Its standard profile enables the qualified expert ring, explicit KV
streaming with 32768 resident cells, CUDA decode graphs/ranking and prefill
chunk 2048. The default context capacity is 64K; set `DS4_CTX` for an explicit
capacity. Native Qwen MTP remains enabled.
Dense projection columns retain the original 256-row arithmetic even for
larger prefill chunks. Set `DS4_PREFILL_CHUNK` or the existing cache variables
to override a setting; `DS4_FLASH_PROFILE=legacy` reproduces the previous
256-row launch for comparisons. The Qwen27 launcher keeps its own profile.
The [local qualification report](../docs/performance/qwen-flash-agent-launcher-20261008.md)
records numerical, state, vision and real-agent A/B results.

From the repository root in WSL:

```sh
./agent/run.sh
```

The launcher installs the locked dependencies when needed and starts the frontend.
From `agent/`, the equivalent command is `uv run --frozen exmekh-agent`.
The distribution and UI command are `exmekh-agent`; Python imports use
`exmekh_agent`. The framework CLI remains `exmekh-lib`.

Copy `.env.example` to `.env` to configure the application. Its web settings
are `EXMEKH_AGENT_HOST`, `EXMEKH_AGENT_PORT`, and `EXMEKH_AGENT_TITLE`. Existing
`.env` files using `FE_HOST`, `FE_PORT`, and `FE_TITLE` must use those new names.
Provider settings such as `BASE_URL`, `API_KEY`, and `DEFAULT_MODEL` stay the same.
The model selector and context bar refresh the provider's `/models` metadata at
startup and every 10 seconds. If the saved model is unavailable, the running
client selects an advertised model; supported aliases remain selectable. DS4
reports the allocated `--ctx`, so switching between Qwen Flash and Qwen 27B also
updates the displayed context capacity. Automatic discovery does not rewrite
`.env`; the Save button persists settings.
On startup, saved projects and chats that point to the previous library's bundled
`site-packages/qs/capabilities` are migrated to the installed library defaults.
Custom capabilities paths, conversation history, and approvals are preserved.

The application will open at:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Running Tests

### Offline Unit Tests
Run the standard (offline) tests:
```shell
cd agent
uv run pytest
```

Tests run by module on all logical CPUs available to pytest by default, keeping
module fixtures on one worker. Real-provider `live` tests (excluding `mock`) and tests marked
`@pytest.mark.serial(reason="...")` run one at a time after all parallel workers
have finished, including fixture teardown. Clipboard tests use this marker
because the operating system clipboard is shared between browsers.
Use `-n 0` to debug without workers, or `-n 16` to choose a worker count.
`PYTEST_XDIST_AUTO_NUM_WORKERS=16 uv run pytest` sets the default for `-n auto`.
On WSL, automatic detection follows the CPU limit configured in `.wslconfig`;
Windows and WSL share the host CPU resources.
Browser servers use available localhost ports and isolated storage, so separate
pytest invocations can run concurrently. Server logs are retained in `.test-tmp`.

The test suite automatically enforces branch coverage and requires 100% on the
Python configuration, state, and ExMekh-Lib integration code. The calculation excludes the
`exmekh_agent.main` and `exmekh_agent.sync` entrypoints and the UI modules listed in `pyproject.toml`, which are verified by
functional and Playwright tests in the same suite. To run only a subset of tests,
use `--no-cov`; the threshold applies to the full suite.

To run quality checks and build distribution separately:

```shell
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

### Live End-to-End Tests
Tests with mock providers exercise the agent workflow without an LLM server:
```shell
cd agent
uv run pytest --no-cov -m mock
```

For a vision-enabled DS4 server, verify multiple images, PDF conversion and
follow-up turns through both APIs:

```shell
uv run pytest --no-cov --run-live -n 0 tests/test_images_live.py
```

DS4 accepts finite local `#/$defs/name` references emitted by nested Pydantic
output models, including nullable branches. All constraints remain active;
external references, recursive schemas and unsupported constraints return an
explicit error. Too little output budget fails instead of accepting partial JSON.

For a Flash server, verify strict Pydantic output through the installed client
with both Chat and Responses, streaming, thinking, greedy/sampled decoding,
nullable nested fields, Unicode and truncated-output rejection. Run against
one server at a time, first target-only and then MTP if enabled:

```shell
DS4_FLASH_STRUCTURED_BASE_URL=http://127.0.0.1:8080/v1 \
  uv run pytest --no-cov --run-live -n 0 tests/test_flash_structured_live.py
```

With the provider configured in `.env` available, run all tests with a real model (conversation, streaming, tools, persistence, title generation, and adversarial cases of `exit_agent`):
```shell
uv run pytest --no-cov --run-live -m "live and not mock" -s -v
```

`--run-live` enables tests that send requests to the model. Without this option (or `EXMEKH_AGENT_RUN_LIVE_TESTS=1`), the standard suite skips them; mock tests remain runnable.
Each adversarial case of `exit_agent` has a 90-second timeout. A timeout terminates the process and marks the case as skipped because recovery was not verified. Cases where the model does not produce the requested malformed payload are also skipped. Deterministic argument tests independently check omitted `items` and invalid item types.

---

## Package Structure

```
agent/
├── .env                  # BASE_URL, API_KEY, MODEL configuration
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── pyproject.toml        # uv configuration and dependencies
├── README.md             # Project documentation
├── .python-version      # Python 3.13
├── run.sh               # WSL launcher for the UI
├── sync_exmekh_lib.sh    # Build and install the sibling ExMekh-Lib sources
├── start-ds4-server.sh  # WSL launcher for Qwen27B
├── start-ds4-server-flash.sh  # WSL launcher for Qwen Flash Next
├── wheels/               # Synchronized exmekh_lib wheel
│   └── exmekh_lib-0.1.0-py3-none-any.whl
├── data/
│   ├── chats/           # JSON storage for conversations
│   └── projects/        # Shared project settings
└── src/
    └── exmekh_agent/
        ├── __init__.py
        ├── main.py       # NiceGUI entry point
        ├── config/
        │   ├── application.py # AppConfig and environment persistence
        │   ├── catalogs.py    # Owned settings and parameterized messages
        │   ├── resources/     # Provider, server, agents, storage, attachments,
        │   │                  # execution and formatting defaults (JSON)
        │   ├── presentation/  # Graph geometry, board policy, sanitization,
        │   │   └── controls/  # Control styles and icons by UI component (JSON)
        │   └── texts/         # UI, errors, logs and approval text by domain (JSON)
        ├── sync.py       # Python sync script
        ├── exmekh_bridge/
        │   ├── agents/        # Execution, collection, restoration and steering
        │   ├── capabilities/  # Capability catalog and design compilation
        │   ├── history/       # Recorded invocations, attachments and edits
        │   ├── providers/     # Provider integration and token accounting
        │   ├── tools/         # Tool events and command approval
        │   ├── resources/     # Packaged model instructions
        │   └── models/        # Public Pydantic model API
        │       ├── execution.py # Tool/skill lifecycle, timing and context usage
        │       ├── messages.py  # Messages, guidance and child conversations
        │       ├── settings.py  # Project settings and chat overrides
        │       └── sessions.py  # Conversation branches and input queues
        ├── utils/             # Shared functions, divided by responsibility
        │   ├── files.py        # Atomic writes and temporary-file cleanup
        │   ├── filenames.py    # Attachment filename normalization
        │   ├── formatting.py   # Bytes, tokens, percentages and durations
        │   ├── exceptions.py   # Exception-chain traversal and rendering
        │   ├── markdown.py     # Markdown/YAML front-matter parsing
        │   └── resources.py    # Cached JSON and text reads
        ├── state/             # Managed sessions, persistence and active turns
        └── ui/
            ├── agents/        # Agent board, selector and capability editor
            ├── chat/          # Timeline, input, message editor and controller
            ├── graph/         # Execution graph data, geometry and view
            ├── history/       # Raw inspector and JSON highlighting
            ├── navigation/    # Header, sidebar and new project dialog
            ├── settings/      # Application and project settings
            ├── shared/        # Disclosure, attachments, Markdown and outputs
            ├── resources/
            │   ├── templates/ # Escaped HTML and SVG templates
            │   ├── css/       # Stylesheets
            │   └── js/        # Browser interactions
            └── styles.py      # Style injection
```

Application defaults are divided by responsibility in `config/resources`;
`config/texts` contains static and parameterized copy, and
`config/presentation` contains layout, styles, icons and rendering policies.
`AppConfig` keeps its public API and environment overrides. Shared functions live
in `utils`; callers import them from the module that owns their responsibility.

UI text is resolved through `ui_text`, presentation configuration through
`presentation_config`, and HTML/SVG through `render_html`. The UI delegates catalog
and settings lookups to `config.catalogs`. Resource paths are relative to the
installed package. Templates escape dynamic values by default; only sanitized
or already rendered fragments are marked safe. Model instructions remain in
`exmekh_bridge/resources/prompts`. Protocol identifiers, model schema keys and
algorithmic operations belong to code; adjustable defaults and presentation belong
to the configuration resources.

The `exmekh_bridge.models` package preserves existing imports and JSON formats.
Invocation correlation belongs to the recorded-history adapter: chat and graph
use the same rules for regular tools, cached reads and legacy `function` replies.
Global settings apply on the UI event loop and persist an independent snapshot
in a worker. Failed persistence is visible in the form and can be retried.

See [architecture and responsibilities](docs/architecture.md) for the public APIs.

## DS4 integration

This checkout lives in `agent/`. The sibling `exmekh_lib/` source copy is ignored by
Git; the versioned wheel in `wheels/` is the runtime dependency. In WSL/Linux,
start either `./start-ds4-server.sh` (Qwen27B) or
`./start-ds4-server-flash.sh` (Flash Next), then `./run.sh` in a separate terminal. The model
server uses port 8080 and the frontend uses port 8000. Set `API_TYPE=responses`
and keep the client context at or below the server allocation reported by `/v1/models`.
See [the DS4 frontend guide](../docs/agentic/frontend.md).

Browser tests use installed Edge when available and otherwise use Playwright Chromium.

The `.python-version` pins Python 3.13, matching the source environment and test baseline.

### Selecting the local inference model

From `agent/`, use a dedicated server launcher for each model:

```sh
./start-ds4-server.sh        # Qwen27B: MTP, context 28000
./start-ds4-server-flash.sh  # Flash Next: MTP, context 65536, prefill chunk 2048
```

Start the frontend with `./run.sh` in another terminal. Each launcher selects
its model profile even if `DS4_MODEL_PROFILE` is set in the environment.
Model discovery reads the real server model and allocated context. Flash uses
its matching projector and CUDA tiered cache. Both launchers enable MTP.

Both scripts accept `--print-command` to review the exact command without
loading weights. `DS4_CTX`, `DS4_PREFILL_CHUNK`, `DS4_MODEL`, `DS4_VISION_MODEL`
and `DS4_KV_DISK_DIR` can override the profile. An empty vision path disables
vision; an empty KV directory disables the shared launcher's KV disk arguments.
See [Flash pipeline qualification](../docs/performance/qwen38-flash-pipeline-20261004.md)
for measured scenarios and remaining limits.

## Architecture

[Component responsibilities](docs/architecture.md) documents the public APIs for
agent designs, capability storage, profile selection, message editing, and output
expansion, together with the regression checks performed during the review.
