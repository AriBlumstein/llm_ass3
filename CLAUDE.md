# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`doit` is a natural-language-to-shell-command agent. It takes an English instruction, asks an LLM to translate it into a bash command, runs a safety/modification check, optionally asks the user for confirmation, executes the command, and records the turn so later instructions can refer back to it.

The project targets **multiple LLM backends through LiteLLM** — OpenAI models (e.g. `gpt-5.4-nano`) *and* local Ollama models (e.g. `ollama/gemma3:4b`). The active model is chosen at runtime via `doit.cfg`, not hardcoded. The README's "OpenAI" framing is historical; the default in `doit.cfg` is an Ollama model.

## Commands

```bash
# Run the agent on a single instruction (preferred entrypoint — see "Launcher" below)
./src/doit "list all files in the current directory"

# Interactive / no-instruction prints help
./src/doit

# Clear the current session's history and start fresh
./src/doit -n        # or --new

# Run the full test suite
uv run pytest

# Run a single test file / test
uv run pytest tests/test_doit.py
uv run pytest tests/test_doit.py::test_run_single_tool_calling_no_modification -v
```

Tests are pure unit tests: every `litellm.completion` call is mocked (`MockResponse`/`MockMessage`/`MockToolCall`), history is redirected to `tmp_path`, and `execute_bash`/`input` are patched. They need **no API key and no network** — do not add tests that make real LLM calls.

## Configuration

- **`doit.cfg`** (`[model]` section) is the runtime switch read by `config_loader.load_config()`:
  - `name` — LiteLLM model id (`openai/...`, `ollama/...`).
  - `tool_calling` — `True`/`False`. This is the single most important flag; it selects between two entirely different prompt/parsing code paths (see Architecture). Set `False` for local models that lack native function calling.
  - `api_base` — optional custom endpoint (e.g. `http://localhost:11434` for Ollama).
- **`.env`** at project root supplies `OPENAI_API_KEY` (loaded by `src/fixtures.py`). `fixtures.py` also holds all prompt text and tuning constants (`LLM_CONTEXT_LIMIT`, `CTX_NUM`).

## Architecture

Source lives under `src/` and is imported as top-level packages (the launcher and tests both put `src/` on `PYTHONPATH`). Three pieces matter:

1. **`doit_module/__main__.py`** — CLI entrypoint (`uv run -m doit_module`). Argparse, top-level error handling, instantiates `BashToolAgent` and calls `run_single`.
2. **`llm_communicator/llm_bash.py`** — the core. `BashToolAgent` orchestrates everything; `execute_bash` runs commands; the prompts/tool schema live here and in `fixtures.py`.
3. **`llm_communicator/history_manager.py`** — JSONL persistence of turns.

### The tool-calling vs. fallback split (most important concept)

`BashToolAgent` branches on `self.tool_calling` in two places, and these branches must stay in sync conceptually:

- **Tool-calling mode**: the model is given `tools_definition` (the `execute_bash_command` function schema from `BashCommandInput`). A command is a native `tool_calls` object; results are fed back as `role: "tool"` messages. Non-command responses come back as plain assistant text.
- **Fallback mode** (`tool_calling = False`): `FALLBACK_SYSTEM_INSTRUCTION` is appended to the system prompt, forcing the model to emit a **raw JSON block** (`executable`, `command`, `explanation`, `rule_triggered`, `response_text`). Results are fed back as `role: "user"` messages (local models can't handle `tool` role). `num_ctx = CTX_NUM` is passed on every call. JSON is extracted via `parse_json_response` (tolerant of markdown fences / conversational noise).

When changing behavior, check **both** branches in `run_single` and the corresponding history-replay formatting.

### Single-turn execution flow (`run_single`)

1. **Reference resolution** — `_analyze_references` decides which past turns the new instruction depends on. A cheap regex heuristic (`context_indicators`) short-circuits obviously-independent instructions *without* an LLM call; only ambiguous ones trigger a secondary LLM classification returning `relevant_ids`.
2. **Transitive closure** — `_resolve_transitive_dependencies` walks `relevant_ids` links so the full dependency chain is pulled in.
3. **Context reconstruction** — only the relevant past turns (not the whole history) are replayed into `conversation_history`, formatted per the tool-calling/fallback split. Outputs over 2000 chars are truncated.
4. **Primary LLM call** → produces a command (or a plain rejection).
5. **`_filter_bash`** — a *second* LLM call using `DOIT_FILTER_PROMPT` classifies whether the command modifies the filesystem (`DECISION: YES/NO`). If yes, the user is prompted `[y/N]` before execution.
6. **`execute_bash`** — runs `bash -c` in a subprocess with a 20s timeout, after a hardcoded regex blacklist (`BANNED_COMMAND_PATTERNS`) that raises `BashSafetyViolationError`.
7. **Persist** — the turn (prompt, command, output, relevant_ids) is appended to history.

There are **two independent safety layers**: the LLM filesystem-modification judge (asks the user) and the regex blacklist (hard block). Keep them distinct — the blacklist is the non-bypassable backstop.

### Behavior is prompt-driven, and the prompts are tested

`DOIT_SYSTEM_PROMPT` in `fixtures.py` encodes a numbered rule system (Rules 1–7: command generation, impossible commands, safety violations, irrelevant input, capability inquiries, assume-file-existence, multi-turn/missing-context). Several tests assert **exact substrings** of these prompts (e.g. `test_doit_system_prompt_*`, `test_history_system_instruction_rules`). If you edit a prompt, expect to update the matching assertions — the prompt text is part of the contract, not just a string.

### History / sessions

Sessions are isolated per shell. Each session has a folder `<repo>/.doit/history_<pid>/` (install-relative, not cwd-relative) holding `doit.jsonl` (doit's own turns) and `cmdlog.tsv` (the user's terminal commands + exit status, written by the shell recorder hook). The **shell** (`doit-init.sh`) owns the folder — it names it from its own `$$` and points `DOIT_CMD_LOG` into it; `history_manager.get_history_file_path()` then **follows** that folder via `dirname(DOIT_CMD_LOG)` rather than recomputing a PID. This matters because the launcher's `DOIT_PPID` (from `$PPID`) does **not** always equal the shell's `$$` — notably in the VS Code integrated terminal — so deriving the folder independently on each side would split `doit.jsonl` from `cmdlog.tsv`. Without shell integration, it falls back to `.doit/history_<DOIT_PPID or getppid>/doit.jsonl`. Each turn has an incrementing `id` and a `relevant_ids` list forming the dependency graph used by reference resolution.

**Output awareness:** doit answers questions about a previous command's output. For its own commands the output is in history; for commands the *user* ran (only command + exit status are stored, never output), Rule 11 directs the model to **re-run/pipe** the command to obtain the data when it succeeded and is safe to repeat. Rule 10 defines attribution defaults (you/we → doit's last action; I → user's last; bare reference → most recent).

### Launcher (`src/doit`)

`src/doit` is a bash wrapper, not a Python entrypoint, for cross-platform reasons: it resolves the real `bash` executable (Git Bash on Windows, `/bin/bash` elsewhere), exports it as `DOIT_BASH` so the Python `_resolve_bash()` uses a POSIX-capable shell, sets `PYTHONPATH` to `src/`, sets `DOIT_PPID`, then `exec uv run -m doit_module`. Prefer invoking through this script rather than calling the module directly.

## Repo conventions

- **Git**: per `.agents/rules/git-rules.md`, ask before running any git command and explain why first — routine work should not need git.
- **ACDL docs**: `data/**/*.acdl` are Agentic Context Description Language specs documenting the agent's prompt/context structure. The authoring rules live in `.agents/rules/skills/acdl-documenter/SKILL.md` — follow them (named prompt definitions, `S`/`U`/`A`/`T` role abbreviations, `@T` time indices, `ForEach` for history) when editing or adding ACDL. The data READMEs contain mermaid diagrams of both execution flows.
- Dependencies and Python (3.12+) are managed with **uv**; use `uv run` / `uv add` rather than invoking `pip` or a bare `python`.
