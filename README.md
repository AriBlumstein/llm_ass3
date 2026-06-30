# llm_ass3

`doit` - translate a natural-language instruction into a shell command and run it
via an OpenAI model or a local Ollama model.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Put your key in a `.env` file at the project root: `OPENAI_API_KEY=sk-...`

## Running

The launcher is a bash script, so it runs the same way on every platform - it
just needs a `bash` to start it.

**Linux / macOS:**

```bash
./src/doit "list all files in the current directory"
```

**Windows:** run it from **Git Bash** (ships with [Git for Windows](https://git-scm.com/download/win)):

```bash
bash src/doit "list all files in the current directory"
```

An instruction is required; running `doit` with no argument prints a usage error.

## Shell integration (persistent `cd`/`export` + user awareness)

A program on your PATH runs as a child process and cannot change your shell's
working directory, environment, or read its live command history. To enable
those, `doit` ships a small shell function you source once:

```bash
# add to ~/.bashrc or ~/.zshrc (doit also offers to add this for you on first run)
source /absolute/path/to/llm_ass3/src/doit-init.sh
```

With it sourced, `doit` becomes a shell function that wraps the launcher and gives you:

- **Persistent `cd` / `export` / `alias` …** — when the agent generates a
  shell-state command (e.g. "go to my project"), the function applies it in *your*
  shell, so the change persists.
- **User awareness (with success/failure)** — sourcing the file also installs a small
  per-command hook (bash `DEBUG` trap + `PROMPT_COMMAND`; zsh `preexec`/`precmd`) that
  logs each command you run **and its exit status** to this session's `cmdlog.tsv`
  (exposed as `DOIT_CMD_LOG`). `doit` reads it and records your manual commands as `user`
  turns — with real `exit 0` / `FAILED` markers — alongside its own `doit` turns in one
  ordered, per-session history. So it can answer `doit "summarize what I just did"`, ground
  new commands in your current directory, tell what came last (e.g. if you manually undo
  something it did), and know whether a command you ran succeeded or failed. (If the hook
  isn't active, it falls back to `fc -l`, which has the commands but no exit status.)

Everything for one shell session lives together in a single gitignored folder,
`<repo>/.doit/history_<pid>/`, holding `cmdlog.tsv` (your commands + exit codes) and
`doit.jsonl` (doit's own turns). The shell picks the folder (from its own PID) and `doit`
**follows it** via `DOIT_CMD_LOG`, so the two files always co-locate — even in terminals
(e.g. VS Code's) where the launcher's process id doesn't match the shell's.

**Clearing history.** `doit -n` (or `--new`) clears just the current window's history and
starts it fresh. `doit --reset` wipes **every** session's history (all `history_<pid>/`
folders across all windows) for a full clean slate, then exits. Both leave your **memories**
(`<repo>/.doit/memories.json`, a sibling of the session folders) completely untouched — use
`--reset` to forget what you've done without forgetting what doit knows about you.

### What was changed outside the program, and why

The only out-of-program change is the one `source …/doit-init.sh` line in your
`~/.bashrc`/`~/.zshrc`. That loads `src/doit-init.sh`, which (a) defines the `doit`
shell function and (b) installs a per-command recorder hook. The function does what a
child process can't — apply directory/shell-state changes in the current shell. The
recorder hook captures each command + its exit status (a child process can't see your
shell's command history or results) and appends them to the gitignored
`<repo>/.doit/history_<pid>/cmdlog.tsv`. Nothing is written to your own shell history file.

**Notes & privacy:** your recent shell commands are sent to the configured LLM as
context (only commands and their exit status, **not** their output). Without the function
(e.g. running `./src/doit` directly), doit still works — it just loses persistence and user
awareness and falls back gracefully. Remove the integration anytime by deleting
the `source` line from your shell rc.

## Output awareness — asking questions about command output

You can ask `doit` about the output of a previous command and it answers either directly
or by running a command to find out:

```bash
doit "list the largest files here"      # doit runs it; it has the full output
doit "which of these is safe to delete?"
doit "why did that command fail?"
```

- For commands **doit itself ran**, it already has the full output (stdout/stderr/exit
  code) and answers from it.
- For commands **you ran yourself** in the terminal, doit has the command and its exit
  status but not the output — so it **re-runs the command** (piping it as needed, e.g.
  `ls -lhS | head`) to get the data, then answers. It only does this for commands that
  succeeded and are safe to repeat (read-only ones like `ls`/`cat`/`grep`); it won't blindly
  re-run something that changed your files. Re-runs still pass the usual safety check.

Attribution follows what you say: "what did **you/we** just do" refers to doit's last
action, "the command **I** just did" to your last command, and an unqualified "that" / "the
previous command" to whichever ran most recently.

## Multi-window — working across several terminals

Each terminal window is its own session, identified by its **shell PID**, with its own
history. By default windows are fully isolated: a follow-up like `doit "sort them by date"`
in one window only ever refers to *that* window's commands — never another's.

When you *do* want to reach across windows, reference another one explicitly:

```bash
doit "list the shell numbers"          # see your open windows: pid — directory — last active
doit "do the folder task we did in the other window here"
doit "redo what I ran in session 12345 here"
```

doit finds the right window by its **PID** (exact), or — for fuzzy references like "the
other terminal" / "the folder task" — by recency and by matching the described task across
your other sessions, asking you to pick from a numbered list if it's unsure. It then applies
that task in your **current** directory (it re-grounds the commands here; it doesn't reuse the
other window's paths), through the usual safety check. The shell PID is the stable handle —
you can always see a window's own with `echo $$`.

## Command plans (multi-step tasks)

For a task that needs several commands in sequence, `doit` plans them as ordered steps, **shows the
whole plan before running anything**, asks once, then runs the steps in order and **stops if one
fails** so a broken step doesn't cascade:

```bash
doit "set up a python project with a venv"
[PLAN] scaffold a python project
The following steps will run in order:
  1. mkdir myproj                - create the project directory
  2. python -m venv myproj/.venv - create a virtualenv
  3. git -C myproj init          - initialize a git repo
Run this 3-step plan? [y/N]: y

[STEP 1/3] mkdir myproj
...
```

If a step fails, the remaining steps are not run and `doit` tells you exactly where it stopped. You
approve the whole plan once (you've seen every command); each step still passes the safety blacklist.
This works with **both** kinds of model: native tool-calling models call an `execute_plan` tool, and
models without a tool API (or an OpenAI model run with `tool_calling = False`) emit the plan as a
JSON `steps` array — either way it runs through the same preview / confirm / stop-on-failure runner.

`cd` and `export`/`alias` work inside a plan: a `cd` step moves the plan's working directory for the
*later* steps (so `mkdir proj`, `cd proj`, `touch main.py` puts the file in `proj`), and the plan's
net directory + shell-state are applied to **your** shell after it finishes — just like a single
`doit "go to my project"`. A `cd` into a missing directory is reported as a failed step and stops the
plan.

### How it stays cross-platform

`uv` runs a native Python interpreter, which on Windows cannot execute a POSIX
path like `/bin/bash`. The launcher resolves the platform's real `bash`
executable (Git Bash on Windows, `/bin/bash` on Linux/macOS) and passes it to
the tool via the `DOIT_BASH` environment variable. You can override it manually:

```bash
DOIT_BASH="/path/to/bash" ./src/doit "..."
```
