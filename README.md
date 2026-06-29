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

### How it stays cross-platform

`uv` runs a native Python interpreter, which on Windows cannot execute a POSIX
path like `/bin/bash`. The launcher resolves the platform's real `bash`
executable (Git Bash on Windows, `/bin/bash` on Linux/macOS) and passes it to
the tool via the `DOIT_BASH` environment variable. You can override it manually:

```bash
DOIT_BASH="/path/to/bash" ./src/doit "..."
```
