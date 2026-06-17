# llm_ass3

`doit` - translate a natural-language instruction into a shell command and run it
via an OpenAI model.

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

Omit the instruction to enter the interactive agent loop:

```bash
bash src/doit            # Windows
./src/doit               # Linux/macOS
```

### How it stays cross-platform

`uv` runs a native Python interpreter, which on Windows cannot execute a POSIX
path like `/bin/bash`. The launcher resolves the platform's real `bash`
executable (Git Bash on Windows, `/bin/bash` on Linux/macOS) and passes it to
the tool via the `DOIT_BASH` environment variable. You can override it manually:

```bash
DOIT_BASH="/path/to/bash" ./src/doit "..."
```
