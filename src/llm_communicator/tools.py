"""
Tool layer for the doit agent.

Holds everything that defines and implements the agent's tools:
  - argument schemas (pydantic models transformed into JSON Schema for the LLM),
  - the tool definitions handed to LiteLLM (`tools_definition`),
  - the local implementations the tool calls map to (`execute_bash`, `ask_user_clarification`),
  - and the safety backstop that guards execution (`BANNED_COMMAND_PATTERNS`).

Unlike the always-run helpers (filter, reference resolution) which live on the agent, these
tools may or may not be invoked on a given turn, so they are self-contained here. The
clarification tool authors its question with its own LLM call configured from `doit.cfg`.

`llm_communicator.llm_bash` imports from here; this module must not import from it.
"""

import sys
from pathlib import Path

# Add the 'src' directory to sys.path so sibling packages (fixtures, doit_module) import.
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import os
import re
import shutil
import json
import functools
import subprocess
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import litellm
from datetime import datetime, timezone

from fixtures import (
    CLARIFY_AUTHOR_PROMPT, ANSWER_HOWTO_PROMPT, MEMORY_MANAGER_PROMPT, CTX_NUM,
    CROSS_SESSION_RESOLVER_PROMPT,
)
from doit_module.config_loader import load_config


def is_openai_model(model_name: str) -> bool:
    """
    True for OpenAI models. `num_ctx` is an Ollama-only completion option; OpenAI models
    reject/ignore it, so callers skip it for these models.
    """
    name = (model_name or "").lower()
    return name.startswith("openai/") or name.startswith("gpt")


# To protect the host system, we configure a basic blacklist of dangerous commands.
BANNED_COMMAND_PATTERNS = [
    r"\brm\s+-[rfRF]+.*\/",           # Root/dangerous recursive deletion
    r"\bchmod\b.*777",                 # Dangerous wildcard permissions
    r"\bkillall\b",                    # Indiscriminate process termination
    r"\bshutdown\b",                   # Host shutdown command
    r"\breboot\b",                     # Host reboot command
    r"\bdd\s+if=/dev/zero",            # Zeroing out drives
    # Classic fork bomb detection. The literal between "};" and "\s*:" is U+202F
    # (narrow no-break space), kept explicit so the pattern's bytes are unambiguous.
    r"\b:\(\)\{\s*:\s*&\s*:\s*\};" + " " + r"\s*:",
]


class BashSafetyViolationError(Exception):
    """Raised when a generated bash command violates execution security policies."""
    pass


class BashCommandInput(BaseModel):
    """
    Schema for the bash tool execution block.
    This structure is automatically transformed into JSON Schema format for OpenAI.
    """
    command: str = Field(
        ...,
        description="The single line or multi-line bash script to execute in the local subshell."
    )
    explanation: str = Field(
        ...,
        description="A short explanation of what this bash command will perform on the host system."
    )


class AnswerInput(BaseModel):
    """
    Schema for the `answer_question` tool: an informational/how-to reply that is NOT
    executed. Mirrors BashCommandInput, but `suggested_command` is purely advisory - it
    is shown to the user and persisted so a later "execute it" can run it.
    """
    explanation: str = Field(
        ...,
        description="The answer or how-to explanation to display to the user."
    )
    suggested_command: str = Field(
        default="",
        description="Optional bash command the user could run to accomplish this. Empty string if none applies. This is NOT executed - it is only suggested."
    )


@functools.lru_cache(maxsize=1)
def _resolve_bash() -> str:
    """
    Locate a bash executable usable by the *current* Python process.
    """
    override = os.environ.get("DOIT_BASH")
    if override and Path(override).exists():
        return override

    if os.name != "nt":
        # Linux / macOS
        for candidate in ("/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash"):
            if Path(candidate).exists():
                return candidate
        found = shutil.which("bash")
        if found:
            return found
        return "bash"

    # Windows
    candidates: List[str] = []
    git_path = shutil.which("git")
    if git_path:
        git_root = Path(git_path).resolve().parent.parent
        candidates.append(str(git_root / "bin" / "bash.exe"))
        candidates.append(str(git_root / "usr" / "bin" / "bash.exe"))

    candidates += [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    found = shutil.which("bash")
    if found and "System32" not in found and "system32" not in found:
        return found

    raise FileNotFoundError("Could not locate a 'bash' executable.")


def execute_bash(command: str, verbose: bool = True, cwd: Optional[str] = None,
                 prelude: Optional[List[str]] = None) -> str:
    """
    Executes a raw bash string in an isolated subprocess under strict constraints.
    Returns stdout/stderr merged result as a single string.

    `cwd` runs the command in that directory (used by the command-plan runner to thread a `cd` across
    steps). `prelude` is a list of already-screened session-state commands (e.g. `export K=V`) run
    BEFORE the command in the same shell, so a plan's earlier `export`/`alias` steps apply to a later
    step. Both default to the normal behavior (current dir, no prelude).
    """
    for pattern in BANNED_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            raise BashSafetyViolationError(
                f"Security Block: Command contains banned structural pattern matching '{pattern}'."
            )

    if verbose:
        print(f"\n[EXEC] Running Command:\n{command}\n")

    try:
        bash_executable = _resolve_bash()
    except FileNotFoundError as exc:
        return f"[Error: {exc}]"

    # Prepend any accumulated session-state (export/alias/...) so it applies to this command.
    run_command = ("\n".join(list(prelude) + [command])) if prelude else command

    try:
        result = subprocess.run(
            [bash_executable, "-c", run_command],
            capture_output=True,
            text=True,
            timeout=20.0,
            cwd=cwd,
        )

        if not verbose:
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr
            return output

        output = ""
        if result.stdout:
            output += f"--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"--- STDERR ---\n{result.stderr}\n"
        if result.returncode is not None:
            # Label the exit status explicitly: 0 means SUCCESS. A bare "0" was being misread by the
            # model as a failure (e.g. refusing to delete a file a prior `touch` actually created).
            status = "SUCCESS" if result.returncode == 0 else "FAILED"
            output += f"--- RETURN CODE ---\n{result.returncode} ({status})\n"

        # A command that exits 0 with no stdout/stderr still succeeded - say so clearly.
        if result.returncode == 0 and not result.stdout and not result.stderr:
            output = "[SUCCESS: command completed (exit code 0), produced no output]\n" + output

        if not output:
            output = "[Success: Command executed with no returning output channels]"

        return output

    except subprocess.TimeoutExpired:
        return "[Error: Command Execution Terminated due to exceeding 15.0s Timeout Limit]"
    except Exception as e:
        return f"[Error occurred during system execution execution loop: {str(e)}]"


# Sent back to the model when the user declines to answer a clarifying question, so the
# model commits to a sensible default instead of asking again.
NO_ANSWER_SENTINEL = (
    "[The user did not provide an answer. Do NOT ask again and do NOT explain what you would "
    "do. You MUST now ACT: produce the bash command for the most sensible default "
    "interpretation by calling the execute_bash_command tool (or, in non-tool-calling mode, "
    "output the command JSON with \"executable\": true).]"
)


def ask_clarification(question: str, options: Optional[List[str]] = None) -> str:
    """
    Ask the user a clarifying question authored by the clarification sub-call
    (BashToolAgent._author_clarification) and return their answer.

    Print a clarifying question (with an optional numbered menu) and read the user's answer.
    If the user provides no answer, re-prompt once; if still empty, return the no-answer
    sentinel so the caller can fall back to a sensible default. A numeric answer that maps
    to an option is resolved to that option's text. If the user picks a number outside the
    available options (e.g. 4 when only 3 are offered), the menu is shown again and the user
    is re-prompted locally, with no extra call to the LLM.
    """
    def prompt_once() -> str:
        if question:
            print(question)
        if options:
            for i, opt in enumerate(options, 1):
                print(f"{i}. {opt}")
        try:
            return input("Your answer (press Enter for a sensible default): ").strip()
        except EOFError:
            return ""

    empty_seen = False
    while True:
        answer = prompt_once()

        if not answer:
            # No answer: re-prompt once, then fall back to a default.
            if empty_seen:
                return NO_ANSWER_SENTINEL
            empty_seen = True
            continue
        empty_seen = False

        if options and answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            # Out-of-range numeric selection: re-show the menu and ask again, no LLM call.
            print(f"Please choose a number between 1 and {len(options)} (or type your answer).")
            continue

        return answer


def parse_json_response(content: str) -> Dict[str, Any]:
    """
    Robust JSON parser that extracts a JSON object {...} even if wrapped in markdown code blocks
    or conversational text.
    """
    content = content.strip()
    first_brace = content.find('{')
    last_brace = content.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = content[first_brace:last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return json.loads(content)


def _author_clarification(instruction: str) -> tuple:
    """
    Author the clarifying question for an ambiguous instruction via a focused LLM call.

    The model/endpoint are read from doit.cfg (load_config) so this tool is self-contained and
    does not depend on the calling agent. Returns (question, options); falls back to a generic
    question if the authoring call fails or returns nothing usable.
    """
    model_name, api_base, tool_calling = load_config()
    completion_params = {
        "model": model_name,
        "api_base": api_base,
        "messages": [
            {"role": "system", "content": CLARIFY_AUTHOR_PROMPT},
            {"role": "user", "content": instruction},
        ],
    }
    if not tool_calling and not is_openai_model(model_name):
        completion_params["num_ctx"] = CTX_NUM

    try:
        response = litellm.completion(**completion_params)
        parsed = parse_json_response(response.choices[0].message.content or "")
        question = (parsed.get("question") or "").strip()
        options = parsed.get("options") or None
        if options and not isinstance(options, list):
            options = None
    except Exception:
        question, options = "", None

    if not question:
        question = f"Your request '{instruction}' is ambiguous. Could you clarify what you mean?"
        options = None
    return question, options


# How-to question openers. Detected deterministically (in Python) so weak non-tool-calling
# models never have to CLASSIFY the request - they only have to ANSWER it via answer_howto_question.
HOWTO_PATTERNS = [
    # "how do I / how does one / how can you / how would someone ..." - many verb+subject combos.
    r"^\s*how (do|does|would|can|could|should|might|to)\b",
    r"^\s*what('?s| is| are) the (command|commands|way|steps|syntax) (to|for)\b",
    r"^\s*what('?s| is) the best way (to|for)\b",
    r"^\s*what command\b",
    r"^\s*is there (a|an|any) (command|way) (to|for|that)\b",
]


def is_howto_question(instruction: str) -> bool:
    """
    True when the instruction is phrased as a how-to question ("how would I ...",
    "what's the command to ..."). Deterministic so it does not depend on the model
    classifying correctly. Follow-ups like "execute that"/"modify it" are NOT how-to
    phrased and intentionally do not match.
    """
    text = instruction or ""
    return any(re.search(p, text, re.IGNORECASE) for p in HOWTO_PATTERNS)


# "Run the previously suggested command" openers. Detected deterministically so a weak
# non-tool-calling model never has to emit the execute turn itself (it tends to return empty
# or mislabel it). The most recent suggested_command from history is run directly instead.
EXECUTE_SUGGESTION_PATTERNS = [
    r"^\s*(execute|run)\s+(it|that|this|the\s+(command|suggestion|previous\s+command))\b",
    r"^\s*do\s+(it|that|this)\b",
    r"^\s*go\s+ahead\b",
    r"^\s*yes,?\s+(run|execute|do)\b",
]


def is_execute_suggestion_request(instruction: str) -> bool:
    """
    True when the instruction asks to run a previously suggested command ("execute that",
    "run it", "go ahead"). Deterministic, so the weak model never has to produce the execute
    turn itself.
    """
    text = instruction or ""
    return any(re.search(p, text, re.IGNORECASE) for p in EXECUTE_SUGGESTION_PATTERNS)


def resolve_cd_hoist(command: str) -> Optional[str]:
    """
    If `command` is a plain `cd <target>` - a SINGLE simple command, no chaining/piping/redirection
    /substitution - whose target resolves to an existing directory, return that absolute directory
    so it can be hoisted to the parent shell (a subprocess can't change the user's cwd). Otherwise
    return None and the command runs normally in the sandboxed subprocess.

    Bails (returns None) on compound commands, `cd -`, and non-existent targets, so we never hoist
    - and thus never skip the sandbox for - anything but a clean directory change.
    """
    if not command:
        return None
    cmd = command.strip()
    # Only a single, simple cd is hoistable. Anything chained/piped/redirected/substituted must run
    # in the subprocess (so its non-cd parts stay sandboxed) - bail.
    if any(tok in cmd for tok in (";", "&&", "||", "|", "\n", "`", "$(", ">", "<", "&")):
        return None
    m = re.match(r"^cd(?:\s+(.*))?$", cmd)
    if not m:
        return None
    target = (m.group(1) or "~").strip()
    if len(target) >= 2 and target[0] == target[-1] and target[0] in ("'", '"'):
        target = target[1:-1]
    if not target or target == "-":
        return None
    expanded = os.path.expanduser(os.path.expandvars(target))
    if not os.path.isabs(expanded):
        expanded = os.path.join(os.getcwd(), expanded)
    expanded = os.path.normpath(expanded)
    return expanded if os.path.isdir(expanded) else None


def resolve_cd_target(command: str, base_cwd: str) -> Optional[str]:
    """
    Pure path resolution for a STANDALONE `cd`, used by the command-plan runner to track the working
    directory across steps. Like `resolve_cd_hoist` but (a) resolves relative paths against `base_cwd`
    (the plan's running cwd, not `os.getcwd()`) and (b) does NO existence check - the caller validates
    with `os.path.isdir` when the step runs (a plan may `cd` into a dir an earlier step just created).
    Returns the absolute normalized target, or None for a compound/`cd -`/non-`cd` command (which the
    runner then executes normally in a subprocess).
    """
    if not command:
        return None
    cmd = command.strip()
    if any(tok in cmd for tok in (";", "&&", "||", "|", "\n", "`", "$(", ">", "<", "&")):
        return None
    m = re.match(r"^cd(?:\s+(.*))?$", cmd)
    if not m:
        return None
    target = (m.group(1) or "~").strip()
    if len(target) >= 2 and target[0] == target[-1] and target[0] in ("'", '"'):
        target = target[1:-1]
    if not target or target == "-":
        return None
    expanded = os.path.expanduser(os.path.expandvars(target))
    if not os.path.isabs(expanded):
        expanded = os.path.join(base_cwd, expanded)
    return os.path.normpath(expanded)


# Shell builtins that mutate session state and so cannot work from a subprocess. `cd` is handled
# separately (resolve_cd_hoist, value-hoisted as a path); `source` is deliberately NOT here (it
# runs a file's code in the live shell - resolve-and-explain only).
SESSION_STATE_BUILTINS = ("export", "alias", "unalias", "set", "shopt", "unset", "pushd", "popd")


# "What did I/you just do" style questions. Their answer is purely the recent history, so they are
# handled deterministically (no LLM, no command run) - a weak model otherwise tends to RUN a command
# (e.g. `ls`) to "find out" instead of just reporting. Patterns kept broad to cover common phrasings;
# anything missed falls through to the normal pipeline (prompt-guided).
ACTIVITY_QUERY_PATTERNS = [
    # "what did I/you just do" + variants
    (r"\bwhat\s+did\s+i\b.*\b(do|run|just|execute)\b", "user"),
    (r"\b(summari[sz]e|recap)\b.*\bi\b", "user"),
    (r"\bwhat\s+have\s+i\s+(done|run|been)\b", "user"),
    (r"\bremind\s+me\s+what\s+i\b", "user"),
    (r"\bmy\s+(recent|last|previous)\b.*\b(command|step|action|thing)", "user"),
    (r"\bwhat\s+did\s+(you|doit|we)\b.*\b(do|run|just|execute)\b", "doit"),
    (r"\bwhat\s+have\s+(you|we)\s+(done|run)\b", "doit"),
    # "what command(s) was/were recently run/ran/executed" - a report query (matters most for a
    # cross-session reference like "... in session 12345", handled in the activity route).
    (r"\bwhat\s+commands?\b.*\b(was|were|ran|run|recently|executed?)\b", "both"),
    (r"\b(recent|last|latest|previous)\s+commands?\b.*\bin\b", "both"),
    (r"\bwhat\s+just\s+happened\b", "both"),
    (r"\bwhat('?s| has)\s+been\s+(going\s+on|happening)\b", "both"),
    # "explain what you/I just did" / "explain that action" - explanation of a recent action
    (r"\bexplain\b.*\bwhat\s+i\b", "user"),
    (r"\bexplain\b.*\b(the|that|this)\s+command\s+i\b", "user"),
    (r"\bi\s+just\s+(ran|performed|did)\b.*\bexplain\b", "user"),
    (r"\bexplain\b.*\bwhat\s+you\b", "doit"),
    (r"\bexplain\b.*\b(that|this|the)\s+(action|command)\b", "both"),
]


def is_activity_query(instruction: str) -> Optional[str]:
    """
    If the instruction asks what was recently done (or to explain a recent action), return whose
    activity it concerns ("user", "doit", or "both"); else None. Deterministic, so these are handled
    from history rather than by running a command (which weak models tend to do).
    """
    text = (instruction or "").lower()
    for pat, subject in ACTIVITY_QUERY_PATTERNS:
        if re.search(pat, text):
            return subject
    return None


# --- Multi-window / cross-session referencing ------------------------------------------------------

# "List my terminal sessions / shell numbers" - answered deterministically from the registry (no LLM,
# no command run), so the user can discover which shell PID to reference.
# PLURAL nouns for the question/possessive forms, so a LIST query ("what are the other sessions") is
# distinguished from a SINGULAR cross-session reference ("the other window"). list/show verbs + the
# "shell numbers" form are matched explicitly.
SESSION_LIST_PATTERNS = [
    r"\b(list|show|see|view|enumerate)\b.*\b(sessions|shells|windows|terminals)\b",
    r"\bshell\s+numbers?\b",
    r"\b(what|which|how\s+many)\b.*\b(sessions|shells|windows|terminals)\b",
    r"\b(my|the|all|other|current|open|active)\s+(\w+\s+)?(sessions|shells|windows|terminals)\b",
]


def is_session_list_query(instruction: str) -> bool:
    """True when the user asks to see their open terminal sessions / shell numbers."""
    text = (instruction or "").lower()
    return any(re.search(p, text) for p in SESSION_LIST_PATTERNS)


# Explicit references to ANOTHER terminal window/session. Kept specific so a bare "them/that/it"
# (a same-session follow-up) does NOT match - that preserves per-window isolation.
CROSS_SESSION_PATTERNS = [
    r"\b(the\s+)?other\s+(window|terminal|session|shell)\b",
    r"\b(window|terminal|session|shell|pid)\s+#?\d+\b",   # "session 12345"
    r"\b#?\d+\s+(window|terminal|session|shell)\b",        # "the 12345 session"
    r"\bin\s+the\s+other\b",
    r"\b(we|i)\s+did\s+in\b",
    r"\bfrom\s+(the\s+)?(window|terminal|session|shell|other)\b",
    r"\bin\s+(window|terminal|session)\s",
    r"\belsewhere\b",
]


def is_cross_session_reference(instruction: str) -> bool:
    """True when the instruction explicitly refers to a DIFFERENT terminal window/session. Gated
    narrowly so ordinary same-session follow-ups stay isolated to the current window."""
    text = (instruction or "").lower()
    return any(re.search(p, text) for p in CROSS_SESSION_PATTERNS)


def extract_session_pid(instruction: str) -> Optional[str]:
    """The shell number from a qualified reference - either order: 'session 12345' / 'pid 12345' OR
    'the 12345 session' / '12345 window' (the digits only), else None. The CALLER checks it against
    the known session pids: a real pid matches exactly; a positional 'window 2' won't match a pid and
    falls through to fuzzy resolution."""
    text = instruction or ""
    m = (re.search(r"\b(?:window|terminal|session|shell|pid)\s+#?(\d+)\b", text, re.IGNORECASE)
         or re.search(r"\b#?(\d+)\s+(?:window|terminal|session|shell)\b", text, re.IGNORECASE))
    return m.group(1) if m else None


def _ago(iso_ts: Optional[str]) -> str:
    """A short 'Nm/Nh/Nd ago' from an ISO-8601 timestamp; '' when unknown/unparseable."""
    if not iso_ts:
        return ""
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - then).total_seconds()
    except Exception:
        return ""
    if secs < 90:
        return "just now"
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{int(secs // n)}{unit} ago"
    return "just now"


def _session_alive_tag(s: Dict[str, Any]) -> str:
    if s.get("is_current"):
        return "this window"
    alive = s.get("alive")
    return "live" if alive else ("closed" if alive is False else "")


def format_sessions_summary(sessions: List[Dict[str, Any]]) -> str:
    """A compact, numbered listing of sessions for the cross-session resolver's context."""
    lines = []
    for i, s in enumerate(sessions, 1):
        tag = _session_alive_tag(s)
        head = f"{i}. pid {s.get('pid')} - {s.get('cwd') or '(unknown dir)'}"
        meta = " - ".join(x for x in (_ago(s.get("last_active_at")), tag) if x)
        if meta:
            head += f" ({meta})"
        lines.append(head)
        for r in s.get("recent", []):
            lines.append(f"     {r}")
    return "\n".join(lines) if lines else "(no other sessions)"


def format_sessions_menu(sessions: List[Dict[str, Any]]) -> List[str]:
    """One human-readable option string per session, for the clarification menu."""
    opts = []
    for s in sessions:
        tag = _session_alive_tag(s)
        recent = s.get("recent") or []
        gist = ("; " + recent[-1]) if recent else ""
        meta = ", ".join(x for x in (_ago(s.get("last_active_at")), tag) if x)
        opts.append(f"pid {s.get('pid')} - {s.get('cwd') or '(unknown dir)'}"
                    + (f" ({meta})" if meta else "") + gist)
    return opts


def resolve_cross_session(instruction: str, sessions_summary: str) -> Dict[str, Any]:
    """
    Focused LLM call (model/endpoint from doit.cfg, like answer_howto_question) that picks WHICH other
    session a cross-session reference means and WHICH of its turns are relevant. Returns
    {"pid": str, "relevant_ids": [int], "confident": bool}; empty/!confident on any failure so the
    caller falls back to a clarifying menu.
    """
    model_name, api_base, tool_calling = load_config()
    user_content = (
        f'User instruction: "{instruction}"\n\n'
        f"Other terminal sessions (each: pid, working directory, recency, recent commands):\n"
        f"{sessions_summary}"
    )
    completion_params = {
        "model": model_name,
        "api_base": api_base,
        "messages": [
            {"role": "system", "content": CROSS_SESSION_RESOLVER_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    if not tool_calling and not is_openai_model(model_name):
        completion_params["num_ctx"] = CTX_NUM
    try:
        response = litellm.completion(**completion_params)
        parsed = parse_json_response(response.choices[0].message.content or "")
        pid = parsed.get("pid")
        rel = parsed.get("relevant_ids", [])
        return {
            "pid": str(pid) if pid not in (None, "") else "",
            "relevant_ids": [int(x) for x in rel if str(x).strip().lstrip("-").isdigit()],
            "confident": bool(parsed.get("confident", False)),
        }
    except Exception:
        return {"pid": "", "relevant_ids": [], "confident": False}


def explain_command(command: str) -> str:
    """
    Focused sub-call that explains what a shell command does, in 1-2 sentences. Used for
    "explain what you/I just did" - more reliable on a weak model than the main multi-rule agent
    (and it cannot run anything, just returns text). Empty string on failure.
    """
    if not command:
        return ""
    model_name, api_base, tool_calling = load_config()
    completion_params = {
        "model": model_name,
        "api_base": api_base,
        "messages": [
            {"role": "system", "content": "You explain shell commands concisely for a user who wants to understand them. Reply with ONLY a 1-2 sentence explanation, no preamble, no markdown."},
            {"role": "user", "content": f"Explain what this command does: {command}"},
        ],
    }
    if not tool_calling and not is_openai_model(model_name):
        completion_params["num_ctx"] = CTX_NUM
    try:
        response = litellm.completion(**completion_params)
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def parse_shell_history(raw: str) -> List[tuple]:
    """
    Parse `fc -l` output (e.g. "  501  cd ~/x") into [(index, command), ...]. Used to import the
    user's recent terminal commands (DOIT_SHELL_HISTORY) for user-awareness. Format is normalized by
    `fc -l` across bash and zsh, so a single regex handles both.
    """
    pairs = []
    for line in (raw or "").splitlines():
        m = re.match(r"^\s*(\d+)\s+(.*)$", line)
        if m:
            cmd = m.group(2).strip()
            if cmd:
                pairs.append((int(m.group(1)), cmd))
    return pairs


def parse_cmd_log(raw: str) -> List[tuple]:
    """
    Parse the per-command exit-status log (each line '<exit status>\\t<command>', written by the
    doit shell recorder) into [(line_index, status, command), ...]. line_index is the de-dup key.
    """
    out = []
    for i, line in enumerate((raw or "").splitlines(), 1):
        if "\t" in line:
            status, cmd = line.split("\t", 1)
            cmd = cmd.strip()
            if cmd:
                out.append((i, status.strip(), cmd))
    return out


def is_doit_invocation(command: str) -> bool:
    """
    True for a `doit ...` line in the user's shell history (the user invoking the agent). These are
    already represented by doit's own turns, so they are excluded from imported user commands to
    avoid double-counting.
    """
    return bool(re.match(r"^\s*doit(\s|$)", command or ""))


def resolve_session_state_hoist(command: str) -> Optional[str]:
    """
    If `command` is a SINGLE session-state builtin invocation (export/alias/set/unset/shopt/pushd/
    popd) with no command substitution, chaining, piping, redirection, or backgrounding, return the
    command to run in the parent shell. Otherwise None (it stays in the sandboxed subprocess).

    Unlike `cd`, these are hoisted as the COMMAND (not a value) because they legitimately need
    parameter expansion - e.g. `export PATH=$PATH:/x` must expand `$PATH`. The screening below is
    what makes running it in the live shell safe: with no command substitution (`$(...)`,
    backticks) and no chaining to a non-builtin, the command can ONLY mutate shell state
    (vars/aliases/options/dir stack) - it cannot execute an arbitrary program. (Limitation: an
    export/alias whose VALUE contains shell metacharacters is conservatively NOT hoisted.)
    """
    if not command:
        return None
    cmd = command.strip()
    if any(tok in cmd for tok in ("$(", "`", ";", "&&", "||", "|", ">", "<", "&", "\n")):
        return None
    parts = cmd.split(None, 1)
    if parts and parts[0] in SESSION_STATE_BUILTINS:
        return cmd
    return None


def answer_howto_question(instruction: str) -> tuple:
    """
    Answer a how-to question via a focused, single-purpose LLM call (model/endpoint from
    doit.cfg). The tight prompt has none of the multi-rule/clarification machinery, so a weak
    local model just answers instead of mis-routing to a clarification. Returns
    (explanation, suggested_command); falls back to a generic explanation on failure.
    """
    model_name, api_base, tool_calling = load_config()
    completion_params = {
        "model": model_name,
        "api_base": api_base,
        "messages": [
            {"role": "system", "content": ANSWER_HOWTO_PROMPT},
            {"role": "user", "content": instruction},
        ],
    }
    if not tool_calling and not is_openai_model(model_name):
        completion_params["num_ctx"] = CTX_NUM

    try:
        response = litellm.completion(**completion_params)
        parsed = parse_json_response(response.choices[0].message.content or "")
        explanation = (parsed.get("explanation") or "").strip()
        suggested = (parsed.get("suggested_command") or "").strip()
    except Exception:
        explanation, suggested = "", ""

    if not explanation and not suggested:
        explanation = f"I could not produce an answer for: {instruction}"
    return explanation, suggested


def ask_user_clarification(instruction: str) -> str:
    """
    Implementation of the `ask_user_clarification` tool. Invoked only when the agent decides the
    request is ambiguous. Authors the clarifying question (LLM call configured from doit.cfg),
    puts it to the user, and returns the user's answer (or the no-answer sentinel).
    """
    question, options = _author_clarification(instruction)
    return ask_clarification(question, options)


# Openers that suggest the instruction states a durable fact/preference worth REMEMBERING. A cheap
# Python gate (like the how-to / context-indicator heuristics) so the memory sub-call is skipped on
# ordinary commands instead of running every turn.
MEMORY_CANDIDATE_PATTERNS = [
    r"\bremember\b", r"\bkeep in mind\b", r"\bdon'?t forget\b", r"\bnote that\b",
    r"\bfrom now on\b", r"\bgoing forward\b", r"\bfor (the )?future\b",
    r"\balways\b", r"\bnever\b",
    r"\bi prefer\b", r"\bi like\b", r"\bi (would |'d )?want you to\b",
    r"\bthis is my\b", r"\bthat'?s my\b",
    r"\bmy [\w-]+ (folder|directory|dir|project|repo|workspace)\b",
    r"\bi changed my mind\b", r"\bask me (each|every) time\b",
]


def is_memory_candidate(instruction: str) -> bool:
    """
    True when the instruction looks like it states something durable to remember about the user.
    Deterministic gate so the memory sub-call doesn't run on every ordinary command.
    """
    text = instruction or ""
    return any(re.search(p, text, re.IGNORECASE) for p in MEMORY_CANDIDATE_PATTERNS)


def extract_memories(instruction: str, existing: List[Dict[str, Any]], executed_command: str = "") -> List[Dict[str, Any]]:
    """
    Focused memory-manager sub-call (model/endpoint from doit.cfg). Given the instruction, the
    current memories, and (optionally) the command just executed, returns a list of operations
    (add/update/delete) to apply to the store. Empty list on any failure. Built like
    `answer_howto_question` / `_author_clarification` - one job, no other machinery.
    """
    model_name, api_base, tool_calling = load_config()
    existing_block = "\n".join(
        f'- id {r.get("id")}: {r.get("content")}' for r in (existing or [])
    ) or "(none)"
    user_content = (
        f'User instruction: "{instruction}"\n'
        f'Command just executed: "{executed_command}"\n\n'
        f"Current memories:\n{existing_block}"
    )
    completion_params = {
        "model": model_name,
        "api_base": api_base,
        "messages": [
            {"role": "system", "content": MEMORY_MANAGER_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    if not tool_calling and not is_openai_model(model_name):
        completion_params["num_ctx"] = CTX_NUM

    try:
        response = litellm.completion(**completion_params)
        parsed = parse_json_response(response.choices[0].message.content or "")
        ops = parsed.get("operations", [])
        return ops if isinstance(ops, list) else []
    except Exception:
        return []


# Defining the tools using OpenAI/LiteLLM's standard format.
EXECUTE_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_bash_command",
        "description": (
            "Executes a localized bash command on the host terminal environment. "
            "Use this tool ONLY to execute the user's requested bash command. "
            "DO NOT call this tool for general knowledge, questions, or irrelevant inputs "
            "(e.g., do not generate 'echo' commands to answer questions). "
            "DO NOT call this tool when returning the rejection warning 'I do not see any previous command within the current window that applies to this'. "
            "Instead, for this warning, you MUST return a plain text response directly without any tool call."
        ),
        "parameters": BashCommandInput.model_json_schema()
    }
}

ANSWER_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": (
            "Use this ONLY to ANSWER a QUESTION the user ASKS about the shell - phrased as a question "
            "seeking knowledge (e.g. 'how do I find files over 100MB?', 'what does chmod do?') - WITHOUT "
            "executing anything. Provide an 'explanation' and, when a command applies, a 'suggested_command' "
            "the user COULD run. This tool NEVER executes the command - it only explains and suggests. "
            "DO NOT use this for an IMPERATIVE instruction to PERFORM an action ('create a file', 'go to X "
            "then make a venv then create main.py'): that is a task to DO, not a question to answer - use "
            "execute_bash_command for a single action, or execute_plan for several actions in sequence."
        ),
        "parameters": AnswerInput.model_json_schema()
    }
}

PLAN_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_plan",
        "description": (
            "Use this for an IMPERATIVE task that genuinely needs SEVERAL shell commands run IN SEQUENCE "
            "(e.g. scaffolding a project: make a directory, create files, init git). A request that lists "
            "actions to PERFORM joined by 'then' / 'and then' / 'first ... then ...' (e.g. 'go to my "
            "projects dir, then create a venv, then create main.py') is exactly this - it is a task to "
            "EXECUTE here, NOT a question to answer, so do NOT call answer_question for it. Provide an "
            "ordered list of 'steps', each a single 'command' plus a short 'explanation', and an optional "
            "one-line 'overview'. The full plan is shown to the user BEFORE anything runs; steps then run "
            "in order and STOP if one fails (so a broken step doesn't cascade). Use execute_bash_command "
            "instead for a SINGLE command (even a piped one-liner)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "overview": {
                    "type": "string",
                    "description": "Optional one-line summary of the overall goal of the plan."
                },
                "steps": {
                    "type": "array",
                    "description": "The ordered steps to run, one command each.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "A single bash command for this step."},
                            "explanation": {"type": "string", "description": "A short explanation of what this step does."}
                        },
                        "required": ["command", "explanation"]
                    }
                }
            },
            "required": ["steps"]
        }
    }
}

CLARIFY_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_user_clarification",
        "description": (
            "Call this when the user's request is genuinely ambiguous (more than one reasonable "
            "interpretation that changes the command, or a missing required detail) so you cannot "
            "produce the correct command without guessing. You do NOT need to write the question - "
            "a separate step authors the clarifying question from the request. Do NOT also call "
            "execute_bash_command in the same response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional: briefly, what is ambiguous or missing about the request."
                }
            },
            "required": []
        }
    }
}

