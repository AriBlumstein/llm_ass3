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

from fixtures import CLARIFY_AUTHOR_PROMPT, ANSWER_HOWTO_PROMPT, CTX_NUM
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


def execute_bash(command: str, verbose: bool = True) -> str:
    """
    Executes a raw bash string in an isolated subprocess under strict constraints.
    Returns stdout/stderr merged result as a single string.
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

    try:
        result = subprocess.run(
            [bash_executable, "-c", command],
            capture_output=True,
            text=True,
            timeout=20.0
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
            output += f"--- RETURN CODE ---\n{result.returncode}\n"

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
    r"^\s*how (do|would|can|could|should) (i|you)\b",
    r"^\s*how to\b",
    r"^\s*what('?s| is| are) the (command|commands|way|steps) (to|for)\b",
    r"^\s*what command\b",
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
            "Use this ONLY to ANSWER an informational or how-to question about the shell "
            "(e.g. 'how do I find files over 100MB?', 'what does chmod do?') WITHOUT executing "
            "anything. Provide an 'explanation' and, when a command applies, a 'suggested_command' "
            "the user COULD run. This tool NEVER executes the command - it only explains and "
            "suggests. To actually run a command, use execute_bash_command instead."
        ),
        "parameters": AnswerInput.model_json_schema()
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

tools_definition: List[Dict[str, Any]] = [EXECUTE_TOOL_DEF, CLARIFY_TOOL_DEF]
