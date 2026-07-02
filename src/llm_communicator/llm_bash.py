import sys
from pathlib import Path

# Add the 'src' directory to sys.path to allow sibling module imports
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fixtures import OPENAI_API_KEY, MODEL_NAME, DOIT_SYSTEM_PROMPT, DOIT_FILTER_PROMPT, LLM_CONTEXT_LIMIT, CTX_NUM, MAX_CLARIFICATION_ROUNDS
from doit_module.config_loader import load_config
import llm_communicator.history_manager as history_manager
import llm_communicator.memory_manager as memory_manager
from llm_communicator.backup_system_prompts import FALLBACK_SYSTEM_INSTRUCTION, FEWSHOT_FALLBACK, FEWSHOT_TOOLCALL
from llm_communicator.tools import (
    BashSafetyViolationError,
    BashCommandInput,
    AnswerInput,
    execute_bash,
    ask_user_clarification,
    is_howto_question,
    answer_howto_question,
    is_execute_suggestion_request,
    resolve_cd_hoist,
    resolve_cd_target,
    resolve_session_state_hoist,
    is_memory_candidate,
    extract_memories,
    parse_shell_history,
    parse_cmd_log,
    explain_exit_code,
    is_doit_invocation,
    is_activity_query,
    explain_command,
    parse_json_response,
    is_openai_model,
    is_session_list_query,
    is_cross_session_reference,
    extract_session_pid,
    resolve_cross_session,
    format_sessions_summary,
    format_sessions_menu,
    ask_clarification,
    _ago,
    NO_ANSWER_SENTINEL,
    EXECUTE_TOOL_DEF,
    ANSWER_TOOL_DEF,
    PLAN_TOOL_DEF,
    CLARIFY_TOOL_DEF,
)

import os
import re
import json
from typing import Any, Dict, List, Optional

import litellm

# Set DOIT_DEBUG=1 in the environment to print the agent's decision trace.
def _debug(*args: Any) -> None:
    if os.environ.get("DOIT_DEBUG"):
        print("[DEBUG]", *args, flush=True)

FILTER_USER_INSTRUCTION = "Does the following command modify the file system?"

HISTORY_SYSYEM_INSTRUCTION = (
    "You are a command-line history reference resolver.\n"
    "Decide if the new instruction refers to any previous commands. Return a JSON object with 'relevant_ids'.\n"
    "Rules:\n"
    "1. If the command specifies a new file name or action directly (e.g. 'create a file called klum'), it is completely independent. Return {\"relevant_ids\": []}.\n"
    "2. If it refers to previous outputs/files/results, OR asks to run/execute/modify a previously SUGGESTED command (e.g. 'execute that', 'run it', 'modify it to ...'), resolve references in chronological order, preferring the most recent match based on semantic and logical dependencies.\n"
    "   - Link to the turn that actually performed the action (e.g. touch/mkdir/ls), OR - for an 'execute it' / 'run that' / 'modify it' style follow-up - to the answer turn whose 'Suggested (not executed)' command the user now wants to run or change.\n"
    "   - DO NOT link to pure rejection/cancelled/warning turns - those that have BOTH an empty command AND no suggested command. A turn with a 'Suggested (not executed)' command IS a valid link target.\n"
    "   - A turn marked '(the USER ran this directly)' is a command the user ran themselves in the terminal; it IS a valid link target. For example, if the user manually ran 'touch klum' and now says 'delete the file I just made', link to that user turn.\n"
    "3. SAFETY CHECK: If you can match two different previous commands that are not connected, choose the most recent one (the command with the larger ID).\n"
    "4. ATTRIBUTION DEFAULTS for an unqualified reference: a bare reference with no 'I'/'you' (e.g. 'the previous command', 'that', 'why did that fail', 're-run that') -> the MOST RECENT command (largest ID), whether the user ran it or doit did. 'what did you/we just do' (you/we) -> the most recent DOIT command. 'the command I just did/ran' (I) -> the most recent command marked '(the USER ran this directly)'.\n"
    "Note: The recent command history is presented below from most recent to oldest."
)



# Cheap heuristic: does the instruction look like a follow-up that references prior context? Used to
# (a) short-circuit obviously-independent instructions before the LLM reference resolver, and (b)
# guarantee the previous command's output is replayed for output-awareness follow-ups.
CONTEXT_INDICATORS = [
    "it", "them", "that", "those", "this", "these", "like", "mean", "meant", "the command",
    "the output", "the results", "re-run", "recursively", "again", "previous", "we just", "before",
    "output", "results", "we listed", "we created", "we did", "we ran", "we made", "above", "how many",
    "execute", "run it", "run that", "modify", "do it",
    # references to the user's own recent actions (user-awareness)
    "i just", "i made", "i created", "i ran", "i deleted", "i removed", "i added",
    "the file", "the folder", "the directory", "the dir",
    # "what did you/I just do" style questions about the most recent action
    "you just", "did you", "did i", "just do", "what did",
    # output-awareness questions ABOUT a previous command's output/result/failure
    "safe to", "why did", "what was", "what does", "error", "fail", "failed", "biggest",
    "largest", "smallest", "safe", "dangerous", "risky",
]
_WORD_BOUNDED_INDICATORS = {
    "it", "them", "that", "those", "this", "these", "like", "mean", "meant",
    "again", "before", "previous", "above", "safe", "risky",
}


def instruction_has_context_indicator(instruction: str) -> bool:
    """True when the instruction contains a word/phrase suggesting it refers to prior context (a
    follow-up), e.g. 'these', 'that command', 'the output', 'why did ... fail'."""
    il = (instruction or "").lower()
    for ind in CONTEXT_INDICATORS:
        if ind in _WORD_BOUNDED_INDICATORS:
            if re.search(r"\b" + re.escape(ind) + r"\b", il):
                return True
        elif ind in il:
            return True
    return False


def _persist_prompt(instruction: str, clar_log: List[tuple]) -> str:
    """
    Fold any clarification answers into the prompt that gets stored in history, so later
    reference-resolution turns understand the disambiguation. Returns the original
    instruction unchanged when no (usable) clarification answer was collected.
    """
    if not clar_log:
        return instruction
    suffix = "; ".join(a for _, a in clar_log if a and a != NO_ANSWER_SENTINEL)
    return f"{instruction} [clarified: {suffix}]" if suffix else instruction


def _clarification_followup(instruction: str, answer: str) -> str:
    """
    Build the message fed back to the generator after the user answers a clarification. It
    restates the FULL original request with the answer applied so the model treats it as a
    self-contained instruction to execute now - not as a contextual follow-up (which would
    otherwise trip the Rule 7 'missing context' rejection).
    """
    if answer == NO_ANSWER_SENTINEL:
        return answer
    return (
        f'The user clarified the request "{instruction}": {answer}. '
        f"This fully resolves the ambiguity. Now generate the bash command for that request "
        f"applying this clarification. Do NOT ask for clarification again, and do NOT reject "
        f"this as missing context."
    )


class BashToolAgent:
    """
    State manager for our autonomous transformer execution loop.
    Maintains system memory prompts and guides LiteLLM tool/fallback interactions.
    """
    def __init__(self, api_key: Optional[str] = None, force_new: bool = False):
        # Load configuration
        self.model_name, self.api_base, self.tool_calling = load_config()

        # Set API key for LiteLLM if provided/found
        key = api_key or OPENAI_API_KEY
        if key:
            # LiteLLM looks at OPENAI_API_KEY for openai models
            os.environ["OPENAI_API_KEY"] = key

        if force_new:
            history_manager.clear_history()

        self.system_prompt = DOIT_SYSTEM_PROMPT
        if not self.tool_calling:
            self.system_prompt += "\n\n" + FALLBACK_SYSTEM_INSTRUCTION

        # Persistent memory: inject known facts about the user into the system prompt on every
        # invocation (empty for new users). Loaded fresh from the global store, so it is
        # session/cwd-independent and reaches the main agent and the clarification decision. It
        # goes into the SYSTEM message only - replayed history turns come after it, so history
        # reference-resolution/replay is unaffected.
        memory_block = memory_manager.render_memories()
        if memory_block:
            self.system_prompt += "\n\n" + memory_block

        self.conversation_history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    def _analyze_references(self, instruction: str, history_metadata: List[Dict[str, Any]]) -> List[int]:
        """
        Queries the same configured LLM to classify if the new instruction is connected
        to any of the previous ones. Returns a list of relevant integer IDs.
        """
        if not history_metadata:
            return []

        # Keep turns that either executed a command OR proposed one via answer_question
        # (suggested_command). Pure conversational/rejection turns carry neither and are not linkable.
        # USER turns (commands the user ran directly) ARE included, so "delete the file I just made"
        # can resolve to a file the user created manually - the strongest place to surface it is the
        # replayed conversation, right next to the instruction.
        history_metadata = [t for t in history_metadata
                            if t.get("command") or t.get("suggested_command")]
        if not history_metadata:
            return []

        # Quick heuristic check for independent instructions to assist small models.
        if not instruction_has_context_indicator(instruction):
            return []

        formatted_history = "\n".join([
            f"- [ID: {t['id']}] "
            + ("(the USER ran this directly) " if t.get("source") == "user" else "")
            + f"Prompt: \"{t['prompt']}\" | Command: \"{t['command']}\""
            + (f" | Suggested (not executed): \"{t['suggested_command']}\"" if t.get("suggested_command") else "")
            for t in reversed(history_metadata)
        ])


        user_content = (
            f"New instruction: \"{instruction}\"\n\n"
            f"Recent command history:\n{formatted_history}"
        )

        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": HISTORY_SYSYEM_INSTRUCTION},
                    {"role": "user", "content": user_content}
                ]
            }
            if self.api_base:
                completion_params["api_base"] = self.api_base
            if not self.tool_calling and not is_openai_model(self.model_name):
                completion_params["num_ctx"] = CTX_NUM

            response = litellm.completion(**completion_params)
            content = response.choices[0].message.content.strip()
            
            parsed = parse_json_response(content)
            relevant_ids = parsed.get("relevant_ids", [])
            
            valid_ids = {t["id"] for t in history_metadata}
            return [int(rid) for rid in relevant_ids if int(rid) in valid_ids]
        except Exception:
            return []

    def _resolve_transitive_dependencies(self, initial_ids: List[int]) -> List[int]:
        """
        Recursively resolves all transitively chained dependencies for the given IDs
        using the history database.
        """
        if not initial_ids:
            return []
            
        path = history_manager.get_history_file_path()
        if not path.exists():
            return sorted(initial_ids)
            
        deps_map = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    turn = json.loads(line)
                    turn_id = turn.get("id")
                    if turn_id is not None:
                        deps_map[turn_id] = turn.get("relevant_ids", [])
        except Exception:
            pass
            
        resolved = set()
        to_visit = list(initial_ids)
        while to_visit:
            curr = to_visit.pop(0)
            if curr not in resolved:
                resolved.add(curr)
                for dep in deps_map.get(curr, []):
                    if dep not in resolved:
                        to_visit.append(dep)
                        
        return sorted(list(resolved))

    def _filter_bash(self, command: str) -> tuple[bool, str]:
        """Filter for bash commands using LLM as a judge to determine if a command will modify file system."""
        completion_params = {
            "model": self.model_name,
            "api_base": self.api_base,
            "messages": [
                {
                    "role": "system",
                    "content": DOIT_FILTER_PROMPT
                },
                {
                    "role": "user",
                    "content": FILTER_USER_INSTRUCTION + command
                }
            ]
        }
        if not self.tool_calling and not is_openai_model(self.model_name):
            completion_params["num_ctx"] = CTX_NUM

        response = litellm.completion(**completion_params)
        content = response.choices[0].message.content.strip()

        decision = False
        explanation = ""

        # Check for new format: "DECISION: YES" or "DECISION: NO"
        match_dec = re.search(r"\bDECISION:\s*(YES|NO|TRUE|FALSE)\b", content, re.IGNORECASE)
        if match_dec:
            val = match_dec.group(1).upper()
            if val in ("YES", "TRUE"):
                decision = True
        else:
            # Fallback to older format parsing
            if content.startswith("TRUE:"):
                decision = True
                explanation = content[5:].strip()

        # Extract explanation
        match_exp = re.search(r"\bEXPLANATION:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if match_exp:
            explanation = match_exp.group(1).strip()
        elif not explanation:
            # Fallback if header is missing
            explanation = content

        if not decision:
            return False, ""

        return True, explanation

    @staticmethod
    def _read_user_commands() -> List[tuple]:
        """
        The user's recent terminal commands as [(index, status, cwd, command)]. Prefers the exit-status
        log (DOIT_CMD_LOG, written by the shell recorder - status is "0"/"N", cwd is the dir the command
        ran in); falls back to `fc -l` history (DOIT_SHELL_HISTORY, status and cwd None). Empty when
        there is no shell integration.
        """
        log = os.environ.get("DOIT_CMD_LOG")
        if log:
            try:
                with open(log, "r", encoding="utf-8") as f:
                    entries = parse_cmd_log(f.read())
                if entries:
                    return entries
            except Exception:
                pass
        raw = os.environ.get("DOIT_SHELL_HISTORY")
        if raw:
            return [(n, None, None, cmd) for n, cmd in parse_shell_history(raw)]
        return []

    @staticmethod
    def _user_cmd_output(status) -> str:
        """A synthetic 'output' for a user command, reflecting its real exit status when known. An
        EMPTY output makes the model read the command as failed, so we always say what happened."""
        if status is None:
            return "[Ran by the user directly in the terminal; exit status not captured by doit.]"
        s = str(status).strip()
        if s == "0":
            return "[Ran by the user directly in the terminal; completed successfully (exit 0). Output not captured by doit.]"
        # No output was captured for a user command, so the exit code is the only failure signal -
        # interpret it (e.g. 127 -> command not found) so "why did that fail?" gets a real reason.
        meaning = explain_exit_code(s)
        detail = f"exit {s}: {meaning}" if meaning else f"exit {s}"
        return f"[Ran by the user directly in the terminal; it FAILED ({detail}). Output not captured by doit.]"

    def _sync_user_history(self) -> None:
        """
        Import the user's manual shell commands into the per-session history as source="user" turns,
        interleaved in order with doit's own turns. Uses the exit-status log when available (so user
        commands carry real success/failure), else `fc -l`. De-dups via hist_n (only commands newer
        than the high-water mark), drops `doit ...` invocation lines (already doit's own turns). Best
        effort; no-op without shell integration.
        """
        entries = self._read_user_commands()
        if not entries:
            return
        try:
            last_seen = history_manager.get_last_user_hist_n()
            for idx, status, cwd, cmd in entries:
                if idx <= last_seen or is_doit_invocation(cmd):
                    continue
                history_manager.append_history_turn(
                    prompt="", command=cmd, output=self._user_cmd_output(status),
                    relevant_ids=[], source="user", hist_n=idx, cwd=cwd,
                )
        except Exception as e:
            _debug("USER-HISTORY sync failed:", e)

    def _activity_items(self, subject: str) -> List[Dict[str, Any]]:
        """Recent command-bearing turns for the subject, with consecutive duplicates collapsed."""
        recent = [t for t in history_manager.get_history_metadata(limit=20) if t.get("command")]
        if subject == "user":
            items = [t for t in recent if t.get("source") == "user"]
        elif subject == "doit":
            items = [t for t in recent if t.get("source") == "doit"]
        else:
            items = recent
        # Collapse consecutive identical commands (e.g. the user ran `touch klum` several times).
        deduped: List[Dict[str, Any]] = []
        for t in items:
            if deduped and deduped[-1]["command"] == t["command"] and deduped[-1].get("source") == t.get("source"):
                continue
            deduped.append(t)
        return deduped

    def _answer_activity_query(self, subject: str) -> str:
        """
        Answer a "what did I/you just do" question deterministically from the recent history (no LLM,
        no command run). Correct attribution: the user's own commands are "you ran", doit's own
        actions are "I ran".
        """
        lead = {"user": "Your most recent command was",
                "doit": "My most recent action was"}.get(subject, "The most recent command was")
        items = self._activity_items(subject)
        if not items:
            return "I don't have any recorded recent activity to report."

        items = items[-6:]
        out = [f"{lead}: {items[-1]['command']}"]
        if len(items) > 1:
            out.append("Recent activity (oldest to newest):")
            for t in items:
                who = "you ran" if t.get("source") == "user" else "I ran"
                out.append(f"  - {who}: {t['command']}")
        return "\n".join(out)

    def _explain_recent_action(self, subject: str) -> str:
        """Answer an 'explain what you/I just did' query: report the most recent command and explain
        it via a focused sub-call (which cannot run anything)."""
        items = self._activity_items(subject)
        if not items:
            return "I don't have a recent command to explain."
        cmd = items[-1]["command"]
        who = "You" if items[-1].get("source") == "user" else "I"
        explanation = explain_command(cmd)
        return f"{who} ran `{cmd}`." + (f" {explanation}" if explanation else "")

    def _build_activity_block(self) -> str:
        """
        The user-awareness block injected into the system prompt: the CURRENT DIRECTORY plus a short,
        ordered, tagged RECENT TERMINAL ACTIVITY list (both the user's manual commands and doit's own
        actions). This is what lets the agent answer "what did I just do", ground commands in the cwd,
        act on user-created files, and reason about what came last (undo).
        """
        try:
            cwd = os.getcwd()
        except Exception:
            cwd = "(unknown)"
        lines = [f"CURRENT DIRECTORY: {cwd}"]

        recent = [t for t in history_manager.get_history_metadata(limit=15) if t.get("command")]
        if recent:
            lines.append(
                "RECENT TERMINAL ACTIVITY (oldest first; [user] = the user ran it directly, "
                "[doit] = doit ran it):"
            )
            for t in recent:
                tag = "user" if t.get("source") == "user" else "doit"
                # Show WHERE a user command ran when it differs from the current directory, so a
                # later "re-run that / why did it fail" can target the right place (Rule 11).
                t_cwd = t.get("cwd")
                where = f" (in {t_cwd})" if tag == "user" and t_cwd and t_cwd != cwd else ""
                lines.append(f"  [{tag}] {t['command']}{where}")
        return "\n".join(lines)

    def _store_memories(self, instruction: str, executed_command: str = "") -> None:
        """
        Persist durable facts/preferences from this instruction via the focused memory-manager
        sub-call, applying its add/update/delete operations to the global store. Best-effort:
        memory handling must never break the turn. Independent of the action, so one instruction
        can both act and be remembered.
        """
        try:
            existing = memory_manager.load_memories()
            ops = extract_memories(instruction, existing, executed_command)
            _debug("MEMORY ops:", ops)
            applied = memory_manager.apply_operations(ops)
            for op in applied:
                kind = (op.get("op") or "").lower()
                if kind == "add":
                    print(f"[Memory] Noted: {op.get('content', '').strip()}")
                elif kind == "update":
                    print("[Memory] Updated a saved preference.")
                elif kind == "delete":
                    print("[Memory] Forgot a saved preference.")
        except Exception as e:
            _debug("MEMORY store failed:", e)

    def _hoist_cd(self, abspath: str) -> str:
        """
        Hoist a `cd` to the PARENT shell. A subprocess can never change the user's cwd, so when the
        doit shell function is active it has set DOIT_CD_FILE; we write the resolved directory there
        and the function runs `cd` in the user's shell. Without that integration we can't move the
        shell, so we print the command for the user to run. `cd` produces no stdout, so nothing the
        agent might later reference is lost by not running it in the subprocess. Returns the marker
        stored as this turn's output, so the turn is still recorded in history.
        """
        cd_file = os.environ.get("DOIT_CD_FILE")
        if cd_file:
            try:
                with open(cd_file, "w", encoding="utf-8") as f:
                    f.write(abspath)
                print(f"[doit] changed directory to {abspath}")
                # The shell will apply this `cd` only AFTER doit exits, so the registry cwd recorded
                # at the top of this turn (os.getcwd(), the pre-cd dir) is now stale. Update it to the
                # target so OTHER windows immediately see this session's new location.
                history_manager.write_session_meta(cwd=abspath)
            except Exception as e:
                print(f"[doit] could not record directory change: {e}")
        else:
            print(f"[doit] shell integration is not active; to move there run:\n  cd {abspath}")
        return f"[changed directory to {abspath}]"

    def _hoist_shell(self, command: str) -> str:
        """
        Hoist a session-state builtin (export/alias/set/unset/shopt/pushd/popd) to the PARENT shell
        via DOIT_SHELL_FILE; the doit shell function runs it there so it persists. The command was
        screened by resolve_session_state_hoist (no substitution/chaining/redirection), so running
        it in the live shell can only mutate shell state, not execute an arbitrary program. These
        builtins produce no stdout the agent would reference, so history/capture are unaffected.
        """
        shell_file = os.environ.get("DOIT_SHELL_FILE")
        if shell_file:
            try:
                with open(shell_file, "w", encoding="utf-8") as f:
                    f.write(command)
                print(f"[doit] applied to your shell: {command}")
            except Exception as e:
                print(f"[doit] could not apply to shell: {e}")
        else:
            print(f"[doit] shell integration is not active; to apply this run:\n  {command}")
        return f"[applied to shell: {command}]"

    def _filter_confirm(self, command: str) -> Optional[str]:
        """
        Run the LLM filesystem-modification judge on `command`; if it judges the command modifies the
        filesystem, ask the user [y/N]. Returns a cancellation marker string if the user declines,
        else None (proceed). Shared by every execution path - subprocess commands AND hoisted
        cd/session-state commands - so the filter vets them all.
        """
        modifies, filter_explanation = self._filter_bash(command)
        if modifies:
            print(f"This command will modify your file system: {filter_explanation}")
            user_choice = input("Do you want to continue? [y/N]: ").strip().lower()
            if user_choice not in ('y', 'yes'):
                return "[Cancelled: User declined to execute command that modifies the file system]"
        return None

    def _dispatch_command(self, command: str) -> str:
        """
        Run a generated command. Every command is first vetted by the filesystem-modification filter
        (_filter_confirm), including hoisted ones - they run in the PARENT shell, so they must not skip
        the safety gate. A plain `cd` is then value-hoisted (see _hoist_cd); a single session-state
        builtin (export/alias/set/unset/shopt/pushd/popd) is command-hoisted (see _hoist_shell);
        everything else runs in the sandboxed subprocess (which also applies the regex blacklist).
        """
        cd_target = resolve_cd_hoist(command)
        shell_cmd = resolve_session_state_hoist(command) if cd_target is None else None

        if cd_target is not None or shell_cmd is not None:
            cancelled = self._filter_confirm(command)   # vet the hoisted command too
            if cancelled is not None:
                return cancelled
            if cd_target is not None:
                _debug("CD hoist:", cd_target)
                return self._hoist_cd(cd_target)
            _debug("SHELL-STATE hoist:", shell_cmd)
            return self._hoist_shell(shell_cmd)

        return self._execute_with_confirmation(command)

    def _execute_with_confirmation(self, command: str) -> str:
        """
        Run a command through the two safety layers: the LLM filesystem-modification judge
        (_filter_confirm, asks [y/N] before a modifying command) and the regex blacklist inside
        execute_bash. Returns the execution output (or a cancelled/error marker). Shared by the
        deterministic "execute that" route so suggestions get the same safety treatment as
        model-generated commands.
        """
        try:
            cancelled = self._filter_confirm(command)
            if cancelled is not None:
                return cancelled
            return execute_bash(command)
        except BashSafetyViolationError as safety_err:
            return f"[Error: {str(safety_err)}]"
        except Exception as e:
            return f"[Error: {str(e)}]"

    @staticmethod
    def _plan_step_failed(result: str) -> bool:
        """Whether an executed plan step's output indicates failure - so the plan stops instead of
        cascading. Reads the markers `execute_bash` emits: a non-zero `(FAILED)` return code, an
        `[Error ...]` marker, or the timeout message."""
        r = result or ""
        return ("(FAILED)" in r) or r.lstrip().startswith("[Error") or "Terminated due to exceeding" in r

    def _run_plan(self, steps: List[Dict[str, Any]], overview: str = "") -> str:
        """
        Execute a multi-step plan: SHOW the whole plan first (so the user sees what will happen),
        confirm ONCE, then run the steps IN ORDER, STOPPING at the first failure so a broken step
        does not cascade. Each step still passes the regex safety blacklist (via execute_bash).
        Returns a transcript stored as the turn's output.
        """
        steps = [s for s in steps if isinstance(s, dict) and s.get("command")]
        if not steps:
            return "[Error: the plan had no runnable steps]"

        header = "[PLAN] " + (overview.strip() + "\n" if overview.strip() else "") + \
                 "The following steps will run in order:"
        print(header)
        for i, s in enumerate(steps, 1):
            expl = s.get("explanation", "")
            print(f"  {i}. {s['command']}" + (f"   - {expl}" if expl else ""))

        try:
            choice = input(f"Run this {len(steps)}-step plan? [y/N]: ").strip().lower()
        except EOFError:
            choice = ""
        if choice not in ("y", "yes"):
            msg = "[Cancelled: User declined the plan]"
            print(msg)
            return msg

        transcript: List[str] = []
        start_cwd = os.getcwd()
        cwd = start_cwd                  # the plan's running directory (threaded across steps)
        shell_state: List[str] = []      # accumulated session-state (export/alias/...) for later steps + hoist
        completed = False

        def _stop(i: int, line: str) -> None:
            stop = (f"[Plan STOPPED at step {i}/{len(steps)}: it failed, so the remaining "
                    f"{len(steps) - i} step(s) were NOT run.]")
            print(line); print(stop)
            transcript.append(f"[step {i}/{len(steps)}] {steps[i-1]['command']}\n{line}")
            transcript.append(stop)

        for i, s in enumerate(steps, 1):
            cmd = s["command"]
            print(f"\n[STEP {i}/{len(steps)}] {cmd}")

            # Standalone `cd`: no output and pointless to subprocess (discarded), so TRACK it - the plan's
            # cwd moves for later steps. Validate in Python (a missing dir is a failed step that stops).
            cd_target = resolve_cd_target(cmd, cwd)
            if cd_target is not None:
                if os.path.isdir(cd_target):
                    cwd = cd_target
                    line = f"[plan: cd -> {cwd}]"
                    print(line)
                    transcript.append(f"[step {i}/{len(steps)}] {cmd}\n{line}")
                    continue
                _stop(i, f"[Error: cd: no such directory: {cd_target}]")
                break

            # Session-state builtin (export/alias/set/...): accumulate; applied to later steps + hoisted.
            shell_cmd = resolve_session_state_hoist(cmd)
            if shell_cmd is not None:
                shell_state.append(shell_cmd)
                line = f"[plan: shell-state -> {shell_cmd}]"
                print(line)
                transcript.append(f"[step {i}/{len(steps)}] {cmd}\n{line}")
                continue

            # Normal step: run in the plan's cwd with accumulated shell-state applied.
            try:
                out = execute_bash(cmd, cwd=cwd, prelude=shell_state or None)
            except BashSafetyViolationError as safety_err:
                out = f"[Error: {str(safety_err)}]"
            except Exception as e:
                out = f"[Error: {str(e)}]"
            print(out)
            transcript.append(f"[step {i}/{len(steps)}] {cmd}\n{out}")
            if self._plan_step_failed(out):
                stop = (f"[Plan STOPPED at step {i}/{len(steps)}: it failed, so the remaining "
                        f"{len(steps) - i} step(s) were NOT run.]")
                print(stop)
                transcript.append(stop)
                break
        else:
            completed = True
            transcript.append(f"[Plan complete: all {len(steps)} steps succeeded.]")

        # Hoist the plan's NET cwd / shell-state to the parent shell (like a single-command hoist), but
        # only when the plan completed - a failed plan leaves the shell where it was.
        if completed:
            if cwd != start_cwd and os.path.isdir(cwd):
                self._hoist_cd(cwd)
            if shell_state:
                self._hoist_shell("; ".join(shell_state))
        return "\n".join(transcript)

    def _build_tools(self, include_clarification: bool) -> List[Dict[str, Any]]:
        """
        Tools offered to the generator. The clarification tool is withdrawn on the final round
        so the model must commit to a command (or an answer) instead of asking again.
        """
        if include_clarification:
            return [EXECUTE_TOOL_DEF, ANSWER_TOOL_DEF, PLAN_TOOL_DEF, CLARIFY_TOOL_DEF]
        return [EXECUTE_TOOL_DEF, ANSWER_TOOL_DEF, PLAN_TOOL_DEF]

    def _format_session_list(self) -> str:
        """Human-readable listing of all known terminal sessions (for the 'list the shell numbers'
        route): each by shell PID + directory + recency + live/closed/this-window tag."""
        sessions = history_manager.list_sessions()
        if not sessions:
            return "No terminal sessions on record yet."
        lines = ["Your terminal sessions (shell PID - directory - last active):"]
        for s in sessions:
            alive = s.get("alive")
            tag = "this window" if s.get("is_current") else ("live" if alive else ("closed" if alive is False else ""))
            meta = ", ".join(x for x in (_ago(s.get("last_active_at")), tag) if x)
            lines.append(f"  pid {s.get('pid')} - {s.get('cwd') or '(unknown dir)'}"
                         + (f" ({meta})" if meta else ""))
        return "\n".join(lines)

    def _match_session_answer(self, answer, others, options):
        """Map a clarification answer (a chosen menu option, a typed pid, or a cwd) back to a session."""
        if not answer or answer == NO_ANSWER_SENTINEL:
            return None
        for s, opt in zip(others, options):
            if answer == opt:
                return s
        pid = extract_session_pid(answer) or "".join(ch for ch in answer if ch.isdigit())
        for s in others:
            if pid and s.get("pid") == pid:
                return s
        for s in others:
            if s.get("cwd") and s["cwd"] in answer:
                return s
        return None

    def _resolve_target_session(self, instruction: str):
        """
        Resolve an EXPLICIT cross-window reference to a target session. Strategy (per spec): an
        explicit pid -> that session; else the CrossSessionResolver picks one (content/recency); else
        the single other session; else a clarifying menu. Returns (target_session_dict, resolver_ids)
        - (None, []) when there are no other sessions or the target can't be determined.
        """
        others = history_manager.list_other_sessions()
        if not others:
            print("[doit] No other terminal sessions found to reference.")
            return None, []
        known = {s["pid"]: s for s in others}

        pid_hint = extract_session_pid(instruction)
        if pid_hint and pid_hint in known:
            return known[pid_hint], []                    # exact pid -> deterministic

        res = resolve_cross_session(instruction, format_sessions_summary(others))
        if res["pid"] in known and (res["confident"] or len(others) == 1):
            return known[res["pid"]], res["relevant_ids"]
        if len(others) == 1:
            return others[0], res.get("relevant_ids", [])

        options = format_sessions_menu(others)            # ambiguous -> ask the user to choose
        answer = ask_clarification("Which terminal session do you mean?", options)
        target = self._match_session_answer(answer, others, options)
        if target is None:
            print("[doit] Could not determine which session you meant.")
        return target, []

    def _answer_cross_session_activity(self, instruction: str):
        """
        Report what was done in ANOTHER window (a cross-session "what did I do in the other window"
        query): resolve the target session and list its recent activity from that session's history -
        no LLM beyond resolution, no command run. Returns the report string, or None if no target
        could be resolved (so the caller falls back to the current-session activity report).
        """
        target, _ = self._resolve_target_session(instruction)
        if target is None:
            return None
        target_dir = history_manager.get_session_dir(target["pid"])
        items = [t for t in history_manager.get_history_metadata(limit=20, session_dir=target_dir)
                 if t.get("command")]
        deduped: List[Dict[str, Any]] = []
        for t in items:
            if deduped and deduped[-1]["command"] == t["command"] and deduped[-1].get("source") == t.get("source"):
                continue
            deduped.append(t)
        items = deduped[-6:]
        cwd = target.get("cwd") or "(unknown dir)"
        if not items:
            return f"Your other session (pid {target['pid']}, {cwd}) has no recorded activity."
        lines = [f"In your other session (pid {target['pid']}, {cwd}):",
                 f"Most recent: {items[-1]['command']}"]
        if len(items) > 1:
            lines.append("Recent activity (oldest to newest):")
            for t in items:
                who = "you ran" if t.get("source") == "user" else "doit ran"
                lines.append(f"  - {who}: {t['command']}")
        return "\n".join(lines)

    def _resolve_cross_session_turns(self, instruction: str) -> List[str]:
        """
        Resolve an EXPLICIT cross-window reference to another session's relevant turns, returned as
        tagged replay notes. Relevant turn ids come from the resolver, else `_analyze_references` over
        the target session, else its most recent commands. Empty list if nothing usable.
        """
        target, res_ids = self._resolve_target_session(instruction)
        if target is None:
            return []

        target_dir = history_manager.get_session_dir(target["pid"])
        target_meta = history_manager.get_history_metadata(limit=LLM_CONTEXT_LIMIT, session_dir=target_dir)
        valid = {m["id"] for m in target_meta}
        ids = [i for i in res_ids if i in valid]
        if not ids:
            ids = [i for i in self._analyze_references(instruction, target_meta) if i in valid]
        if not ids:
            ids = [m["id"] for m in target_meta if m.get("command")][-3:]

        cwd = target.get("cwd") or "(unknown dir)"
        notes: List[str] = []
        for t in history_manager.get_full_turns(ids, session_dir=target_dir):
            out = t.get("output", "")
            if out and len(out) > 2000:
                out = out[:2000] + "\n... [truncated]"
            note = f'[from your other terminal session (pid {target["pid"]}) in {cwd}]: "{t.get("prompt", "")}"'
            cmd = t.get("command") or t.get("suggested_command")
            if cmd:
                note += f" -> ran `{cmd}`"
            if out:
                note += f"\n{out}"
            notes.append(note)
        _debug("CROSS-SESSION: target pid", target["pid"], "ids", ids)
        return notes

    def run_single(self, instruction: str) -> None:
        """
        Coordinates a single user query execution.
        """
        # 0. Record/refresh this session's registry entry (pid, cwd, last-active) so OTHER windows can
        # discover and reference it. Best-effort, every path.
        history_manager.write_session_meta()

        # 0a. User awareness: import the user's manual shell commands (since the last turn) into the
        # per-session history as source="user" turns, so doit is aware of what the user did directly
        # and in what order relative to its own actions. Runs first, on every path.
        self._sync_user_history()

        # 0a-ii. Deterministic activity-query route (all modes). "what did I/you just do" is answered
        # straight from the recorded history - no LLM, no command run - so a weak tool-calling model
        # can't decide to RUN a command (e.g. `ls`) to "find out" instead of just reporting. Phrasings
        # the regex misses fall through to the normal pipeline (prompt-guided).
        activity_subject = is_activity_query(instruction)
        if activity_subject is not None:
            _debug("ACTIVITY-QUERY route:", activity_subject)
            # If the activity question EXPLICITLY targets another window ("what did I do in the other
            # window", "what did I run in session 12345"), report THAT session instead of this one.
            # Falls back to the current-session report when no other session can be resolved.
            if is_cross_session_reference(instruction):
                _debug("CROSS-SESSION ACTIVITY route")
                cross_answer = self._answer_cross_session_activity(instruction)
                if cross_answer is not None:
                    print(cross_answer)
                    history_manager.append_history_turn(instruction, "", cross_answer, [])
                    return
            # "explain ..." wants an explanation of the recent command (focused sub-call); a plain
            # "what did ..." wants a report (no LLM). Either way, no command is run to find out.
            if "explain" in instruction.lower():
                answer = self._explain_recent_action(activity_subject)
            else:
                answer = self._answer_activity_query(activity_subject)
            print(answer)
            history_manager.append_history_turn(instruction, "", answer, [])
            return

        # 0a-iii. Deterministic "list my sessions / shell numbers" route (all modes). Answered from
        # the session registry - no LLM, no command run - so the user can see which shell PID to
        # reference for cross-window requests.
        if is_session_list_query(instruction):
            _debug("SESSION-LIST route")
            answer = self._format_session_list()
            print(answer)
            history_manager.append_history_turn(instruction, "", answer, [])
            return

        # 0. Deterministic how-to route (fallback mode only). Weak non-tool-calling models cannot
        # reliably CLASSIFY a how-to question (they mis-route it to clarification), but they can
        # ANSWER one. So we detect the how-to phrasing in Python and hand it to a focused, single
        # purpose sub-call - no multi-rule prompt, no clarification machinery to copy. The answer
        # is persisted with its suggested_command so a later "execute that" can run it. Native
        # tool-calling models route this correctly via Rule 9, so they keep that path.
        if not self.tool_calling and is_howto_question(instruction) and not is_cross_session_reference(instruction):
            explanation, suggested = answer_howto_question(instruction)
            _debug("HOWTO route:", repr(explanation), "| suggested:", repr(suggested))
            print(explanation)
            if suggested:
                print(f"\nSuggested command (not executed): {suggested}")
            history_manager.append_history_turn(
                instruction, "", explanation, [], suggested_command=suggested
            )
            return

        # 0b. Deterministic "execute that" route (fallback mode only). The weak model often
        # returns empty or mislabels a request to run a previously suggested command, so we
        # resolve it ourselves: pull the most recent suggested_command from history and run it
        # through the SAME safety pipeline. Only fires when such a suggestion exists; otherwise
        # falls through to normal handling. Native tool-calling models handle this via Rule 9.
        if not self.tool_calling and is_execute_suggestion_request(instruction):
            found = history_manager.get_latest_suggested_command()
            if found:
                src_id, suggested = found
                _debug("EXECUTE-SUGGESTION route: src_id=", src_id, "command=", repr(suggested))
                print(f"[EXECUTING SUGGESTED] {suggested}")
                output = self._execute_with_confirmation(suggested)
                print(f"[RESULT] Shell Response:\n{output}")
                history_manager.append_history_turn(instruction, suggested, output, [src_id])
                return
            _debug("EXECUTE-SUGGESTION route: no prior suggested_command; falling through")

        # 1-3. Resolve which prior turns to replay. An EXPLICIT cross-window reference ("the other
        # terminal", "session 12345", "the folder task we did in the other window") pulls the relevant
        # turns from ANOTHER session and, by design, does NOT mix in this window's history (the
        # reference is explicitly elsewhere). Otherwise we resolve within THIS session only - the
        # default that keeps windows isolated.
        cross_session_notes: List[str] = []
        if is_cross_session_reference(instruction):
            _debug("CROSS-SESSION route")
            cross_session_notes = self._resolve_cross_session_turns(instruction)

        if cross_session_notes:
            llm_relevant_ids = []
            relevant_turns = []
        else:
            metadata = history_manager.get_history_metadata(limit=LLM_CONTEXT_LIMIT)
            llm_relevant_ids = self._analyze_references(instruction, metadata)
            relevant_ids = self._resolve_transitive_dependencies(llm_relevant_ids)
            # Output-awareness safety net: for a follow-up, ALWAYS make the most recent command that
            # has real output available, so "why did that fail?" / "which of these is safe to delete?"
            # work even if the resolver linked nothing (or an output-less user command). Deterministic;
            # independent instructions (no context indicator) are untouched.
            if instruction_has_context_indicator(instruction):
                latest_out = history_manager.get_latest_output_turn_id()
                if latest_out is not None and latest_out not in relevant_ids:
                    relevant_ids = sorted(relevant_ids + [latest_out])
            relevant_turns = history_manager.get_full_turns(relevant_ids)

        # 4. Reconstruct conversation history starting with system prompt. Append the user-awareness
        # block (current directory + recent terminal activity) to the system message - built per-run
        # (after the sync) like the memory block, in the SYSTEM message only so history replay is
        # undisturbed.
        system_content = self.system_prompt
        activity_block = self._build_activity_block()
        if activity_block:
            system_content += "\n\n" + activity_block
        self.conversation_history = [
            {
                "role": "system",
                "content": system_content
            }
        ]
        
        # 5. Populate history with relevant turns formatted correctly
        if self.tool_calling:
            # Seed with in-context few-shot demonstrations of the clarification decision
            # (ambiguous -> ask_user_clarification; clear -> execute_bash_command).
            self.conversation_history.extend(FEWSHOT_TOOLCALL)

            for turn in relevant_turns:
                # A user turn is a command the user ran DIRECTLY in the terminal: replay it as a
                # plain note (no doit tool call/result), so the agent treats it as the user's action.
                # Include the "it ran" marker so the agent doesn't read empty output as a failure.
                if turn.get("source") == "user":
                    where = f" in {turn['cwd']}" if turn.get("cwd") else ""
                    note = f"[I ran this command directly in the terminal{where}]: {turn.get('command', '')}"
                    if turn.get("output"):
                        note += f"\n{turn['output']}"
                    self.conversation_history.append({"role": "user", "content": note})
                    continue

                self.conversation_history.append({
                    "role": "user",
                    "content": turn["prompt"]
                })

                command = turn.get("command", "")
                suggested = turn.get("suggested_command", "")
                output = turn.get("output", "")
                if output and len(output) > 2000:
                    output = output[:2000] + "\n... [TRUNCATED due to length]"

                if command:
                    tool_call_id = f"call_{turn['id']}"
                    self.conversation_history.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "execute_bash_command",
                                "arguments": json.dumps({
                                    "command": command,
                                    "explanation": f"execute {command}"
                                })
                            }
                        }]
                    })
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "execute_bash_command",
                        "content": output
                    })
                elif suggested:
                    # Replay an answer turn: the model proposed (but did not run) a command.
                    # Surfacing it as the answer_question tool call lets a follow-up like
                    # "execute it" re-emit it as execute_bash_command, or "modify it" revise it.
                    tool_call_id = f"call_{turn['id']}"
                    self.conversation_history.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "answer_question",
                                "arguments": json.dumps({
                                    "explanation": output,
                                    "suggested_command": suggested
                                })
                            }
                        }]
                    })
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "answer_question",
                        "content": "[Answer delivered to the user. Suggested command was NOT executed.]"
                    })
                else:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": output
                    })

            # Cross-session context: turns pulled from ANOTHER window, tagged, injected right before
            # the instruction so the agent applies the task HERE (Rule 12 re-grounds in the cwd).
            for note in cross_session_notes:
                self.conversation_history.append({"role": "user", "content": note})

            # Append the user's current instruction
            self.conversation_history.append({
                "role": "user",
                "content": instruction
            })
        else:
            # Fallback non-tool-calling mode: strict alternating user/assistant roles.
            # Seed with in-context few-shot demonstrations of the clarification decision so
            # weak local models reliably recognize ambiguous requests (e.g. "by date").
            self.conversation_history.extend(FEWSHOT_FALLBACK)

            user_content = ""
            for turn in relevant_turns:
                # A user turn (command the user ran directly) is folded into the running user_content
                # so it stays right next to the instruction and never breaks user/assistant alternation.
                # Include the "it ran" marker so empty output isn't read as a failure.
                if turn.get("source") == "user":
                    where = f" in {turn['cwd']}" if turn.get("cwd") else ""
                    note = f"[I ran this command directly in the terminal{where}]: {turn.get('command', '')}"
                    if turn.get("output"):
                        note += f"\n{turn['output']}"
                    user_content = (user_content + "\n\n" + note) if user_content else note
                    continue

                prompt = turn["prompt"]
                command = turn.get("command", "")
                suggested = turn.get("suggested_command", "")
                output = turn.get("output", "")
                if output and len(output) > 2000:
                    output = output[:2000] + "\n... [TRUNCATED due to length]"

                if user_content:
                    user_content += f"\n\n{prompt}"
                else:
                    user_content = prompt

                self.conversation_history.append({
                    "role": "user",
                    "content": user_content
                })

                if command:
                    assistant_json = {
                        "executable": True,
                        "command": command,
                        "explanation": f"execute {command}",
                        "rule_triggered": 1,
                        "response_text": ""
                    }
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps(assistant_json)
                    })
                    user_content = f"Command execution output:\n{output}"
                elif suggested:
                    # Replay an answer turn (Rule 9): the assistant explained and SUGGESTED a
                    # command but did not run it. Surfacing the suggested_command lets a
                    # follow-up "execute that" set executable:true with this command.
                    assistant_json = {
                        "executable": False,
                        "command": "",
                        "suggested_command": suggested,
                        "explanation": "answered a how-to question; command suggested, not executed",
                        "rule_triggered": 9,
                        "response_text": output
                    }
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps(assistant_json)
                    })
                    user_content = ""
                else:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": output
                    })
                    user_content = ""

            # Cross-session context (folded into the running user message so alternation is preserved).
            for note in cross_session_notes:
                user_content = (user_content + "\n\n" + note) if user_content else note

            if user_content:
                user_content += f"\n\n{instruction}"
            else:
                user_content = instruction

            self.conversation_history.append({
                "role": "user",
                "content": user_content
            })

        executed_command = ""
        execution_output = ""
        suggested_command = ""
        clarifications_used = 0
        clar_log = []   # (question, answer) pairs, folded into the persisted prompt
        handled = False

        while not handled:
            rounds_remaining = MAX_CLARIFICATION_ROUNDS - clarifications_used

            completion_params = {
                "model": self.model_name,
                "messages": self.conversation_history,
            }
            if self.api_base:
                completion_params["api_base"] = self.api_base

            if self.tool_calling:
                # Withdraw the clarification tool on the final round to force a command.
                completion_params["tools"] = self._build_tools(rounds_remaining > 0)
                completion_params["tool_choice"] = "auto"
            elif not is_openai_model(self.model_name):
                completion_params["num_ctx"] = CTX_NUM

            _debug(f"GEN call: tool_calling={self.tool_calling}, round_remaining={rounds_remaining}, "
                   f"messages={len(self.conversation_history)}, "
                   f"tools={[t['function']['name'] for t in completion_params.get('tools', [])]}")
            _debug("GEN messages roles =", [m.get("role") for m in self.conversation_history])
            _debug("GEN last user msg  =", repr(self.conversation_history[-1].get("content")))

            response = litellm.completion(**completion_params)
            assistant_message = response.choices[0].message
            self.conversation_history.append(assistant_message.model_dump(exclude_none=True))

            _debug("GEN response.tool_calls =",
                   [(tc.function.name, tc.function.arguments) for tc in (assistant_message.tool_calls or [])])
            _debug("GEN response.content   =", repr(assistant_message.content))

            if self.tool_calling and assistant_message.tool_calls:
                _debug("BRANCH: tool_calling + tool_calls")
                clar_calls = [tc for tc in assistant_message.tool_calls
                              if tc.function.name == "ask_user_clarification"]
                if clar_calls and rounds_remaining > 0:
                    # The agent decided it is ambiguous. The clarification tool authors the
                    # question (its own LLM call) and asks the user; feed the answer back and
                    # re-prompt the generator.
                    answer = ask_user_clarification(instruction)
                    clar_log.append((instruction, answer))
                    followup = _clarification_followup(instruction, answer)
                    for tool_call in assistant_message.tool_calls:
                        body = followup if tool_call.function.name == "ask_user_clarification" \
                            else "[Deferred: answer the clarification question first.]"
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": body
                        })
                    clarifications_used += 1
                    continue

                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "execute_bash_command":
                        try:
                            args_data = json.loads(tool_call.function.arguments)
                            command_input = BashCommandInput(**args_data)
                            executed_command = command_input.command

                            print(f"[TOOL REQUESTED] Command: {command_input.command}")
                            print(f"[TOOL REQUESTED] Explanation: {command_input.explanation}")

                            execution_result = self._dispatch_command(command_input.command)
                        except json.JSONDecodeError:
                            execution_result = "[Error: Generated JSON arguments failed structure validation rules]"
                        except BashSafetyViolationError as safety_err:
                            execution_result = f"[Error: {str(safety_err)}]"
                        except Exception as e:
                            execution_result = f"[Error: Failed to process tool call arguments: {str(e)}]"

                        print(f"[RESULT] Shell Response:\n{execution_result}")
                        execution_output = execution_result

                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": execution_result
                        })
                    elif tool_call.function.name == "answer_question":
                        # Informational/how-to reply: explain and (optionally) suggest a command
                        # WITHOUT executing it. The suggestion is persisted so a later
                        # "execute it" can resolve and run it.
                        try:
                            args_data = json.loads(tool_call.function.arguments)
                            answer_input = AnswerInput(**args_data)
                            suggested_command = answer_input.suggested_command
                            print(answer_input.explanation)
                            if answer_input.suggested_command:
                                print(f"\nSuggested command (not executed): {answer_input.suggested_command}")
                            execution_output = answer_input.explanation
                            tool_result = "[Answer delivered to the user. Suggested command was NOT executed.]"
                        except Exception as e:
                            tool_result = f"[Error: Failed to process answer tool arguments: {str(e)}]"
                            print(tool_result)
                            execution_output = tool_result

                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_result
                        })
                    elif tool_call.function.name == "execute_plan":
                        # Multi-step plan: show all steps, confirm once, run in order, stop on failure.
                        try:
                            args_data = json.loads(tool_call.function.arguments)
                            steps = args_data.get("steps", [])
                            overview = args_data.get("overview", "") or ""
                            executed_command = "; ".join(
                                s.get("command", "") for s in steps if isinstance(s, dict) and s.get("command")
                            )
                            plan_result = self._run_plan(steps, overview)
                        except json.JSONDecodeError:
                            plan_result = "[Error: Generated JSON arguments failed structure validation rules]"
                        except Exception as e:
                            plan_result = f"[Error: Failed to process plan tool arguments: {str(e)}]"

                        print(f"[RESULT] Plan transcript:\n{plan_result}")
                        execution_output = plan_result

                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": plan_result
                        })
                handled = True

            elif not self.tool_calling and assistant_message.content:
                _debug("BRANCH: fallback (parse JSON content)")
                content = assistant_message.content
                try:
                    parsed = parse_json_response(content)
                    _debug("fallback parsed =", parsed)
                    command = parsed.get("command", "")
                    suggested_command = parsed.get("suggested_command", "") or ""
                    explanation = parsed.get("explanation", "")
                    response_text = parsed.get("response_text", "")
                    if not response_text:
                        response_text = parsed.get("reason", "") or parsed.get("error", "") or explanation
                    rule_triggered = parsed.get("rule_triggered")

                    # Backward compatible executable check
                    executable = parsed.get("executable", bool(command and command != "bash: command not found"))
                except Exception:
                    # Fallback to direct conversational response
                    print(content)
                    history_manager.append_history_turn(_persist_prompt(instruction, clar_log), "", content, llm_relevant_ids)
                    return

                # Rule 11 (output awareness) must ANSWER BY RUNNING the command, not merely suggest it.
                # A weak/cautious model sometimes conflates it with a Rule 9 how-to: it returns the
                # re-run as `suggested_command` with `executable: false` (as seen on "of the files I
                # listed, which are directories?"). Deterministically promote it to an executed command
                # so the question actually gets answered; it still passes the modification safety check.
                if (not executable and not command and suggested_command
                        and str(rule_triggered) == "11"):
                    _debug("RULE-11 SALVAGE: promoting suggested_command to an executed re-run")
                    command = suggested_command
                    suggested_command = ""
                    executable = True

                if parsed.get("needs_clarification") and rounds_remaining > 0:
                    # The agent flagged ambiguity. The clarification tool authors the question
                    # (its own LLM call) and asks the user; feed the answer back and re-prompt.
                    answer = ask_user_clarification(instruction)
                    clar_log.append((instruction, answer))
                    self.conversation_history.append({
                        "role": "user",
                        "content": _clarification_followup(instruction, answer)
                    })
                    clarifications_used += 1
                    continue

                # Multi-step plan (Rule 13) in fallback mode: the JSON carries a `steps` array instead
                # of a single command. Run it through the SAME plan runner the tool path uses (preview,
                # one [y/N], in-order, stop-on-failure, cd/shell-state hoisting).
                plan_steps = parsed.get("steps")
                if isinstance(plan_steps, list) and any(
                        isinstance(s, dict) and s.get("command") for s in plan_steps):
                    overview = parsed.get("overview", "") or ""
                    executed_command = "; ".join(
                        s.get("command", "") for s in plan_steps if isinstance(s, dict) and s.get("command"))
                    execution_output = self._run_plan(plan_steps, overview)
                    print(f"[RESULT] Plan transcript:\n{execution_output}")
                    self.conversation_history.append({
                        "role": "user",
                        "content": f"Plan execution output:\n{execution_output}"
                    })
                    handled = True
                elif not executable:
                    # A Rule 9 answer carries a suggested_command (not executed); persist it so a
                    # later "execute that" can resolve and run it. Rejections carry neither.
                    if suggested_command:
                        print(f"\nSuggested command (not executed): {suggested_command}")
                    if response_text:
                        print(response_text)
                        history_manager.append_history_turn(_persist_prompt(instruction, clar_log), "", response_text, llm_relevant_ids, suggested_command=suggested_command)
                    elif content:
                        print(content)
                        history_manager.append_history_turn(_persist_prompt(instruction, clar_log), "", content, llm_relevant_ids, suggested_command=suggested_command)
                    return
                else:
                    executed_command = command
                    print(f"[TEXT PARSED] Command: {command}")
                    print(f"[TEXT PARSED] Explanation: {explanation}")

                    try:
                        execution_result = self._dispatch_command(command)
                    except Exception as e:
                        execution_result = f"[Error: Fallback JSON parsing/execution failed: {str(e)}]"

                    print(f"[RESULT] Shell Response:\n{execution_result}")
                    execution_output = execution_result

                    self.conversation_history.append({
                        "role": "user",
                        "content": f"Command execution output:\n{execution_result}"
                    })
                    handled = True

            elif assistant_message.content:
                _debug("BRANCH: trailing content (tool_calling model returned TEXT, no tool call)")
                content = assistant_message.content.strip()
                try:
                    parsed = parse_json_response(content)
                    response_text = parsed.get("response_text", "")
                    if not response_text:
                        response_text = parsed.get("reason", "") or parsed.get("error", "") or parsed.get("explanation", "")
                    if response_text:
                        print(response_text)
                        execution_output = response_text
                    else:
                        print(content)
                        execution_output = content
                except Exception:
                    print(assistant_message.content)
                    execution_output = assistant_message.content
                handled = True
            else:
                # The model returned no tool call and no content (weak local models sometimes
                # emit an empty completion). Don't persist a junk "Error Unknown" turn that would
                # pollute later reference resolution - report and stop without recording it.
                _debug("BRANCH: empty model response (no tool_calls, no content)")
                print("Sorry, I couldn't produce a response for that. Please rephrase or try again.")
                return

        # Log new turn to history
        if executed_command or execution_output or suggested_command:
            history_manager.append_history_turn(
                _persist_prompt(instruction, clar_log),
                executed_command,
                execution_output,
                llm_relevant_ids,
                suggested_command=suggested_command,
            )

        # Persistent memory: store durable facts/preferences from this instruction, independent of
        # the action above (so "move to X. this is my project folder." both cd's and remembers).
        # Gated so ordinary commands skip the sub-call. The deterministic how-to/execute routes
        # return earlier, but their phrasings never match is_memory_candidate, so memory statements
        # always reach here via the main pipeline.
        if is_memory_candidate(instruction):
            self._store_memories(instruction, executed_command)
