import sys
from pathlib import Path

# Add the 'src' directory to sys.path to allow sibling module imports
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fixtures import OPENAI_API_KEY, MODEL_NAME, DOIT_SYSTEM_PROMPT, DOIT_FILTER_PROMPT, LLM_CONTEXT_LIMIT, CTX_NUM, MAX_CLARIFICATION_ROUNDS
from doit_module.config_loader import load_config
import llm_communicator.history_manager as history_manager
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
    resolve_session_state_hoist,
    parse_json_response,
    is_openai_model,
    NO_ANSWER_SENTINEL,
    EXECUTE_TOOL_DEF,
    ANSWER_TOOL_DEF,
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
    "3. SAFETY CHECK: If you can match two different previous commands that are not connected, choose the most recent one (the command with the larger ID).\n"
    "Note: The recent command history is presented below from most recent to oldest."
)



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
        # (suggested_command). Pure conversational/rejection turns carry neither and are
        # not linkable. Answer turns must remain so "execute it"/"modify it" can refer back.
        history_metadata = [t for t in history_metadata if t.get("command") or t.get("suggested_command")]
        if not history_metadata:
            return []

        # Quick heuristic check for independent instructions to assist small models
        instruction_lower = instruction.lower()
        context_indicators = [
            "it", "them", "that", "those", "this", "these", "like", "mean", "meant", "the command",
            "the output", "the results", "re-run", "recursively", "again", "previous", "we just", "before",
            "output", "results", "we listed", "we created", "we did", "we ran", "we made", "above", "how many",
            "execute", "run it", "run that", "modify", "do it"
        ]
        has_context_indicator = False
        for indicator in context_indicators:
            if indicator in ("it", "them", "that", "those", "this", "these", "like", "mean", "meant", "again", "before", "previous", "above"):
                if re.search(r'\b' + re.escape(indicator) + r'\b', instruction_lower):
                    has_context_indicator = True
                    break
            else:
                if indicator in instruction_lower:
                    has_context_indicator = True
                    break

        if not has_context_indicator:
            return []

        formatted_history = "\n".join([
            f"- [ID: {t['id']}] Prompt: \"{t['prompt']}\" | Command: \"{t['command']}\""
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

    def _dispatch_command(self, command: str) -> str:
        """
        Run a generated command. A plain `cd` is value-hoisted to the parent shell (see _hoist_cd);
        a single session-state builtin (export/alias/set/unset/shopt/pushd/popd) is command-hoisted
        (see _hoist_shell); everything else goes through the sandboxed subprocess with the safety
        filter and confirmation (see _execute_with_confirmation).
        """
        cd_target = resolve_cd_hoist(command)
        if cd_target is not None:
            _debug("CD hoist:", cd_target)
            return self._hoist_cd(cd_target)
        shell_cmd = resolve_session_state_hoist(command)
        if shell_cmd is not None:
            _debug("SHELL-STATE hoist:", shell_cmd)
            return self._hoist_shell(shell_cmd)
        return self._execute_with_confirmation(command)

    def _execute_with_confirmation(self, command: str) -> str:
        """
        Run a command through the two safety layers: the LLM filesystem-modification judge
        (asks the user for [y/N] before a modifying command) and the regex blacklist inside
        execute_bash. Returns the execution output (or a cancelled/error marker). Shared by the
        deterministic "execute that" route so suggestions get the same safety treatment as
        model-generated commands.
        """
        try:
            modifies, filter_explanation = self._filter_bash(command)
            if modifies:
                print(f"This command will modify your file system: {filter_explanation}")
                user_choice = input("Do you want to continue? [y/N]: ").strip().lower()
                if user_choice not in ('y', 'yes'):
                    return "[Cancelled: User declined to execute command that modifies the file system]"
            return execute_bash(command)
        except BashSafetyViolationError as safety_err:
            return f"[Error: {str(safety_err)}]"
        except Exception as e:
            return f"[Error: {str(e)}]"

    def _build_tools(self, include_clarification: bool) -> List[Dict[str, Any]]:
        """
        Tools offered to the generator. The clarification tool is withdrawn on the final round
        so the model must commit to a command (or an answer) instead of asking again.
        """
        if include_clarification:
            return [EXECUTE_TOOL_DEF, ANSWER_TOOL_DEF, CLARIFY_TOOL_DEF]
        return [EXECUTE_TOOL_DEF, ANSWER_TOOL_DEF]

    def run_single(self, instruction: str) -> None:
        """
        Coordinates a single user query execution.
        """
        # 0. Deterministic how-to route (fallback mode only). Weak non-tool-calling models cannot
        # reliably CLASSIFY a how-to question (they mis-route it to clarification), but they can
        # ANSWER one. So we detect the how-to phrasing in Python and hand it to a focused, single
        # purpose sub-call - no multi-rule prompt, no clarification machinery to copy. The answer
        # is persisted with its suggested_command so a later "execute that" can run it. Native
        # tool-calling models route this correctly via Rule 9, so they keep that path.
        if not self.tool_calling and is_howto_question(instruction):
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

        # 1. Fetch metadata for last 20 actions from history
        metadata = history_manager.get_history_metadata(limit=LLM_CONTEXT_LIMIT)

        # 2. Query LLM to identify relevant previous turns
        llm_relevant_ids = self._analyze_references(instruction, metadata)
        relevant_ids = self._resolve_transitive_dependencies(llm_relevant_ids)

        # 3. Retrieve full records of identified relevant turns
        relevant_turns = history_manager.get_full_turns(relevant_ids)

        # 4. Reconstruct conversation history starting with system prompt
        self.conversation_history = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
        # 5. Populate history with relevant turns formatted correctly
        if self.tool_calling:
            # Seed with in-context few-shot demonstrations of the clarification decision
            # (ambiguous -> ask_user_clarification; clear -> execute_bash_command).
            self.conversation_history.extend(FEWSHOT_TOOLCALL)

            for turn in relevant_turns:
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

                if not executable:
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
