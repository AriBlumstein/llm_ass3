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
    execute_bash,
    ask_user_clarification,
    parse_json_response,
    is_openai_model,
    NO_ANSWER_SENTINEL,
    EXECUTE_TOOL_DEF,
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
    "2. If it refers to previous outputs/files/results, resolve references in chronological order, preferring the most recent match based on semantic and logical dependencies.\n"
    "   - You MUST link only to the actual command turn that successfully executed the action (e.g., touch/mkdir/ls).\n"
    "   - DO NOT link to empty commands (\"command\": \"\"), failed/cancelled turns, or warning rejections.\n"
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

        # Filter out turns that did not execute any command (empty command)
        history_metadata = [t for t in history_metadata if t.get("command")]
        if not history_metadata:
            return []

        # Quick heuristic check for independent instructions to assist small models
        instruction_lower = instruction.lower()
        context_indicators = [
            "it", "them", "that", "those", "this", "these", "like", "mean", "meant", "the command", 
            "the output", "the results", "re-run", "recursively", "again", "previous", "we just", "before",
            "output", "results", "we listed", "we created", "we did", "we ran", "we made", "above", "how many"
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

    def _build_tools(self, include_clarification: bool) -> List[Dict[str, Any]]:
        """
        Tools offered to the generator. The clarification tool is withdrawn on the final round
        so the model must commit to a command instead of asking again.
        """
        if include_clarification:
            return [EXECUTE_TOOL_DEF, CLARIFY_TOOL_DEF]
        return [EXECUTE_TOOL_DEF]

    def run_single(self, instruction: str) -> None:
        """
        Coordinates a single user query execution.
        """
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

                            modifies, filter_explanation = self._filter_bash(command_input.command)
                            should_execute = True
                            if modifies:
                                print(f"This command will modify your file system: {filter_explanation}")
                                user_choice = input("Do you want to continue? [y/N]: ").strip().lower()
                                if user_choice not in ('y', 'yes'):
                                    should_execute = False
                                    execution_result = "[Cancelled: User declined to execute command that modifies the file system]"

                            if should_execute:
                                execution_result = execute_bash(command_input.command)
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
                handled = True

            elif not self.tool_calling and assistant_message.content:
                _debug("BRANCH: fallback (parse JSON content)")
                content = assistant_message.content
                try:
                    parsed = parse_json_response(content)
                    _debug("fallback parsed =", parsed)
                    command = parsed.get("command", "")
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
                    if response_text:
                        print(response_text)
                        history_manager.append_history_turn(_persist_prompt(instruction, clar_log), "", response_text, llm_relevant_ids)
                    elif content:
                        print(content)
                        history_manager.append_history_turn(_persist_prompt(instruction, clar_log), "", content, llm_relevant_ids)
                    return

                executed_command = command
                print(f"[TEXT PARSED] Command: {command}")
                print(f"[TEXT PARSED] Explanation: {explanation}")

                try:
                    modifies, filter_explanation = self._filter_bash(command)
                    should_execute = True
                    if modifies:
                        print(f"This command will modify your file system: {filter_explanation}")
                        user_choice = input("Do you want to continue? [y/N]: ").strip().lower()
                        if user_choice not in ('y', 'yes'):
                            should_execute = False
                            execution_result = "[Cancelled: User declined to execute command that modifies the file system]"

                    if should_execute:
                        execution_result = execute_bash(command)
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
                print("Error Unknown")
                execution_output = "Error Unknown"
                handled = True

        # Log new turn to history
        if executed_command or execution_output:
            history_manager.append_history_turn(_persist_prompt(instruction, clar_log), executed_command, execution_output, llm_relevant_ids)
