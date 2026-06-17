import sys
from pathlib import Path

# Add the 'src' directory to sys.path to allow sibling module imports
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fixtures import OPENAI_API_KEY, MODEL_NAME, DOIT_SYSTEM_PROMPT, DOIT_FILTER_PROMPT
from doit_module.config_loader import load_config

import os
import re
import json
import shutil
import functools
import subprocess
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import litellm

# To protect the host system, we configure a basic blacklist of dangerous commands.
BANNED_COMMAND_PATTERNS = [
    r"\brm\s+-[rfRF]+.*\/",           # Root/dangerous recursive deletion
    r"\bchmod\b.*777",                 # Dangerous wildcard permissions
    r"\bkillall\b",                    # Indiscriminate process termination
    r"\bshutdown\b",                   # Host shutdown command
    r"\breboot\b",                     # Host reboot command
    r"\bdd\s+if=/dev/zero",            # Zeroing out drives
    r"\b:\(\)\{\s*:\s*&\s*:\s*\}; \s*:", # Classic fork bomb detection
]

FALLBACK_SYSTEM_INSTRUCTION = """IMPORTANT: You do not support native tool calling in this environment. Instead, you MUST respond with a raw JSON block containing exactly the following keys:
- "executable": (boolean) true if a bash command is generated to execute, false if not.
- "command": (string) the single-line or multi-line bash script to execute (only when executable is true, otherwise empty "").
- "explanation": (string) a short explanation of what the command does, or why it is not executable.
- "rule_triggered": (integer) the number of the system instruction rule triggered (1 for command generation, 2 for impossible, 3 for safety violation, 4 for irrelevant input, 5 for exit, 6 for capability inquiry, 7 for assume file existence).
- "response_text": (string) the conversational text, warning, or error message to display directly to the user (required when executable is false, e.g., "bash: command not found" for rule 2, "The command is not safe to execute. <reason>" for rule 3, or "Exiting..." for rule 5).

Do not include any other conversational text, pleasantries, markdown formatting (outside of the JSON block itself), or preamble.

Example response for successful command:
{
  "executable": true,
  "command": "ls -la",
  "explanation": "list all files in the current directory",
  "rule_triggered": 1,
  "response_text": ""
}

Example response for impossible command (Rule 2):
{
  "executable": false,
  "command": "",
  "explanation": "Cannot perform this task as a bash command because <reason>",
  "rule_triggered": 2,
  "response_text": "bash: command not found"
}

Example response for safety violation (Rule 3):
{
  "executable": false,
  "command": "",
  "explanation": "Command attempts to recursively delete system root",
  "rule_triggered": 3,
  "response_text": "The command is not safe to execute. This would destroy your system directory."
}

Example response for irrelevant input (Rule 4):
{
  "executable": false,
  "command": "",
  "explanation": "The input is not related to executing bash commands",
  "rule_triggered": 4,
  "response_text": "My sole purpose is to execute bash commands. Your message is irrelevant."
}

Example response for exit command (Rule 5):
{
  "executable": false,
  "command": "",
  "explanation": "The user wants to exit the session",
  "rule_triggered": 5,
  "response_text": "Exiting..."
}

Example response for capability inquiry (Rule 6):
{
  "executable": false,
  "command": "",
  "explanation": "The user is asking about my capabilities",
  "rule_triggered": 6,
  "response_text": "My purpose is to translate natural language descriptions into executable bash commands."
}

Example response for assuming file existence (Rule 7):
{
  "executable": true,
  "command": "cat /etc/passwd",
  "explanation": "Attempting to read the /etc/passwd file",
  "rule_triggered": 7,
  "response_text": ""
}
"""

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


def parse_json_response(content: str) -> Dict[str, Any]:
    """
    Robust JSON parser that extracts JSON object {...} even if wrapped in markdown code blocks
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


# Defining the tools list using OpenAI/LiteLLM's standard format.
tools_definition: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash_command",
            "description": (
                "Executes a localized bash command on the host terminal environment. "
                "Use this tool ONLY to execute the user's requested bash command. "
                "DO NOT call this tool for general knowledge, questions, or irrelevant inputs "
                "(e.g., do not generate 'echo' commands to answer questions)."
            ),
            "parameters": BashCommandInput.model_json_schema()
        }
    }
]


class BashToolAgent:
    """
    State manager for our autonomous transformer execution loop.
    Maintains system memory prompts and guides LiteLLM tool/fallback interactions.
    """
    def __init__(self, api_key: Optional[str] = None):
        # Load configuration
        self.model_name, self.api_base, self.tool_calling = load_config()

        # Set API key for LiteLLM if provided/found
        key = api_key or OPENAI_API_KEY
        if key:
            # LiteLLM looks at OPENAI_API_KEY for openai models
            os.environ["OPENAI_API_KEY"] = key

        system_prompt = DOIT_SYSTEM_PROMPT
        if not self.tool_calling:
            system_prompt += "\n\n" + FALLBACK_SYSTEM_INSTRUCTION

        self.conversation_history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

    def _filter_bash(self, command: str) -> tuple[bool, str]:
        """Filter for bash commands using LLM as a judge to determine if a command will modify file system."""
        response = litellm.completion(
            model=self.model_name,
            api_base=self.api_base,
            messages=[
                {
                    "role": "system",
                    "content": DOIT_FILTER_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Does the following command modify the file system? {command}"
                }
            ]
        )
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

    def run_agent_loop(self, max_iterations: int = 5) -> str:
        """
        Coordinates the agent run cycle. Iterates until standard completions are hit.
        """
        for step in range(max_iterations):
            self.conversation_history.append({"role": "user", "content": input("Describe your command: ")})
            
            print(f"\n--- [AGENT ITERATION STEP {step + 1}/{max_iterations}] ---")

            completion_params = {
                "model": self.model_name,
                "messages": self.conversation_history,
            }
            if self.api_base:
                completion_params["api_base"] = self.api_base

            if self.tool_calling:
                completion_params["tools"] = tools_definition
                completion_params["tool_choice"] = "auto"

            # Call LiteLLM
            response = litellm.completion(**completion_params)
            assistant_message = response.choices[0].message
            
            # Record assistant output state in the history
            self.conversation_history.append(assistant_message.model_dump(exclude_none=True))

            if self.tool_calling and assistant_message.tool_calls:
                 for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "execute_bash_command":
                        try:
                            args_data = json.loads(tool_call.function.arguments)
                            command_input = BashCommandInput(**args_data)
                            
                            print(f"[TOOL REQUESTED] Command: {command_input.command}")
                            print(f"[TOOL REQUESTED] Explanation: {command_input.explanation}")
                            
                            modifies, filter_explanation = self._filter_bash(command_input.command)
                            should_execute = True
                            if modifies:
                                print(f"This command will modify your file system: {filter_explanation}")
                                user_choice = input("Do you want to continue? [y/n]: ").strip().lower()
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

                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": execution_result
                    })

            elif not self.tool_calling and assistant_message.content:
                # Fallback non-tool-calling parser
                content = assistant_message.content
                if content == "Exiting...":
                    print("Exiting...")
                    break
                
                try:
                    parsed = parse_json_response(content)
                    command = parsed.get("command", "")
                    explanation = parsed.get("explanation", "")
                    response_text = parsed.get("response_text", "")
                    if not response_text:
                        response_text = parsed.get("reason", "") or explanation
                    rule_triggered = parsed.get("rule_triggered")
                    
                    # Backward compatible executable check
                    executable = parsed.get("executable", bool(command and command != "bash: command not found"))
                except Exception:
                    # Fallback to direct conversational response
                    print(content)
                    continue

                if not executable:
                    # Display the response_text to the user
                    if response_text:
                        print(response_text)
                    elif content:
                        print(content)
                    
                    if rule_triggered == 5 or response_text == "Exiting...":
                        break
                    continue

                print(f"[TEXT PARSED] Command: {command}")
                print(f"[TEXT PARSED] Explanation: {explanation}")
                
                try:
                    modifies, filter_explanation = self._filter_bash(command)
                    should_execute = True
                    if modifies:
                        print(f"This command will modify your file system: {filter_explanation}")
                        user_choice = input("Do you want to continue? [y/n]: ").strip().lower()
                        if user_choice not in ('y', 'yes'):
                            should_execute = False
                            execution_result = "[Cancelled: User declined to execute command that modifies the file system]"
                            
                    if should_execute:
                        execution_result = execute_bash(command)
                except Exception as e:
                    execution_result = f"[Error: Fallback JSON parsing/execution failed: {str(e)}]"
                    
                print(f"[RESULT] Shell Response:\n{execution_result}")
                
                # Append execution result as a user message since non-tool models don't support tool roles
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Command execution output:\n{execution_result}"
                })

            elif assistant_message.content:
                content = assistant_message.content.strip()
                if content == "Exiting...":
                    break
                try:
                    parsed = parse_json_response(content)
                    response_text = parsed.get("response_text", "")
                    if not response_text:
                        response_text = parsed.get("reason", "") or parsed.get("explanation", "")
                    if response_text:
                        print(response_text)
                    else:
                        print(content)
                except Exception:
                    print(assistant_message.content)
                    if assistant_message.content == "Exiting...":
                        break
                    else:
                        self.conversation_history.append({"role": "assistant", "content": assistant_message.content})
            else:
                print("Error Unknown")

        return "Agent execution terminated: reached maximum reasoning iterations boundary limit."

    def run_single(self, instruction: str) -> None:
        """
        Coordinates a single user query execution.
        """
        self.conversation_history.append({"role": "user", "content": instruction})

        completion_params = {
            "model": self.model_name,
            "messages": self.conversation_history,
        }
        if self.api_base:
            completion_params["api_base"] = self.api_base

        if self.tool_calling:
            completion_params["tools"] = tools_definition
            completion_params["tool_choice"] = "auto"

        response = litellm.completion(**completion_params)
        assistant_message = response.choices[0].message
        self.conversation_history.append(assistant_message.model_dump(exclude_none=True))

        if self.tool_calling and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "execute_bash_command":
                    try:
                        args_data = json.loads(tool_call.function.arguments)
                        command_input = BashCommandInput(**args_data)

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

                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": execution_result
                    })
                    
        elif not self.tool_calling and assistant_message.content:
            content = assistant_message.content
            if content == "Exiting...":
                print("Exiting...")
                return
                
            try:
                parsed = parse_json_response(content)
                command = parsed.get("command", "")
                explanation = parsed.get("explanation", "")
                response_text = parsed.get("response_text", "")
                if not response_text:
                    response_text = parsed.get("reason", "") or explanation
                rule_triggered = parsed.get("rule_triggered")
                
                # Backward compatible executable check
                executable = parsed.get("executable", bool(command and command != "bash: command not found"))
            except Exception:
                # Fallback to direct conversational response
                print(content)
                return

            if not executable:
                if response_text:
                    print(response_text)
                elif content:
                    print(content)
                return

            
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
            
            self.conversation_history.append({
                "role": "user",
                "content": f"Command execution output:\n{execution_result}"
            })
            
        elif assistant_message.content:
            content = assistant_message.content.strip()
            try:
                parsed = parse_json_response(content)
                response_text = parsed.get("response_text", "")
                if not response_text:
                    response_text = parsed.get("reason", "") or parsed.get("explanation", "")
                if response_text:
                    print(response_text)
                else:
                    print(content)
            except Exception:
                print(assistant_message.content)
        else:
            print("Error Unknown")
