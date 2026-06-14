import sys
from pathlib import Path

# Add the 'src' directory to sys.path to allow sibling module imports
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fixtures import OPENAI_API_KEY, MODEL_NAME, DOIT_SYSTEM_PROMPT

import re
import json
import subprocess
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessage
except ImportError:
    print(
        "Missing dependencies! Please ensure you run this within a uv managed environment.\n"
        "Run the following to initialize and install dependencies:\n"
        "  uv add openai pydantic\n",
        file=sys.stderr
    )
    sys.exit(1)

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

def execute_bash(command: str, verbose: bool = True) -> str:
    """
    Executes a raw bash string in an isolated subprocess under strict constraints.
    Returns stdout/stderr merged result as a single string.
    """
    # Defensive Pre-execution Check against Injection/Destructive Patterns
    for pattern in BANNED_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            raise BashSafetyViolationError(
                f"Security Block: Command contains banned structural pattern matching '{pattern}'."
            )

    if verbose:
        print(f"\n[EXEC] Running Command:\n{command}\n")

    try:
        # We explicitly invoke bash as the shell shell binary rather than relying on default sh
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=15.0  # Safe execution timeout boundary to prevent infinite processes
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
        if result.returncode:
            output += f"--- RETURN CODE ---\n{result.returncode}\n"
            

        if not output:
            output = "[Success: Command executed with no returning output channels]"

        return output

    except subprocess.TimeoutExpired:
        return "[Error: Command Execution Terminated due to exceeding 15.0s Timeout Limit]"
    except Exception as e:
        return f"[Error occurred during system execution execution loop: {str(e)}]"

# Defining the tools list using OpenAI's standard format.
tools_definition: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash_command",
            "description": (
                "Executes a localized bash command on the host terminal environment. "
                "Use this tool to execute the user's requested command if it is possible."
            ),
            "parameters": BashCommandInput.model_json_schema()
        }
    }
]

class BashToolAgent:
    """
    State manager for our autonomous transformer execution loop.
    Maintains system memory prompts and guides OpenAI tool interactions.
    """
    def __init__(self, api_key: Optional[str] = None):
        # Fallback sequence initialization using import from fixtures
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.conversation_history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": DOIT_SYSTEM_PROMPT
            }
        ]

    def run_agent_loop(self, max_iterations: int = 5) -> str:
        """
        Coordinates the agent run cycle. Iterates until standard completions are hit
        or maximum step boundaries are crossed.
        """

        for step in range(max_iterations):
            self.conversation_history.append({"role": "user", "content": input("Describe your command: ")})
            
            print(f"\n--- [AGENT ITERATION STEP {step + 1}/{max_iterations}] ---")

            # Call the LLM with our available tool schemas
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.conversation_history,
                tools=tools_definition,
                tool_choice="auto"
            )

            assistant_message: ChatCompletionMessage = response.choices[0].message
            
            # Record assistant output state in the history
            self.conversation_history.append(assistant_message.model_dump(exclude_none=True))

            

            if assistant_message.tool_calls:
                 for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "execute_bash_command":
                        try:
                            # Extract and parse model generated arguments
                            args_data = json.loads(tool_call.function.arguments)
                            command_input = BashCommandInput(**args_data)
                            
                            print(f"[TOOL REQUESTED] Explanation: {command_input.explanation}")
                            
                            # Run command inside sandboxed runtime environment
                            execution_result = execute_bash(command_input.command)
                        except json.JSONDecodeError:
                            execution_result = "[Error: Generated JSON arguments failed structure validation rules]"
                        except BashSafetyViolationError as safety_err:
                            execution_result = f"[Error: {str(safety_err)}]"
                        except Exception as e:
                            execution_result = f"[Error: Failed to process tool call arguments: {str(e)}]"

                    print(f"[RESULT] Shell Response:\n{execution_result}")

                    # Feedback execution output directly back into LLM context window
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": execution_result
                    })

            elif assistant_message.content:
                print(assistant_message.content)
                if assistant_message.content == "Exiting...":
                    break
                else:
                    self.conversation_history.append({"role": "assistant", "content": assistant_message.content})

            
            else: #unknown error has occured 
                print("Error Unknown ")
                


        return "Agent execution terminated: reached maximum reasoning iterations boundary limit."

    def run_single(self, instruction: str) -> None:
        """
        Coordinates a single user query execution. Translates instruction to a shell command,
        prints it, executes it, and prints the output.
        """
        self.conversation_history.append({"role": "user", "content": instruction})

        # Call the LLM with our available tool schemas
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.conversation_history,
            tools=tools_definition,
            tool_choice="auto"
        )

        assistant_message: ChatCompletionMessage = response.choices[0].message
        self.conversation_history.append(assistant_message.model_dump(exclude_none=True))

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "execute_bash_command":
                    # Extract and parse model generated arguments
                    args_data = json.loads(tool_call.function.arguments)
                    command_input = BashCommandInput(**args_data)

                    # Print the command
                    print(command_input.command)

                    # Execute command
                    execution_result = execute_bash(command_input.command, verbose=False)

                    # Print the execution result to the screen
                    if execution_result:
                        print(execution_result, end="")
        elif assistant_message.content:
            print(assistant_message.content)
        else:
            print("Error Unknown")


if __name__ == "__main__":
    pass


    