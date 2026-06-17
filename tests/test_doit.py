import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure the 'src' directory is in the import path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from llm_communicator.llm_bash import (
    BashToolAgent,
    BashSafetyViolationError,
    execute_bash,
    parse_json_response,
)

# Helpers to mock Chat Completion responses
class MockFunction:
    def __init__(self, name, arguments_dict):
        self.name = name
        self.arguments = json.dumps(arguments_dict)

class MockToolCall:
    def __init__(self, tool_id, name, arguments_dict):
        self.id = tool_id
        self.type = "function"
        self.function = MockFunction(name, arguments_dict)

    def model_dump(self, *args, **kwargs):
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments
            }
        }

class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, *args, **kwargs):
        dump = {"role": "assistant"}
        if self.content is not None:
            dump["content"] = self.content
        if self.tool_calls is not None:
            dump["tool_calls"] = [t.model_dump() for t in self.tool_calls]
        return dump

class MockChoice:
    def __init__(self, message):
        self.message = message

class MockResponse:
    def __init__(self, message):
        self.choices = [MockChoice(message)]


# =====================================================================
# SECTION 1: Config Loader Tests
# =====================================================================

def test_config_loader_file_not_exist(tmp_path):
    """Verify default fallback values when doit.cfg does not exist."""
    config_file = tmp_path / "non_existent_doit.cfg"
    from doit_module.config_loader import load_config
    model, api_base, tool_calling = load_config(config_file)
    
    assert model == "gpt-5.4-nano"
    assert api_base is None
    assert tool_calling is True


def test_config_loader_file_exists(tmp_path):
    """Verify parsed values when doit.cfg is present."""
    config_file = tmp_path / "doit.cfg"
    config_file.write_text("""[model]
name = ollama/gemma3:4b
api_base = http://localhost:11434
tool_calling = false
""")
    
    from doit_module.config_loader import load_config
    model, api_base, tool_calling = load_config(config_file)
    
    assert model == "ollama/gemma3:4b"
    assert api_base == "http://localhost:11434"
    assert tool_calling is False


# =====================================================================
# SECTION 2: JSON Response Parsing Tests
# =====================================================================

def test_parse_json_response_clean():
    """Verify parsing a raw clean JSON string."""
    json_str = '{"command": "ls", "explanation": "list files"}'
    parsed = parse_json_response(json_str)
    assert parsed["command"] == "ls"
    assert parsed["explanation"] == "list files"


def test_parse_json_response_with_markdown_wrapper():
    """Verify parsing a JSON string wrapped in markdown code blocks."""
    json_str = """
```json
{
  "command": "mkdir new_dir",
  "explanation": "create new directory"
}
```
"""
    parsed = parse_json_response(json_str)
    assert parsed["command"] == "mkdir new_dir"
    assert parsed["explanation"] == "create new directory"


def test_parse_json_response_with_conversational_noise():
    """Verify parsing a JSON string embedded in conversational text."""
    json_str = """
Sure! Here is the command you need:
{
  "command": "rm file.txt",
  "explanation": "delete file"
}
Let me know if you need anything else!
"""
    parsed = parse_json_response(json_str)
    assert parsed["command"] == "rm file.txt"
    assert parsed["explanation"] == "delete file"


# =====================================================================
# SECTION 3: Safety Filter / Dangerous Banned Commands Tests
# =====================================================================

@pytest.mark.parametrize("banned_command", [
    "rm -rf /",
    "rm -Rf /some/directory",
    "chmod 777 script.sh",
    "chmod a+rwx 777",
    "killall nginx",
    "shutdown -h now",
    "reboot",
    "dd if=/dev/zero of=/dev/sda",
    "a:(){ : & : };\u202f:",
])
def test_dangerous_commands_violate_safety(banned_command):
    with pytest.raises(BashSafetyViolationError) as exc_info:
        execute_bash(banned_command)
    assert "Security Block: Command contains banned structural pattern" in str(exc_info.value)


def test_safe_command_execution():
    output = execute_bash("echo 'safety check'", verbose=False)
    assert "safety check" in output


# =====================================================================
# SECTION 4: File System Modification Classifier (_filter_bash) Tests
# =====================================================================

@patch("llm_communicator.llm_bash.litellm.completion")
def test_filter_bash_modifies_file_system(mock_completion):
    mock_message = MockMessage(content="TRUE: Creates a directory")
    mock_completion.return_value = MockResponse(mock_message)

    agent = BashToolAgent(api_key="fake-key")
    modifies, explanation = agent._filter_bash("mkdir test_dir")

    assert modifies is True
    assert explanation == "Creates a directory"
    mock_completion.assert_called_once()


@patch("llm_communicator.llm_bash.litellm.completion")
def test_filter_bash_does_not_modify_file_system(mock_completion):
    mock_message = MockMessage(content="FALSE")
    mock_completion.return_value = MockResponse(mock_message)

    agent = BashToolAgent(api_key="fake-key")
    modifies, explanation = agent._filter_bash("pwd")

    assert modifies is False
    assert explanation == ""


# =====================================================================
# SECTION 5: Agent Flow Tests (Tool Calling Enabled)
# =====================================================================

@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_run_single_tool_calling_no_modification(mock_execute_bash, mock_completion):
    # Mock generation (tool call) & filter (FALSE)
    tool_call = MockToolCall("call_1", "execute_bash_command", {"command": "ls -la", "explanation": "list files"})
    msg_gen = MockMessage(tool_calls=[tool_call])
    msg_filter = MockMessage(content="FALSE")
    
    mock_completion.side_effect = [
        MockResponse(msg_gen),
        MockResponse(msg_filter)
    ]

    mock_execute_bash.return_value = "file1\nfile2"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    
    with patch("builtins.input") as mock_input:
        agent.run_single("Show me my files")
        mock_input.assert_not_called()

    mock_execute_bash.assert_called_once_with("ls -la")


# =====================================================================
# SECTION 6: Agent Flow Tests (Fallback Text Parsing Mode)
# =====================================================================

@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
@patch("builtins.input")
def test_run_single_non_tool_calling_approved(mock_input, mock_execute_bash, mock_completion, mock_load_config):
    # Configure agent to be non-tool-calling
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    
    # Mock text output containing JSON command
    msg_gen = MockMessage(content='{"command": "mkdir -p new_folder", "explanation": "create folder"}')
    msg_filter = MockMessage(content="TRUE: Creates a new folder")
    
    mock_completion.side_effect = [
        MockResponse(msg_gen),
        MockResponse(msg_filter)
    ]
    
    mock_input.return_value = "y"
    mock_execute_bash.return_value = "[Success]"
    
    agent = BashToolAgent()
    assert agent.tool_calling is False
    
    agent.run_single("Make a folder")
    
    # User was prompted
    mock_input.assert_called_once()
    assert "Do you want to continue?" in mock_input.call_args[0][0]
    
    # Command was executed
    mock_execute_bash.assert_called_once_with("mkdir -p new_folder")


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
@patch("builtins.input")
def test_run_single_non_tool_calling_declined(mock_input, mock_execute_bash, mock_completion, mock_load_config):
    # Configure agent to be non-tool-calling
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    
    # Mock text output containing JSON command
    msg_gen = MockMessage(content='{"command": "rm -rf old_folder", "explanation": "delete folder"}')
    msg_filter = MockMessage(content="TRUE: Deletes a folder")
    
    mock_completion.side_effect = [
        MockResponse(msg_gen),
        MockResponse(msg_filter)
    ]
    
    mock_input.return_value = "n"
    
    agent = BashToolAgent()
    agent.run_single("Remove folder")
    
    mock_input.assert_called_once()
    mock_execute_bash.assert_not_called()
    assert agent.conversation_history[-1]["content"] == "Command execution output:\n[Cancelled: User declined to execute command that modifies the file system]"
