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


@pytest.fixture(autouse=True)
def isolate_history(tmp_path, monkeypatch):
    """Automatically isolate all tests from the host's actual history file."""
    from llm_communicator import history_manager
    test_file = tmp_path / "test_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)


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


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("builtins.print")
def test_run_single_non_tool_calling_reason_fallback(mock_print, mock_completion, mock_load_config):
    # Configure agent to be non-tool-calling
    mock_load_config.return_value = ("ollama/qwen3:4b-instruct", None, False)
    
    # Mock text output containing JSON with 'reason' instead of 'response_text'
    msg_gen = MockMessage(content='{"executable": false, "reason": "General knowledge question"}')
    
    mock_completion.side_effect = [
        MockResponse(msg_gen)
    ]
    
    agent = BashToolAgent()
    agent.run_single("can pigs fly")
    
    # Check that the reason was printed
    mock_print.assert_any_call("General knowledge question")


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("builtins.print")
def test_run_single_tool_calling_reason_fallback_in_content(mock_print, mock_completion, mock_load_config):
    # Configure agent to be tool-calling
    mock_load_config.return_value = ("ollama/qwen3:4b-instruct", None, True)
    
    # Mock text output containing JSON in assistant_message.content (no tool calls)
    msg_gen = MockMessage(content='{"executable": false, "reason": "Pigs do not fly"}')
    
    mock_completion.side_effect = [
        MockResponse(msg_gen)
    ]
    
    agent = BashToolAgent()
    agent.run_single("can pigs fly")
    
    # Check that the reason was printed
    mock_print.assert_any_call("Pigs do not fly")


def test_system_prompt_contains_anti_echo_warning():
    """Verify that DOIT_SYSTEM_PROMPT contains specific instructions forbidding echo/printf workarounds, and pipelining instructions."""
    from fixtures import DOIT_SYSTEM_PROMPT
    assert "echo" in DOIT_SYSTEM_PROMPT.lower()
    assert "printf" in DOIT_SYSTEM_PROMPT.lower()
    assert "irrelevant" in DOIT_SYSTEM_PROMPT.lower()
    assert "native tool calling" in DOIT_SYSTEM_PROMPT.lower()
    assert "heredoc" in DOIT_SYSTEM_PROMPT.lower()
    assert "cancelled" in DOIT_SYSTEM_PROMPT.lower()
    assert "piping" in DOIT_SYSTEM_PROMPT.lower()



# =====================================================================
# SECTION 7: Selective History & Multi-turn Tests
# =====================================================================

def test_history_manager_basic_operations(tmp_path, monkeypatch):
    """Verify that history_manager methods read, write, and clear history correctly."""
    from llm_communicator import history_manager
    
    # Mock history file path to be in our temp directory
    test_file = tmp_path / "test_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)
    
    # Check initial states
    assert history_manager.get_history_metadata() == []
    assert history_manager.get_full_turns([1]) == []
    
    # Append turns
    history_manager.append_history_turn("list files", "ls", "file1\nfile2")
    history_manager.append_history_turn("show process", "ps", "pid 123")
    
    # Verify metadata (outputs omitted)
    metadata = history_manager.get_history_metadata()
    assert len(metadata) == 2
    assert metadata[0] == {"id": 1, "prompt": "list files", "command": "ls"}
    assert metadata[1] == {"id": 2, "prompt": "show process", "command": "ps"}
    
    # Verify full turns retrieval
    full_turns = history_manager.get_full_turns([1])
    assert len(full_turns) == 1
    assert full_turns[0]["id"] == 1
    assert full_turns[0]["prompt"] == "list files"
    assert full_turns[0]["command"] == "ls"
    assert full_turns[0]["output"] == "file1\nfile2"
    
    # Clear history
    history_manager.clear_history()
    assert not test_file.exists()
    assert history_manager.get_history_metadata() == []


@patch("llm_communicator.llm_bash.litellm.completion")
def test_analyze_references_with_llm(mock_completion):
    """Verify that analyze_references queries LLM and correctly parses relevant IDs."""
    agent = BashToolAgent(api_key="fake-key")
    
    # Mock LLM response returning JSON
    mock_message = MockMessage(content='{"relevant_ids": [1, 2]}')
    mock_completion.return_value = MockResponse(mock_message)
    
    metadata = [
        {"id": 1, "prompt": "list files", "command": "ls"},
        {"id": 2, "prompt": "show process", "command": "ps"}
    ]
    
    relevant = agent._analyze_references("sort them", metadata)
    assert relevant == [1, 2]
    
    # Mock LLM returning empty or invalid
    mock_message_empty = MockMessage(content='{"relevant_ids": []}')
    mock_completion.return_value = MockResponse(mock_message_empty)
    assert agent._analyze_references("hello", metadata) == []


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_agent_runs_with_selective_history(mock_execute_bash, mock_completion, tmp_path, monkeypatch):
    """Verify agent correctly selective-retrieves and formats history messages in main prompt."""
    from llm_communicator import history_manager
    
    test_file = tmp_path / "test_agent_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)
    
    # 1. Setup existing history: 2 turns
    history_manager.append_history_turn("list files", "ls", "file1\nfile2")
    history_manager.append_history_turn("make dir", "mkdir src", "")
    
    # 2. Setup mock LLM behavior:
    # First call (analyze_references): returns [1] (only the 'list files' turn is relevant to sorting)
    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    # Second call (main execution): returns tool call to sort files
    tool_call = MockToolCall("call_new", "execute_bash_command", {"command": "ls -S", "explanation": "sort files"})
    msg_execute = MockMessage(tool_calls=[tool_call])
    # Third call (filter command): returns False (does not modify)
    msg_filter = MockMessage(content="FALSE")
    
    mock_completion.side_effect = [
        MockResponse(msg_analyze),
        MockResponse(msg_execute),
        MockResponse(msg_filter)
    ]
    
    mock_execute_bash.return_value = "file2\nfile1"
    
    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    
    agent.run_single("now sort them by size")
    
    # Verify that conversation history passed to main model included the retrieved turn 1 but not turn 2
    history_roles = [msg["role"] for msg in agent.conversation_history]
    assert "system" in history_roles
    
    # User message 1 (prompt of turn 1)
    assert agent.conversation_history[1]["role"] == "user"
    assert agent.conversation_history[1]["content"] == "list files"
    
    # Assistant message 1 (tool call for turn 1)
    assert agent.conversation_history[2]["role"] == "assistant"
    assert agent.conversation_history[2]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls", "explanation": "execute ls"}'
    
    # Tool output (result of turn 1)
    assert agent.conversation_history[3]["role"] == "tool"
    assert agent.conversation_history[3]["content"] == "file1\nfile2"
    
    # User message 2 (the new instruction)
    assert agent.conversation_history[4]["role"] == "user"
    assert agent.conversation_history[4]["content"] == "now sort them by size"
    
    # Verify the third action ("make dir") was excluded from the context
    for msg in agent.conversation_history:
        if msg.get("content") == "make dir" or msg.get("content") == "mkdir src":
            pytest.fail("Excluded history turn was incorrectly included in LLM context")
            
    # Verify new turn was appended to history file
    metadata = history_manager.get_history_metadata()
    assert len(metadata) == 3
    assert metadata[2]["prompt"] == "now sort them by size"
    assert metadata[2]["command"] == "ls -S"


def test_resolve_transitive_dependencies(tmp_path, monkeypatch):
    """Verify that _resolve_transitive_dependencies correctly resolves transitively chained dependencies."""
    from llm_communicator import history_manager
    test_file = tmp_path / "test_transitive_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)

    # Turn 1: independent
    history_manager.append_history_turn("list files", "ls", "file1\nfile2", relevant_ids=[])
    # Turn 2: depends on Turn 1
    history_manager.append_history_turn("find executable", "find . -perm /111", "", relevant_ids=[1])
    # Turn 3: depends on Turn 2
    history_manager.append_history_turn("sort them", "sort", "", relevant_ids=[2])

    agent = BashToolAgent(api_key="fake-key")
    
    # Resolve deps for Turn 3 (starts with [2])
    resolved = agent._resolve_transitive_dependencies([2])
    assert resolved == [1, 2]

    # Resolve deps for independent or empty
    assert agent._resolve_transitive_dependencies([]) == []
    assert agent._resolve_transitive_dependencies([1]) == [1]


def test_find_project_root(tmp_path, monkeypatch):
    """Verify that find_project_root finds the root based on project markers."""
    from llm_communicator import history_manager
    
    # Create a dummy structure: root/subdir
    root_dir = tmp_path / "project_root"
    subdir = root_dir / "subdir"
    subdir.mkdir(parents=True)
    
    # Place a marker (pyproject.toml) in root
    (root_dir / "pyproject.toml").touch()
    
    # Mock Path.cwd() to return subdir
    monkeypatch.setattr(Path, "cwd", lambda: subdir)
    
    # find_project_root should find root_dir
    resolved = history_manager.find_project_root()
    assert resolved == root_dir.resolve()


def test_cli_new_resets_history(tmp_path, monkeypatch):
    """Verify that 'doit -n' or 'doit --new' clears history and exits cleanly."""
    from doit_module.__main__ import main
    from llm_communicator import history_manager

    # Write some history
    history_manager.append_history_turn("list files", "ls", "file1\nfile2", relevant_ids=[])
    assert len(history_manager.get_history_metadata()) == 1

    # Mock sys.argv to run `doit -n` without instructions
    monkeypatch.setattr(sys, "argv", ["doit", "-n"])

    # Mock sys.exit to capture exit code
    exit_mock = MagicMock()
    monkeypatch.setattr(sys, "exit", exit_mock)

    # Mock print to avoid stdout noise
    print_mock = MagicMock()
    monkeypatch.setattr("builtins.print", print_mock)

    main()

    # History should be cleared
    assert len(history_manager.get_history_metadata()) == 0
    exit_mock.assert_called_once_with(0)
    print_mock.assert_any_call("Session history cleared and new session started.")


def test_cli_no_args_prints_help(monkeypatch):
    """Verify that 'doit' with no args prints help and exits with error code 1."""
    from doit_module.__main__ import main

    monkeypatch.setattr(sys, "argv", ["doit"])

    exit_mock = MagicMock()
    monkeypatch.setattr(sys, "exit", exit_mock)

    import argparse
    help_mock = MagicMock()
    monkeypatch.setattr(argparse.ArgumentParser, "print_help", help_mock)

    main()

    help_mock.assert_called_once()
    exit_mock.assert_called_once_with(1)


def test_history_system_instruction_rules():
    """Verify HISTORY_SYSYEM_INSTRUCTION contains the semantic matching, chronological, and safety check rules."""
    from llm_communicator.llm_bash import HISTORY_SYSYEM_INSTRUCTION
    
    assert "semantic and logical dependencies" in HISTORY_SYSYEM_INSTRUCTION
    assert "chronological order" in HISTORY_SYSYEM_INSTRUCTION
    assert "SAFETY CHECK" in HISTORY_SYSYEM_INSTRUCTION
    assert "most recent one (the command with the larger ID)" in HISTORY_SYSYEM_INSTRUCTION


def test_doit_system_prompt_cancelled_rules():
    """Verify DOIT_SYSTEM_PROMPT Rule 7 details how cancelled or rejected commands are handled."""
    from fixtures import DOIT_SYSTEM_PROMPT
    
    assert "CANCELLED" in DOIT_SYSTEM_PROMPT or "cancelled" in DOIT_SYSTEM_PROMPT
    assert "REJECTED" in DOIT_SYSTEM_PROMPT or "rejected" in DOIT_SYSTEM_PROMPT
    assert "since the previous step/s was not executed, doing a command here does not make sense" in DOIT_SYSTEM_PROMPT







