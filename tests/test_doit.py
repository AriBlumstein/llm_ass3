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
    NO_ANSWER_SENTINEL,
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
    """Automatically isolate all tests from the host's actual history AND memory files."""
    from llm_communicator import history_manager, memory_manager
    test_file = tmp_path / "test_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)
    # Memory is a GLOBAL store (~/.doit/memories.json) - isolate it so no test touches the real
    # one, and every BashToolAgent constructed in tests sees an empty store unless it adds memories.
    mem_file = tmp_path / "test_memories.json"
    monkeypatch.setattr(memory_manager, "get_memory_file_path", lambda: mem_file)


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


def test_exit_code_zero_is_labelled_success():
    """A successful command (exit 0) must be clearly marked SUCCESS so the model never reads
    `0` as a failure (the touch-then-delete bug)."""
    output = execute_bash("true", verbose=True)            # exits 0, no stdout/stderr
    assert "SUCCESS" in output
    assert "FAILED" not in output


def test_nonzero_exit_code_is_labelled_failed():
    output = execute_bash("sh -c 'exit 3'", verbose=True)  # exits 3
    assert "3 (FAILED)" in output


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
    assert metadata[0] == {"id": 1, "prompt": "list files", "command": "ls", "suggested_command": ""}
    assert metadata[1] == {"id": 2, "prompt": "show process", "command": "ps", "suggested_command": ""}
    
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
    
    # Since the heuristic is active, self-contained queries like "hello" do not call the LLM
    mock_completion.reset_mock()
    mock_completion.return_value = MockResponse(mock_message)
    assert agent._analyze_references("hello", metadata) == []
    assert not mock_completion.called

    # Mock LLM returning empty or invalid
    mock_completion.reset_mock()
    mock_message_empty = MockMessage(content='{"relevant_ids": []}')
    mock_completion.return_value = MockResponse(mock_message_empty)
    assert agent._analyze_references("do this instead", metadata) == []
    assert mock_completion.called


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

    # Reconstructed real history begins after the system prompt + injected tool-calling few-shot.
    from llm_communicator.backup_system_prompts import FEWSHOT_TOOLCALL
    base = 1 + len(FEWSHOT_TOOLCALL)

    # User message 1 (prompt of turn 1)
    assert agent.conversation_history[base]["role"] == "user"
    assert agent.conversation_history[base]["content"] == "list files"

    # Assistant message 1 (tool call for turn 1)
    assert agent.conversation_history[base + 1]["role"] == "assistant"
    assert agent.conversation_history[base + 1]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls", "explanation": "execute ls"}'

    # Tool output (result of turn 1)
    assert agent.conversation_history[base + 2]["role"] == "tool"
    assert agent.conversation_history[base + 2]["content"] == "file1\nfile2"

    # User message 2 (the new instruction)
    assert agent.conversation_history[base + 3]["role"] == "user"
    assert agent.conversation_history[base + 3]["content"] == "now sort them by size"
    
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

    # Mock sys.exit to raise SystemExit
    def mock_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", mock_exit)

    # Mock print to avoid stdout noise
    print_mock = MagicMock()
    monkeypatch.setattr("builtins.print", print_mock)

    with pytest.raises(SystemExit) as excinfo:
        main()

    # History should be cleared
    assert len(history_manager.get_history_metadata()) == 0
    assert excinfo.value.code == 0
    print_mock.assert_any_call("Session history cleared and new session started.")


def test_cli_no_args_prints_help(monkeypatch):
    """Verify that 'doit' with no args prints help and exits with error code 1."""
    from doit_module.__main__ import main

    monkeypatch.setattr(sys, "argv", ["doit"])

    def mock_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", mock_exit)

    import argparse
    help_mock = MagicMock()
    monkeypatch.setattr(argparse.ArgumentParser, "print_help", help_mock)

    with pytest.raises(SystemExit) as excinfo:
        main()

    help_mock.assert_called_once()
    assert excinfo.value.code == 1


def test_history_system_instruction_rules():
    """Verify HISTORY_SYSYEM_INSTRUCTION contains the semantic matching, chronological, and safety check rules."""
    from llm_communicator.llm_bash import HISTORY_SYSYEM_INSTRUCTION
    
    assert "semantic and logical dependencies" in HISTORY_SYSYEM_INSTRUCTION
    assert "chronological order" in HISTORY_SYSYEM_INSTRUCTION
    assert "SAFETY CHECK" in HISTORY_SYSYEM_INSTRUCTION
    assert "most recent one (the command with the larger ID)" in HISTORY_SYSYEM_INSTRUCTION
    # The resolver must treat answer turns (suggested-but-not-executed) as linkable so
    # "execute that" / "modify it" can refer back to them.
    assert "Suggested (not executed)" in HISTORY_SYSYEM_INSTRUCTION


def test_doit_system_prompt_cancelled_rules():
    """Verify DOIT_SYSTEM_PROMPT Rule 7 details how cancelled or rejected commands are handled."""
    from fixtures import DOIT_SYSTEM_PROMPT
    
    assert "CANCELLED" in DOIT_SYSTEM_PROMPT or "cancelled" in DOIT_SYSTEM_PROMPT
    assert "REJECTED" in DOIT_SYSTEM_PROMPT or "rejected" in DOIT_SYSTEM_PROMPT
    assert "since the previous step/s was not executed, doing a command here does not make sense" in DOIT_SYSTEM_PROMPT
def test_doit_system_prompt_missing_context_rules():
    """Verify DOIT_SYSTEM_PROMPT Rule 7 details how missing previous context is handled."""
    from fixtures import DOIT_SYSTEM_PROMPT
    
    assert "I do not see any previous command within the current window that applies to this" in DOIT_SYSTEM_PROMPT


@patch("builtins.print")
@patch("llm_communicator.llm_bash.litellm.completion")
def test_json_error_parsing(mock_completion, mock_print, tmp_path, monkeypatch):
    """Verify that if the LLM returns JSON with an error key, it parses and prints it correctly."""
    from llm_communicator import history_manager
    test_file = tmp_path / "test_error_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)

    # Setup mock LLM behavior:
    # First call (analyze_references): returns empty dependency list
    msg_analyze = MockMessage(content='{"relevant_ids": []}')
    # Second call (main execution): returns JSON error response
    msg_execute = MockMessage(content='{"error": "error from qwen"}')

    mock_completion.return_value = MockResponse(msg_execute)

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    agent.run_single("some prompt")

    mock_print.assert_any_call("error from qwen")


def test_doit_system_prompt_augmented_rules():
    """Verify DOIT_SYSTEM_PROMPT Rule 7 details connection matching and generation strategies."""
    from fixtures import DOIT_SYSTEM_PROMPT
    
    assert "connect to the previous command based on either the command itself or its prompt" in DOIT_SYSTEM_PROMPT
    assert "Appending/chaining/piping" in DOIT_SYSTEM_PROMPT
    assert "Working on the output" in DOIT_SYSTEM_PROMPT
    assert "Understanding the previous prompt to know the user's intentions" in DOIT_SYSTEM_PROMPT
def test_doit_system_prompt_filesystem_metadata_rules():
    """Verify DOIT_SYSTEM_PROMPT Rule 7 details filesystem metadata / properties checking rule."""
    from fixtures import DOIT_SYSTEM_PROMPT
    
    assert "neither the previous command nor its output contains the necessary metadata or attributes" in DOIT_SYSTEM_PROMPT
    assert "generate a new Bash command to query the filesystem directly to retrieve the needed information" in DOIT_SYSTEM_PROMPT
    assert "invoke the `execute_bash_command` tool" in DOIT_SYSTEM_PROMPT

def test_rejection_warning_direct_response(tmp_path, monkeypatch):
    """Verify that if the LLM returns a contextual warning rejection inside a JSON content response, it is printed/logged without execution."""
    from llm_communicator import history_manager
    test_file = tmp_path / "test_rejection_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)

    # Mock analyze references to return empty dependencies, then direct warning JSON in content
    msg_execute = MockMessage(content='{"response_text": "I do not see any previous command within the current window that applies to this"}')

    with patch("builtins.print") as mock_print, patch("llm_communicator.llm_bash.litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse(msg_execute)

        # Patch execute_bash to ensure it is never called
        mock_execute_bash = MagicMock(return_value="should not be executed")
        monkeypatch.setattr("llm_communicator.llm_bash.execute_bash", mock_execute_bash)

        agent = BashToolAgent(api_key="fake-key")
        agent.tool_calling = True

        agent.run_single("delete the file we just created")

        # Assert that print was called with the warning string
        mock_print.assert_any_call("I do not see any previous command within the current window that applies to this")
        # Verify that execute_bash was NOT called
        mock_execute_bash.assert_not_called()

    # Verify history entry
    assert test_file.exists()
    with open(test_file, "r", encoding="utf-8") as f:
        turn = json.loads(f.readline().strip())
        assert turn["command"] == ""
        assert turn["output"] == "I do not see any previous command within the current window that applies to this"


def test_ctx_num_passed_in_non_tool_calling():
    """Verify CTX_NUM is passed as num_ctx for a non-tool-calling OLLAMA model (num_ctx is an
    Ollama-only option)."""
    from fixtures import CTX_NUM
    from llm_communicator.llm_bash import BashToolAgent

    with patch("llm_communicator.llm_bash.load_config") as mock_load_config, \
         patch("llm_communicator.llm_bash.litellm.completion") as mock_completion, \
         patch("llm_communicator.llm_bash.execute_bash", return_value="[Success]"), \
         patch("builtins.input", return_value="y"):

        mock_load_config.return_value = ("ollama/gemma3:4b", None, False)

        msg_gen = MockMessage(content='{"command": "mkdir -p new_folder", "explanation": "create folder"}')
        msg_filter = MockMessage(content="DECISION: YES\\nEXPLANATION: Creates a folder")

        mock_completion.side_effect = [
            MockResponse(msg_gen),
            MockResponse(msg_filter)
        ]

        agent = BashToolAgent()
        agent.run_single("Make a folder")

        assert mock_completion.call_count >= 2
        for call_args in mock_completion.call_args_list:
            kwargs = call_args[1]
            assert kwargs.get("num_ctx") == CTX_NUM


def test_num_ctx_not_passed_for_openai_model():
    """num_ctx must NOT be sent for an OpenAI model (it is Ollama-only), even in non-tool mode."""
    from llm_communicator.llm_bash import BashToolAgent

    with patch("llm_communicator.llm_bash.load_config") as mock_load_config, \
         patch("llm_communicator.llm_bash.litellm.completion") as mock_completion, \
         patch("llm_communicator.llm_bash.execute_bash", return_value="[Success]"), \
         patch("builtins.input", return_value="y"):

        mock_load_config.return_value = ("openai/gpt-5.4-nano", None, False)

        msg_gen = MockMessage(content='{"command": "mkdir -p new_folder", "explanation": "create folder"}')
        msg_filter = MockMessage(content="DECISION: YES\\nEXPLANATION: Creates a folder")
        mock_completion.side_effect = [MockResponse(msg_gen), MockResponse(msg_filter)]

        agent = BashToolAgent()
        agent.run_single("Make a folder")

        assert mock_completion.call_count >= 2
        for call_args in mock_completion.call_args_list:
            assert "num_ctx" not in call_args[1]


def test_case2_self_contained_command_with_context(tmp_path, monkeypatch):
    """Verify that a self-contained command (like 'create a file called klum') is executed even when context exists in the history."""
    from llm_communicator import history_manager
    test_file = tmp_path / "test_case2_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)

    # 1. Setup existing history: 1 turn (e.g. list files)
    history_manager.append_history_turn("list files in the cwd", "ls -l", "file1\nfile2")

    # 2. Setup mock LLM behavior:
    # First call (analyze_references): returns [1] (simulating reference match)
    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    # Second call (main execution): returns tool call to create file 'klum'
    tool_call = MockToolCall("call_1", "execute_bash_command", {"command": "touch klum", "explanation": "create file called klum"})
    msg_execute = MockMessage(tool_calls=[tool_call])
    # Third call (filter command): returns YES (modifies)
    msg_filter = MockMessage(content="DECISION: YES\nEXPLANATION: Creates a file")

    with patch("llm_communicator.llm_bash.litellm.completion") as mock_completion, \
         patch("llm_communicator.llm_bash.execute_bash", return_value="[Success]") as mock_execute_bash, \
         patch("builtins.input", return_value="y"):
        
        mock_completion.side_effect = [
            MockResponse(msg_analyze),
            MockResponse(msg_execute),
            MockResponse(msg_filter)
        ]

        agent = BashToolAgent(api_key="fake-key")
        agent.tool_calling = True

        agent.run_single("create a file called klum instead of this")

        # Verify that execute_bash was called to create the file
        mock_execute_bash.assert_called_with("touch klum")


# =====================================================================
# SECTION 9: Clarifying Question Tests (tool decides, sub-LLM authors)
# =====================================================================

def _author(question, options=None):
    """MockResponse for the _author_clarification sub-call."""
    obj = {"question": question}
    if options is not None:
        obj["options"] = options
    return MockResponse(MockMessage(content=json.dumps(obj)))


@patch("llm_communicator.tools.litellm.completion")
def test_author_clarification_parses(mock_completion):
    """_author_clarification (in tools.py, configured from doit.cfg) turns the sub-call JSON
    into (question, options)."""
    from llm_communicator.tools import _author_clarification
    mock_completion.return_value = _author("Which date?", ["a", "b"])
    question, options = _author_clarification("sort by date")
    assert question == "Which date?"
    assert options == ["a", "b"]


@patch("llm_communicator.tools.litellm.completion")
def test_author_clarification_falls_back_on_junk(mock_completion):
    """If the sub-call returns unparseable output, a generic question is produced."""
    from llm_communicator.tools import _author_clarification
    mock_completion.return_value = MockResponse(MockMessage(content="not json"))
    question, options = _author_clarification("do the thing")
    assert "do the thing" in question
    assert options is None


def test_ask_clarification_reprompts_on_out_of_range_choice():
    """Picking a number outside the offered options re-shows the menu and re-prompts locally,
    with NO call to the LLM, then resolves the valid pick to its option text."""
    from llm_communicator.tools import ask_clarification
    with patch("builtins.input", side_effect=["4", "2"]) as mock_input:
        result = ask_clarification("Pick one:", ["alpha", "beta", "gamma"])
    assert result == "beta"
    assert mock_input.call_count == 2


def test_ask_clarification_in_range_choice_maps_to_option():
    """A valid numeric pick maps to the option text in a single prompt."""
    from llm_communicator.tools import ask_clarification
    with patch("builtins.input", side_effect=["3"]) as mock_input:
        result = ask_clarification("Pick one:", ["alpha", "beta", "gamma"])
    assert result == "gamma"
    assert mock_input.call_count == 1


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_clarification_tool_calling(mock_execute_bash, mock_completion):
    """Tool-calling: agent calls ask_user_clarification, the sub-call authors the question,
    the user answers, and the agent then runs a command."""
    clar_call = MockToolCall("call_clar", "ask_user_clarification", {"reason": "ambiguous date"})
    exec_call = MockToolCall("call_exec", "execute_bash_command",
                             {"command": "ls -lt ~", "explanation": "sort by mtime"})
    mock_completion.side_effect = [
        MockResponse(MockMessage(tool_calls=[clar_call])),     # generation -> clarify tool
        _author("Which date?", ["modification date", "access date"]),  # sub-call authors
        MockResponse(MockMessage(tool_calls=[exec_call])),     # generation -> command
        MockResponse(MockMessage(content="DECISION: NO")),     # filter
    ]
    mock_execute_bash.return_value = "ok"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    with patch("builtins.input", return_value="modification date") as mock_input:
        agent.run_single("list home folder sorted by date")
        mock_input.assert_called()

    tool_msgs = [m for m in agent.conversation_history
                 if m.get("role") == "tool" and m.get("name") == "ask_user_clarification"]
    assert tool_msgs and "modification date" in tool_msgs[-1]["content"]
    mock_execute_bash.assert_called_once_with("ls -lt ~")


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_clarification_fallback(mock_execute_bash, mock_completion, mock_load_config):
    """Fallback JSON: agent sets needs_clarification, sub-call authors, answer fed back, command runs."""
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    clar_json = json.dumps({"needs_clarification": True, "executable": False, "command": "", "rule_triggered": 8})
    cmd_json = json.dumps({"executable": True, "command": "ls -lt ~", "explanation": "mtime"})
    mock_completion.side_effect = [
        MockResponse(MockMessage(content=clar_json)),          # generation -> needs_clarification
        _author("Which date?", ["modification date", "access date"]),  # sub-call authors
        MockResponse(MockMessage(content=cmd_json)),           # generation -> command
        MockResponse(MockMessage(content="DECISION: NO")),     # filter
    ]
    mock_execute_bash.return_value = "ok"

    agent = BashToolAgent()
    assert agent.tool_calling is False

    with patch("builtins.input", return_value="modification date") as mock_input:
        agent.run_single("list home folder sorted by date")
        mock_input.assert_called()

    user_msgs = [m["content"] for m in agent.conversation_history if m.get("role") == "user"]
    assert any("modification date" in c for c in user_msgs)
    mock_execute_bash.assert_called_once_with("ls -lt ~")


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_clarification_empty_answer_uses_default(mock_execute_bash, mock_completion):
    """Empty answer (twice) sends the sentinel back, and the agent proceeds with a default."""
    clar_call = MockToolCall("call_clar", "ask_user_clarification", {})
    exec_call = MockToolCall("call_exec", "execute_bash_command",
                             {"command": "ls -lt ~", "explanation": "default mtime"})
    mock_completion.side_effect = [
        MockResponse(MockMessage(tool_calls=[clar_call])),
        _author("Which date?"),
        MockResponse(MockMessage(tool_calls=[exec_call])),
        MockResponse(MockMessage(content="DECISION: NO")),
    ]
    mock_execute_bash.return_value = "ok"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    with patch("builtins.input", return_value="") as mock_input:
        agent.run_single("list home folder sorted by date")
        assert mock_input.call_count == 2

    tool_msgs = [m for m in agent.conversation_history
                 if m.get("role") == "tool" and m.get("name") == "ask_user_clarification"]
    assert tool_msgs[-1]["content"] == NO_ANSWER_SENTINEL
    mock_execute_bash.assert_called_once_with("ls -lt ~")


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_clarification_round_cap(mock_execute_bash, mock_completion):
    """The agent stops asking after MAX_CLARIFICATION_ROUNDS and the clarify tool is withdrawn."""
    from fixtures import MAX_CLARIFICATION_ROUNDS
    clar = lambda: MockResponse(MockMessage(tool_calls=[
        MockToolCall("call_clar", "ask_user_clarification", {})]))
    # Each round: generation (clar) + author sub-call; final round generation only.
    seq = []
    for _ in range(MAX_CLARIFICATION_ROUNDS):
        seq.append(clar())            # generation -> clar
        seq.append(_author("Which?"))  # sub-call authors
    seq.append(clar())                # final generation (clar tool withdrawn, still mocked as clar)
    mock_completion.side_effect = seq

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    with patch("builtins.input", return_value="x"):
        agent.run_single("do something ambiguous")

    gen_calls = [c for c in mock_completion.call_args_list if "tools" in c.kwargs]
    final_names = [t["function"]["name"] for t in gen_calls[-1].kwargs["tools"]]
    assert "ask_user_clarification" not in final_names
    assert "execute_bash_command" in final_names
    mock_execute_bash.assert_not_called()


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_no_clarification_when_clear(mock_execute_bash, mock_completion):
    """A clear request runs directly: no clarify tool call, no sub-call, no prompt."""
    exec_call = MockToolCall("call_exec", "execute_bash_command",
                             {"command": "ls -la", "explanation": "list files"})
    mock_completion.side_effect = [
        MockResponse(MockMessage(tool_calls=[exec_call])),
        MockResponse(MockMessage(content="DECISION: NO")),
    ]
    mock_execute_bash.return_value = "file1"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    with patch("builtins.input") as mock_input:
        agent.run_single("list files")
        mock_input.assert_not_called()

    mock_execute_bash.assert_called_once_with("ls -la")


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_clarification_folds_answer_into_persisted_prompt(mock_execute_bash, mock_completion, tmp_path, monkeypatch):
    """The resolved clarification answer is folded into the prompt stored in history."""
    from llm_communicator import history_manager
    test_file = tmp_path / "clar_history.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)

    clar_call = MockToolCall("call_clar", "ask_user_clarification", {})
    exec_call = MockToolCall("call_exec", "execute_bash_command",
                             {"command": "ls -lt ~", "explanation": "mtime"})
    mock_completion.side_effect = [
        MockResponse(MockMessage(tool_calls=[clar_call])),
        _author("Which date?"),
        MockResponse(MockMessage(tool_calls=[exec_call])),
        MockResponse(MockMessage(content="DECISION: NO")),
    ]
    mock_execute_bash.return_value = "ok"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    with patch("builtins.input", return_value="modification date"):
        agent.run_single("list home folder sorted by date")

    turns = history_manager.get_history_metadata(limit=10)
    assert turns
    assert "[clarified: modification date]" in turns[-1]["prompt"]


def test_author_prompt_documents_question():
    """The clarification-authoring prompt asks for a question (and optional options)."""
    from fixtures import CLARIFY_AUTHOR_PROMPT
    assert '"question"' in CLARIFY_AUTHOR_PROMPT
    assert '"options"' in CLARIFY_AUTHOR_PROMPT


def test_system_prompt_contains_clarification_rule():
    """DOIT_SYSTEM_PROMPT instructs the agent to ask for clarification when ambiguous."""
    from fixtures import DOIT_SYSTEM_PROMPT
    lowered = DOIT_SYSTEM_PROMPT.lower()
    assert "clarif" in lowered
    assert "ambiguous" in lowered
    assert "ask_user_clarification" in DOIT_SYSTEM_PROMPT


def test_fallback_instruction_documents_needs_clarification():
    """The fallback JSON contract documents the needs_clarification flag."""
    from llm_communicator.llm_bash import FALLBACK_SYSTEM_INSTRUCTION
    assert "needs_clarification" in FALLBACK_SYSTEM_INSTRUCTION


# =====================================================================
# SECTION: Richer interactions (answer_question + execute it / modify it)
# =====================================================================

@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_answer_question_does_not_execute_and_persists_suggestion(mock_execute_bash, mock_completion):
    """A how-to question -> answer_question tool: nothing runs, the suggestion is persisted."""
    from llm_communicator import history_manager

    answer_call = MockToolCall("call_ans", "answer_question", {
        "explanation": "Use find with -size to filter by file size.",
        "suggested_command": "find . -size +100M",
    })
    # Empty history => no analyze LLM call; only the generation call happens.
    mock_completion.side_effect = [MockResponse(MockMessage(tool_calls=[answer_call]))]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    with patch("builtins.input") as mock_input:
        agent.run_single("how do I find files larger than 100MB?")
        mock_input.assert_not_called()

    mock_execute_bash.assert_not_called()
    assert mock_completion.call_count == 1

    turns = history_manager.get_history_metadata(limit=10)
    assert len(turns) == 1
    assert turns[-1]["command"] == ""
    assert turns[-1]["suggested_command"] == "find . -size +100M"


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_execute_it_runs_previously_suggested_command(mock_execute_bash, mock_completion):
    """'execute it' resolves a prior answer turn and runs its suggested command."""
    from llm_communicator import history_manager

    # Seed an answer turn that suggested (but did not run) a command.
    history_manager.append_history_turn(
        "how do I find files larger than 100MB?",
        "",
        "Use find with -size to filter by file size.",
        relevant_ids=[],
        suggested_command="find . -size +100M",
    )

    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    exec_call = MockToolCall("call_exec", "execute_bash_command",
                             {"command": "find . -size +100M", "explanation": "find large files"})
    msg_execute = MockMessage(tool_calls=[exec_call])
    msg_filter = MockMessage(content="DECISION: NO")

    mock_completion.side_effect = [
        MockResponse(msg_analyze),
        MockResponse(msg_execute),
        MockResponse(msg_filter),
    ]
    mock_execute_bash.return_value = "./big.iso"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    with patch("builtins.input") as mock_input:
        agent.run_single("execute it")
        mock_input.assert_not_called()  # read-only command -> no confirmation prompt

    mock_execute_bash.assert_called_once_with("find . -size +100M")

    # The prior answer turn was replayed as an answer_question tool call so the model
    # could see the suggested command.
    replayed = [
        tc["function"]["name"]
        for msg in agent.conversation_history
        for tc in (msg.get("tool_calls") or [])
    ]
    assert "answer_question" in replayed


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_modify_it_revises_suggestion_without_executing(mock_execute_bash, mock_completion):
    """'modify it to ...' produces a new answer_question with the revised suggestion; nothing runs."""
    from llm_communicator import history_manager

    history_manager.append_history_turn(
        "how do I find files larger than 100MB?",
        "",
        "Use find with -size.",
        relevant_ids=[],
        suggested_command="find . -size +100M",
    )

    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    answer_call = MockToolCall("call_ans2", "answer_question", {
        "explanation": "Raise the size threshold to 1GB.",
        "suggested_command": "find . -size +1G",
    })
    mock_completion.side_effect = [
        MockResponse(msg_analyze),
        MockResponse(MockMessage(tool_calls=[answer_call])),
    ]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True

    agent.run_single("modify it to find files over 1GB")

    mock_execute_bash.assert_not_called()
    turns = history_manager.get_history_metadata(limit=10)
    assert turns[-1]["suggested_command"] == "find . -size +1G"
    assert turns[-1]["command"] == ""


def test_answer_tool_offered_in_tool_list():
    """The generator is offered the answer_question tool alongside execute/clarify."""
    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    names = {t["function"]["name"] for t in agent._build_tools(include_clarification=True)}
    assert {"execute_bash_command", "answer_question", "ask_user_clarification"} <= names
    # On the final round the clarification tool is withdrawn, but answer_question stays.
    final = {t["function"]["name"] for t in agent._build_tools(include_clarification=False)}
    assert "answer_question" in final
    assert "ask_user_clarification" not in final


def test_system_prompt_documents_answer_rule():
    """DOIT_SYSTEM_PROMPT documents the answer_question / execute-it behavior."""
    from fixtures import DOIT_SYSTEM_PROMPT
    assert "answer_question" in DOIT_SYSTEM_PROMPT
    lowered = DOIT_SYSTEM_PROMPT.lower()
    assert "how-to" in lowered or "how do i" in lowered


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_fallback_answer_persists_suggestion(mock_execute_bash, mock_completion, mock_load_config):
    """Non-tool-calling Rule 9 answer: nothing runs, suggested_command is persisted from JSON."""
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)

    # Empty history => no analyze LLM call; only the generation call.
    msg_gen = MockMessage(content=json.dumps({
        "executable": False, "command": "", "suggested_command": "find . -type f -perm -111 -print",
        "explanation": "how-to", "rule_triggered": 9,
        "response_text": "Use find: find . -type f -perm -111 -print", "needs_clarification": False,
    }))
    mock_completion.side_effect = [MockResponse(msg_gen)]

    agent = BashToolAgent()
    assert agent.tool_calling is False
    agent.run_single("how would I view all the executable files recursively")

    mock_execute_bash.assert_not_called()
    turns = history_manager.get_history_metadata(limit=10)
    assert len(turns) == 1
    assert turns[-1]["command"] == ""
    assert turns[-1]["suggested_command"] == "find . -type f -perm -111 -print"


def test_fallback_instruction_documents_suggested_command():
    """The fallback JSON contract documents the suggested_command field and Rule 9."""
    from llm_communicator.llm_bash import FALLBACK_SYSTEM_INSTRUCTION
    assert "suggested_command" in FALLBACK_SYSTEM_INSTRUCTION
    assert '"rule_triggered": 9' in FALLBACK_SYSTEM_INSTRUCTION


# =====================================================================
# SECTION: Deterministic how-to routing (fallback mode)
# =====================================================================

@pytest.mark.parametrize("text", [
    "how would I view all the executable files recursively in the cwd",
    "how do I count the lines in a file",
    "How can I find empty directories?",
    "how to list hidden files",
    "what's the command to show disk usage",
    "what is the command for listing processes",
])
def test_is_howto_question_matches(text):
    from llm_communicator.tools import is_howto_question
    assert is_howto_question(text) is True


@pytest.mark.parametrize("text", [
    "list the files in my home folder",
    "execute that",
    "modify it to find files over 1GB",
    "create a file called notes.txt",
    "remove the directory we just made",
])
def test_is_howto_question_rejects_non_howto(text):
    from llm_communicator.tools import is_howto_question
    assert is_howto_question(text) is False


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.answer_howto_question")
@patch("llm_communicator.llm_bash.execute_bash")
def test_fallback_howto_route_answers_and_persists(mock_execute_bash, mock_answer, mock_load_config):
    """In fallback mode a how-to question is deterministically routed to the answer sub-call:
    it answers, persists the suggestion, and never runs the main generation loop."""
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    mock_answer.return_value = (
        "Use find to list executable files recursively.",
        "find . -type f -perm -111 -print",
    )

    agent = BashToolAgent()
    assert agent.tool_calling is False

    with patch("llm_communicator.llm_bash.litellm.completion") as mock_completion:
        agent.run_single("how would I view all the executable files recursively in the cwd")
        # The deterministic route must NOT invoke the main generator/filter at all.
        mock_completion.assert_not_called()

    mock_answer.assert_called_once()
    mock_execute_bash.assert_not_called()

    turns = history_manager.get_history_metadata(limit=10)
    assert len(turns) == 1
    assert turns[-1]["command"] == ""
    assert turns[-1]["suggested_command"] == "find . -type f -perm -111 -print"


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.answer_howto_question")
def test_tool_calling_does_not_use_howto_route(mock_answer, mock_load_config):
    """Native tool-calling models keep the Rule 9 / answer_question path; the deterministic
    fallback route must not fire for them."""
    mock_load_config.return_value = ("openai/gpt-5.4-nano", None, True)
    answer_call = MockToolCall("call_ans", "answer_question", {
        "explanation": "Use find ...", "suggested_command": "find . -type f -perm -111",
    })
    with patch("llm_communicator.llm_bash.litellm.completion") as mock_completion:
        mock_completion.side_effect = [MockResponse(MockMessage(tool_calls=[answer_call]))]
        agent = BashToolAgent(api_key="fake-key")
        assert agent.tool_calling is True
        agent.run_single("how would I view executable files")

    mock_answer.assert_not_called()


@patch("llm_communicator.tools.litellm.completion")
def test_answer_howto_question_parses_subcall(mock_completion):
    """answer_howto_question returns (explanation, suggested_command) from the focused sub-call."""
    from llm_communicator.tools import answer_howto_question
    mock_completion.return_value = MockResponse(MockMessage(content=json.dumps({
        "explanation": "Count lines with wc.", "suggested_command": "wc -l file.txt",
    })))
    explanation, suggested = answer_howto_question("how do I count lines in file.txt")
    assert explanation == "Count lines with wc."
    assert suggested == "wc -l file.txt"


# =====================================================================
# SECTION: Deterministic "execute that" routing (fallback mode)
# =====================================================================

@pytest.mark.parametrize("text", [
    "execute that", "run it", "run that", "execute it",
    "do it", "go ahead", "yes, run it", "run the command",
])
def test_is_execute_suggestion_request_matches(text):
    from llm_communicator.tools import is_execute_suggestion_request
    assert is_execute_suggestion_request(text) is True


@pytest.mark.parametrize("text", [
    "list the files", "how would I run a script", "create a file called run",
    "show running processes",
])
def test_is_execute_suggestion_request_rejects(text):
    from llm_communicator.tools import is_execute_suggestion_request
    assert is_execute_suggestion_request(text) is False


def test_get_latest_suggested_command(tmp_path, monkeypatch):
    from llm_communicator import history_manager
    test_file = tmp_path / "h.jsonl"
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda: test_file)

    assert history_manager.get_latest_suggested_command() is None
    history_manager.append_history_turn("q1", "", "ans1", [], suggested_command="ls -la")
    history_manager.append_history_turn("ran", "ls -la", "out", [1])  # executed, no suggestion
    history_manager.append_history_turn("q2", "", "ans2", [], suggested_command="find . -type f")

    assert history_manager.get_latest_suggested_command() == (3, "find . -type f")


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_fallback_execute_route_runs_latest_suggestion(mock_execute_bash, mock_completion, mock_load_config):
    """In fallback mode 'execute that' deterministically runs the latest suggested command
    through the safety filter, without relying on the model to emit an execute turn."""
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)

    history_manager.append_history_turn(
        "how would I list executable files recursively", "", "Use find ...",
        [], suggested_command="find . -type f -perm -111 -print",
    )

    # Only the safety-filter call should hit the model; the main generator must not.
    mock_completion.return_value = MockResponse(MockMessage(content="DECISION: NO"))
    mock_execute_bash.return_value = "./run.sh"

    agent = BashToolAgent()
    with patch("builtins.input") as mock_input:
        agent.run_single("execute that")
        mock_input.assert_not_called()  # read-only -> no [y/N]

    mock_execute_bash.assert_called_once_with("find . -type f -perm -111 -print")
    turns = history_manager.get_history_metadata(limit=10)
    assert turns[-1]["command"] == "find . -type f -perm -111 -print"
    # The executed turn links back to the answer turn that supplied the suggestion.
    full = history_manager.get_full_turns([2])
    assert full[0]["relevant_ids"] == [1]


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_fallback_execute_route_falls_through_without_suggestion(mock_execute_bash, mock_completion, mock_load_config):
    """'execute that' with no prior suggestion falls through to normal handling (no crash)."""
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    # No history. Empty model response should be handled gracefully (no junk turn).
    mock_completion.return_value = MockResponse(MockMessage(content=""))

    agent = BashToolAgent()
    agent.run_single("execute that")

    mock_execute_bash.assert_not_called()
    from llm_communicator import history_manager
    assert history_manager.get_history_metadata() == []  # nothing junk persisted



# =====================================================================
# SECTION: cd / shell-state persistence (hoist to the parent shell)
# =====================================================================

def test_resolve_cd_hoist_existing_dir(tmp_path):
    from llm_communicator.tools import resolve_cd_hoist
    assert resolve_cd_hoist(f"cd {tmp_path}") == os.path.normpath(str(tmp_path))
    assert resolve_cd_hoist(f'cd "{tmp_path}"') == os.path.normpath(str(tmp_path))


@pytest.mark.parametrize("cmd", [
    "ls -la",                 # not a cd
    "cd /no/such/dir_doit_zz",  # target does not exist
    "cd -",                   # OLDPWD - not hoisted
    "cd ~ && ls",             # compound -> stays sandboxed
    "cd x; rm y",             # compound
    "echo $(cd /tmp)",        # substitution
    "cd /tmp | cat",          # piped
])
def test_resolve_cd_hoist_bails(cmd):
    from llm_communicator.tools import resolve_cd_hoist
    assert resolve_cd_hoist(cmd) is None


def test_hoist_cd_writes_sentinel_file(tmp_path, monkeypatch):
    cdfile = tmp_path / "cdfile"
    monkeypatch.setenv("DOIT_CD_FILE", str(cdfile))
    agent = BashToolAgent(api_key="fake-key")
    out = agent._hoist_cd("/home/u/proj")
    assert cdfile.read_text() == "/home/u/proj"
    assert "/home/u/proj" in out


def test_hoist_cd_without_integration_does_not_crash(monkeypatch):
    monkeypatch.delenv("DOIT_CD_FILE", raising=False)
    agent = BashToolAgent(api_key="fake-key")
    out = agent._hoist_cd("/home/u/proj")  # no shell function active
    assert "/home/u/proj" in out


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_run_single_cd_is_hoisted_not_executed(mock_execute_bash, mock_completion, tmp_path, monkeypatch):
    """A plain `cd` is VETTED by the filter, then hoisted to the parent shell (DOIT_CD_FILE), NOT run
    in the subprocess, and the turn is still recorded in history."""
    from llm_communicator import history_manager
    cdfile = tmp_path / "cdfile"
    monkeypatch.setenv("DOIT_CD_FILE", str(cdfile))

    tool_call = MockToolCall("c1", "execute_bash_command", {"command": f"cd {tmp_path}", "explanation": "move"})
    msg_filter = MockMessage(content="DECISION: NO")   # filter now runs on the hoisted command too
    mock_completion.side_effect = [MockResponse(MockMessage(tool_calls=[tool_call])), MockResponse(msg_filter)]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input") as mock_input:
        agent.run_single("go to the project folder")
        mock_input.assert_not_called()           # filter said NO -> no confirmation prompt

    assert cdfile.read_text() == os.path.normpath(str(tmp_path))
    mock_execute_bash.assert_not_called()        # hoisted, never run in the subprocess
    assert mock_completion.call_count == 2       # generation + filter
    md = history_manager.get_history_metadata()
    assert md[-1]["command"] == f"cd {tmp_path}"


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_hoisted_command_can_be_blocked_by_filter(mock_execute_bash, mock_completion, tmp_path, monkeypatch):
    """The filter now vets hoisted commands: if it flags one and the user declines, it is NOT
    hoisted (nothing written to the sentinel file)."""
    cdfile = tmp_path / "cdfile"
    monkeypatch.setenv("DOIT_CD_FILE", str(cdfile))

    tool_call = MockToolCall("c9", "execute_bash_command", {"command": f"cd {tmp_path}", "explanation": "move"})
    msg_filter = MockMessage(content="DECISION: YES")   # filter flags it as modifying
    mock_completion.side_effect = [MockResponse(MockMessage(tool_calls=[tool_call])), MockResponse(msg_filter)]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="n"):     # user declines
        agent.run_single("go to the project folder")

    assert not cdfile.exists()                           # NOT hoisted
    mock_execute_bash.assert_not_called()


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_history_preserved_normal_command_still_sandboxed(mock_execute_bash, mock_completion):
    """KEY REGRESSION: a normal output-producing command still runs in the subprocess via the new
    dispatch and its output is captured into history exactly as before."""
    from llm_communicator import history_manager
    tool_call = MockToolCall("c2", "execute_bash_command", {"command": "ls -la", "explanation": "list"})
    msg_filter = MockMessage(content="DECISION: NO")
    mock_completion.side_effect = [MockResponse(MockMessage(tool_calls=[tool_call])), MockResponse(msg_filter)]
    mock_execute_bash.return_value = "file1\nfile2"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("list the files")

    mock_execute_bash.assert_called_once_with("ls -la")
    full = history_manager.get_full_turns([1])
    assert full[0]["output"] == "file1\nfile2"


# =====================================================================
# SECTION: session-state family hoist (export/alias/set/... except source)
# =====================================================================

@pytest.mark.parametrize("cmd", [
    "export FOO=bar",
    "export PATH=$PATH:/opt/bin",   # parameter expansion is allowed (hoisted as a command)
    "alias g=git",
    "set -o vi",
    "shopt -s globstar",
    "unset FOO",
    "pushd /tmp",
    "popd",
])
def test_resolve_session_state_hoist_matches(cmd):
    from llm_communicator.tools import resolve_session_state_hoist
    assert resolve_session_state_hoist(cmd) == cmd


@pytest.mark.parametrize("cmd", [
    "ls -la",                       # not a session-state builtin
    "cd /tmp",                      # cd is handled separately, not here
    "source ~/.bashrc",            # source is never hoisted
    "export FOO=$(rm -rf ~)",       # command substitution -> refused
    "export A=1 && export B=2",     # chaining -> refused
    "alias x='ls; pwd'",            # metacharacter in value -> refused
    "export FOO=bar | cat",         # piping -> refused
])
def test_resolve_session_state_hoist_bails(cmd):
    from llm_communicator.tools import resolve_session_state_hoist
    assert resolve_session_state_hoist(cmd) is None


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_run_single_export_is_hoisted_not_executed(mock_execute_bash, mock_completion, tmp_path, monkeypatch):
    """An `export` is hoisted to the parent shell (written to DOIT_SHELL_FILE), not run in the
    subprocess, and the turn is still recorded in history."""
    from llm_communicator import history_manager
    shfile = tmp_path / "shfile"
    monkeypatch.setenv("DOIT_SHELL_FILE", str(shfile))

    tool_call = MockToolCall("e1", "execute_bash_command", {"command": "export EDITOR=vim", "explanation": "set editor"})
    msg_filter = MockMessage(content="DECISION: NO")   # filter now runs on the hoisted builtin too
    mock_completion.side_effect = [MockResponse(MockMessage(tool_calls=[tool_call])), MockResponse(msg_filter)]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input") as mock_input:
        agent.run_single("set my editor to vim")
        mock_input.assert_not_called()

    assert shfile.read_text() == "export EDITOR=vim"
    mock_execute_bash.assert_not_called()
    assert mock_completion.call_count == 2   # generation + filter
    md = history_manager.get_history_metadata()
    assert md[-1]["command"] == "export EDITOR=vim"


# =====================================================================
# SECTION: first-run shell-integration bootstrap
# =====================================================================

def _si_setup(monkeypatch, tmp_path, shell="/bin/bash", interactive=True, active=False, rc_exists=True):
    from doit_module import shell_integration
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", shell)
    if active:
        monkeypatch.setenv("DOIT_CD_FILE", str(tmp_path / "cd"))
    else:
        monkeypatch.delenv("DOIT_CD_FILE", raising=False)
    monkeypatch.setattr(shell_integration, "_interactive", lambda: interactive)
    rc = tmp_path / ".bashrc"
    if rc_exists:
        rc.write_text("# existing rc\n")
    return shell_integration, rc


def test_bootstrap_installs_on_accept(monkeypatch, tmp_path):
    si, rc = _si_setup(monkeypatch, tmp_path)
    si.ensure_shell_integration(input_fn=lambda _p: "y")
    assert "doit-init.sh" in rc.read_text()


def test_bootstrap_accepts_on_empty_enter(monkeypatch, tmp_path):
    si, rc = _si_setup(monkeypatch, tmp_path)
    si.ensure_shell_integration(input_fn=lambda _p: "")   # Enter -> default yes
    assert "doit-init.sh" in rc.read_text()


def test_bootstrap_declined_writes_nothing(monkeypatch, tmp_path):
    si, rc = _si_setup(monkeypatch, tmp_path)
    si.ensure_shell_integration(input_fn=lambda _p: "n")
    assert "doit-init.sh" not in rc.read_text()


def test_bootstrap_noop_when_active(monkeypatch, tmp_path):
    si, rc = _si_setup(monkeypatch, tmp_path, active=True)
    called = {"v": False}
    def _inp(_p):
        called["v"] = True
        return "y"
    si.ensure_shell_integration(input_fn=_inp)
    assert called["v"] is False           # never prompts when already integrated
    assert "doit-init.sh" not in rc.read_text()


def test_bootstrap_noop_when_non_interactive(monkeypatch, tmp_path):
    si, rc = _si_setup(monkeypatch, tmp_path, interactive=False)
    called = {"v": False}
    si.ensure_shell_integration(input_fn=lambda _p: called.__setitem__("v", True) or "y")
    assert called["v"] is False
    assert "doit-init.sh" not in rc.read_text()


def test_bootstrap_does_not_reask_when_already_present(monkeypatch, tmp_path):
    si, rc = _si_setup(monkeypatch, tmp_path)
    init = si._init_script_path()
    rc.write_text(f'source "{init}"\n')   # already installed
    called = {"v": False}
    si.ensure_shell_integration(input_fn=lambda _p: called.__setitem__("v", True) or "y")
    assert called["v"] is False           # accepted previously -> no re-prompt


def test_bootstrap_unsupported_shell_is_skipped(monkeypatch, tmp_path):
    si, rc = _si_setup(monkeypatch, tmp_path, shell="/usr/bin/fish")
    called = {"v": False}
    si.ensure_shell_integration(input_fn=lambda _p: called.__setitem__("v", True) or "y")
    assert called["v"] is False
    assert "doit-init.sh" not in rc.read_text()


def test_bootstrap_picks_zshrc_for_zsh(monkeypatch, tmp_path):
    si, _ = _si_setup(monkeypatch, tmp_path, shell="/usr/bin/zsh")
    (tmp_path / ".zshrc").write_text("# zsh rc\n")
    si.ensure_shell_integration(input_fn=lambda _p: "y")
    assert "doit-init.sh" in (tmp_path / ".zshrc").read_text()


# =====================================================================
# SECTION: Persistent user memory
# =====================================================================

def test_memory_store_crud():
    from llm_communicator import memory_manager
    assert memory_manager.load_memories() == []
    a = memory_manager.add_memory("alpha")
    b = memory_manager.add_memory("beta")
    assert [m["content"] for m in memory_manager.load_memories()] == ["alpha", "beta"]
    memory_manager.update_memory(a, "alpha-2")
    assert memory_manager.load_memories()[0]["content"] == "alpha-2"
    memory_manager.delete_memory(b)  # tombstone
    assert [m["content"] for m in memory_manager.load_memories()] == ["alpha-2"]
    assert memory_manager.add_memory("   ") == -1  # empty ignored


def test_render_memories_block():
    from llm_communicator import memory_manager
    assert memory_manager.render_memories() == ""  # empty for new users
    memory_manager.add_memory("~/x is the project folder")
    block = memory_manager.render_memories()
    assert "KNOWN FACTS ABOUT THE USER" in block
    assert "~/x is the project folder" in block
    assert "MOST RECENT" in block   # recency-precedence hint for conflicting memories


def test_memory_supersession_via_operations():
    """'I changed my mind' style: delete the old memory + add the new; new wins."""
    from llm_communicator import memory_manager
    id1 = memory_manager.add_memory("the user prefers sorting by modification time")
    memory_manager.add_memory("~/x is the project")
    memory_manager.apply_operations([
        {"op": "delete", "id": id1},
        {"op": "add", "content": "when sorting, always ask the user about the order"},
    ])
    active = [m["content"] for m in memory_manager.load_memories()]
    assert "the user prefers sorting by modification time" not in active
    assert "when sorting, always ask the user about the order" in active
    assert "~/x is the project" in active


@pytest.mark.parametrize("text", [
    "remember that ~/x is my project folder",
    "this is my LLM class project folder",
    "I prefer sorting by modification date",
    "I changed my mind about the sorting order, ask me each time",
    "from now on always use long listing",
    "keep in mind I work mostly in ~/dev",
])
def test_is_memory_candidate_matches(text):
    from llm_communicator.tools import is_memory_candidate
    assert is_memory_candidate(text) is True


@pytest.mark.parametrize("text", [
    "list the files", "sort them by size", "how do I find big files",
    "execute that", "create a file called notes.txt",
])
def test_is_memory_candidate_rejects(text):
    from llm_communicator.tools import is_memory_candidate
    assert is_memory_candidate(text) is False


@patch("llm_communicator.tools.litellm.completion")
def test_extract_memories_parses_ops(mock_completion):
    from llm_communicator.tools import extract_memories
    mock_completion.return_value = MockResponse(MockMessage(content=json.dumps({
        "operations": [{"op": "add", "content": "x is the project"}]
    })))
    ops = extract_memories("remember x is the project", [])
    assert ops == [{"op": "add", "content": "x is the project"}]


def test_memory_block_injected_into_system_prompt():
    from llm_communicator import memory_manager
    memory_manager.add_memory("~/school/llms/ass3 is the user's LLM class project folder")
    agent = BashToolAgent(api_key="fake-key")
    assert "LLM class project folder" in agent.system_prompt
    assert "KNOWN FACTS ABOUT THE USER" in agent.system_prompt


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.extract_memories")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_memory_extractor_gated_off_for_plain_command(mock_exec, mock_completion, mock_extract, mock_load_config):
    """A plain command must NOT trigger the memory sub-call."""
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    msg_gen = MockMessage(content='{"executable": true, "command": "ls", "explanation": "list", "rule_triggered": 1, "response_text": ""}')
    msg_filter = MockMessage(content="DECISION: NO")
    mock_completion.side_effect = [MockResponse(msg_gen), MockResponse(msg_filter)]
    mock_exec.return_value = "out"

    agent = BashToolAgent()
    agent.run_single("list the files")
    mock_extract.assert_not_called()


@patch("llm_communicator.llm_bash.load_config")
@patch("llm_communicator.llm_bash.extract_memories")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_action_and_memory_from_one_instruction(mock_exec, mock_completion, mock_extract, mock_load_config):
    """`move to X. this is my project folder.` -> the action runs AND the memory is stored."""
    from llm_communicator import memory_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    msg_gen = MockMessage(content='{"executable": true, "command": "cd /home/u/school/llms/ass3", "explanation": "move", "rule_triggered": 1, "response_text": ""}')
    msg_filter = MockMessage(content="DECISION: NO")
    mock_completion.side_effect = [MockResponse(msg_gen), MockResponse(msg_filter)]
    mock_exec.return_value = ""
    mock_extract.return_value = [{"op": "add", "content": "~/school/llms/ass3 is the user's LLM class project folder"}]

    agent = BashToolAgent()
    agent.run_single("move to ~/school/llms/ass3. this is my LLM class project folder")

    mock_extract.assert_called_once()
    mems = memory_manager.load_memories()
    assert len(mems) == 1 and "class project folder" in mems[0]["content"]


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_history_preserved_with_memory_injected(mock_execute_bash, mock_completion):
    """KEY REGRESSION: with a memory injected into the system prompt, multi-turn history replay
    and output capture must still work exactly as before."""
    from llm_communicator import history_manager, memory_manager

    memory_manager.add_memory("~/school/llms/ass3 is the user's LLM class project folder")
    history_manager.append_history_turn("list files", "ls", "file1\nfile2")

    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    tool_call = MockToolCall("call_new", "execute_bash_command", {"command": "ls -S", "explanation": "sort"})
    msg_execute = MockMessage(tool_calls=[tool_call])
    msg_filter = MockMessage(content="DECISION: NO")
    mock_completion.side_effect = [MockResponse(msg_analyze), MockResponse(msg_execute), MockResponse(msg_filter)]
    mock_execute_bash.return_value = "file2\nfile1"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("now sort them by size")

    # 1) Memory lives in the SYSTEM message...
    sys_msg = agent.conversation_history[0]
    assert sys_msg["role"] == "system"
    assert "LLM class project folder" in sys_msg["content"]

    # 2) ...and the prior turn is still replayed intact AFTER it (history not disturbed).
    assert any(m.get("role") == "user" and m.get("content") == "list files" for m in agent.conversation_history)
    assert any(m.get("role") == "tool" and m.get("content") == "file1\nfile2" for m in agent.conversation_history)

    # 3) Output of the new command is captured and persisted to history as before.
    mock_execute_bash.assert_called_once_with("ls -S")
    md = history_manager.get_history_metadata()
    assert len(md) == 2
    assert md[-1]["command"] == "ls -S"
    full = history_manager.get_full_turns([2])
    assert full[0]["output"] == "file2\nfile1"
