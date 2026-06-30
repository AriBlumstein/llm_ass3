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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)
    # Memory is a GLOBAL store (~/.doit/memories.json) - isolate it so no test touches the real
    # one, and every BashToolAgent constructed in tests sees an empty store unless it adds memories.
    mem_file = tmp_path / "test_memories.json"
    monkeypatch.setattr(memory_manager, "get_memory_file_path", lambda: mem_file)
    # Clear the shell-integration env vars by default so tests don't pick up the developer's REAL
    # recorded commands / session folder (set when doit-init.sh is sourced in the test runner's
    # shell). Tests that exercise these set them explicitly via monkeypatch.setenv.
    for _var in ("DOIT_CMD_LOG", "DOIT_PPID", "DOIT_SHELL_HISTORY"):
        monkeypatch.delenv(_var, raising=False)


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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)
    
    # Check initial states
    assert history_manager.get_history_metadata() == []
    assert history_manager.get_full_turns([1]) == []
    
    # Append turns
    history_manager.append_history_turn("list files", "ls", "file1\nfile2")
    history_manager.append_history_turn("show process", "ps", "pid 123")
    
    # Verify metadata (outputs omitted)
    metadata = history_manager.get_history_metadata()
    assert len(metadata) == 2
    assert metadata[0] == {"id": 1, "source": "doit", "prompt": "list files", "command": "ls", "suggested_command": ""}
    assert metadata[1] == {"id": 2, "source": "doit", "prompt": "show process", "command": "ps", "suggested_command": ""}
    
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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)
    
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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)

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


def test_cli_new_alone_resets_history(monkeypatch):
    """`doit -n` (no instruction) clears history and exits cleanly - no instruction required."""
    from doit_module.__main__ import main
    from llm_communicator import history_manager

    history_manager.append_history_turn("list files", "ls", "file1\nfile2", relevant_ids=[])
    assert len(history_manager.get_history_metadata()) == 1

    monkeypatch.setattr(sys, "argv", ["doit", "-n"])

    def mock_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", mock_exit)
    print_mock = MagicMock()
    monkeypatch.setattr("builtins.print", print_mock)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert history_manager.get_history_metadata() == []
    assert excinfo.value.code == 0
    print_mock.assert_any_call("Session history cleared and new session started.")


@patch("doit_module.__main__.ensure_shell_integration")
@patch.object(BashToolAgent, "run_single")
def test_cli_new_with_instruction_resets_and_runs(mock_run_single, mock_integration, monkeypatch):
    """`doit -n "<instruction>"` clears history (force_new) and then runs the instruction."""
    from doit_module.__main__ import main
    from llm_communicator import history_manager

    history_manager.append_history_turn("list files", "ls", "file1\nfile2", relevant_ids=[])
    monkeypatch.setattr(sys, "argv", ["doit", "-n", "show me my files"])
    main()

    assert history_manager.get_history_metadata() == []
    mock_run_single.assert_called_once_with("show me my files")


def test_cli_no_args_errors_required_argument(monkeypatch, capsys):
    """`doit` with no instruction now errors via argparse (required argument), exit code 2."""
    from doit_module.__main__ import main
    monkeypatch.setattr(sys, "argv", ["doit"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2                       # argparse's "missing required argument"
    assert "instruction" in capsys.readouterr().err


def test_cli_reset_wipes_all_histories_but_keeps_memories(tmp_path, monkeypatch):
    """`doit --reset` deletes EVERY session's history folder under .doit/ while leaving
    memories.json (a sibling of the folders, not inside them) untouched, then exits 0."""
    from doit_module.__main__ import main
    from llm_communicator import history_manager

    # Build a fake .doit root: two session folders (this window + another) plus the memory file.
    fake_root = tmp_path / ".doit"
    (fake_root / "history_111").mkdir(parents=True)
    (fake_root / "history_111" / "doit.jsonl").write_text('{"id": 1}\n')
    (fake_root / "history_222").mkdir(parents=True)
    (fake_root / "history_222" / "cmdlog.tsv").write_text("0\tls\n")
    memories = fake_root / "memories.json"
    memories.write_text('[{"id": 1, "content": "my projects dir is ~/projects", "active": true}]')

    monkeypatch.setattr(history_manager, "doit_root", lambda: fake_root)
    monkeypatch.setattr(sys, "argv", ["doit", "--reset"])

    def mock_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", mock_exit)
    print_mock = MagicMock()
    monkeypatch.setattr("builtins.print", print_mock)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 0
    # All history folders gone...
    assert not (fake_root / "history_111").exists()
    assert not (fake_root / "history_222").exists()
    assert list(fake_root.glob("history_*")) == []
    # ...but memories survive, byte-for-byte.
    assert memories.exists()
    assert "my projects dir" in memories.read_text()
    print_mock.assert_any_call("Reset complete: cleared history for 2 session(s). Memories were kept.")


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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)

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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)

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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)

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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)

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
    "how does one make a file executable",       # "does one" - the reported gemma failure
    "how does someone delete a directory",
    "how can you find large files",
    "how should I check disk space",
    "what's the best way to compress a folder",
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
    "how many files are in this directory",      # "how many" - NOT a how-to
    "how big is this file",
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
    monkeypatch.setattr(history_manager, "get_history_file_path", lambda *a, **k: test_file)

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


# =====================================================================
# SECTION: User awareness (recent user shell commands + unified ordering)
# =====================================================================

def test_parse_shell_history():
    from llm_communicator.tools import parse_shell_history
    raw = "  501  cd ~/x\n  502  mkdir data\n  503  doit \"summarize\"\n"
    assert parse_shell_history(raw) == [(501, "cd ~/x"), (502, "mkdir data"), (503, 'doit "summarize"')]
    assert parse_shell_history("") == []


@pytest.mark.parametrize("cmd,expected", [
    ('doit "make a folder"', True),
    ("doit -n", True),
    ("cd ~/x", False),
    ("mkdir doit_dir", False),   # not a doit invocation
    ("python train.py", False),
])
def test_is_doit_invocation(cmd, expected):
    from llm_communicator.tools import is_doit_invocation
    assert is_doit_invocation(cmd) is expected


def test_get_last_user_hist_n():
    from llm_communicator import history_manager
    assert history_manager.get_last_user_hist_n() == 0
    history_manager.append_history_turn("", "cd ~/x", "", source="user", hist_n=501)
    history_manager.append_history_turn("make file", "touch a", "ok")   # doit turn, no hist_n
    history_manager.append_history_turn("", "mkdir data", "", source="user", hist_n=503)
    assert history_manager.get_last_user_hist_n() == 503


@patch("llm_communicator.llm_bash.load_config")
def test_sync_user_history_dedups_and_drops_doit(mock_load_config, monkeypatch):
    """User commands are imported once (dedup via hist_n), `doit ...` lines are dropped, and
    re-running with the same window adds nothing new."""
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    agent = BashToolAgent()

    monkeypatch.setenv("DOIT_SHELL_HISTORY", "  10  cd ~/x\n  11  mkdir data\n  12  doit \"hi\"\n")
    agent._sync_user_history()
    turns = history_manager.get_history_metadata()
    user_cmds = [t["command"] for t in turns if t["source"] == "user"]
    assert user_cmds == ["cd ~/x", "mkdir data"]   # doit line dropped

    # same window again -> nothing new
    agent._sync_user_history()
    assert len([t for t in history_manager.get_history_metadata() if t["source"] == "user"]) == 2

    # newer commands -> only the new one imported
    monkeypatch.setenv("DOIT_SHELL_HISTORY", "  12  doit \"hi\"\n  13  rm klum\n")
    agent._sync_user_history()
    user_cmds = [t["command"] for t in history_manager.get_history_metadata() if t["source"] == "user"]
    assert user_cmds == ["cd ~/x", "mkdir data", "rm klum"]


@patch("llm_communicator.llm_bash.load_config")
def test_sync_noop_without_shell_history(mock_load_config, monkeypatch):
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    agent = BashToolAgent()
    agent._sync_user_history()
    assert history_manager.get_history_metadata() == []


def test_activity_block_tags_user_and_doit():
    from llm_communicator import history_manager
    history_manager.append_history_turn("", "cd ~/x", "", source="user", hist_n=1)
    history_manager.append_history_turn("make file", "touch klum", "ok")   # doit
    history_manager.append_history_turn("", "rm klum", "", source="user", hist_n=2)

    agent = BashToolAgent(api_key="fake-key")
    block = agent._build_activity_block()
    assert "CURRENT DIRECTORY:" in block
    assert "[user] cd ~/x" in block
    assert "[doit] touch klum" in block
    assert "[user] rm klum" in block
    # ordering preserved: doit touch comes before the user rm (undo)
    assert block.index("[doit] touch klum") < block.index("[user] rm klum")


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_user_turn_is_linked_and_replayed(mock_execute_bash, mock_completion):
    """A file the user created MANUALLY is a first-class linkable turn: 'delete the file I just made'
    resolves to it and it is replayed in the conversation (next to the instruction), so the agent
    deletes the right file - not a hallucinated one."""
    from llm_communicator import history_manager
    history_manager.append_history_turn("", "touch klum", "", source="user", hist_n=1)

    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    tool_call = MockToolCall("d1", "execute_bash_command", {"command": "rm klum", "explanation": "delete"})
    msg_filter = MockMessage(content="DECISION: YES")
    mock_completion.side_effect = [
        MockResponse(msg_analyze),
        MockResponse(MockMessage(tool_calls=[tool_call])),
        MockResponse(msg_filter),
    ]
    mock_execute_bash.return_value = "[Success]"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("delete the file I just made")

    # The user's manual `touch klum` was replayed as a note in the conversation...
    assert any("[I ran this command directly in the terminal]: touch klum" in (m.get("content") or "")
               for m in agent.conversation_history)
    # ...and the agent deleted that exact file.
    mock_execute_bash.assert_called_once_with("rm klum")


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_user_turn_folded_next_to_instruction_in_fallback(mock_execute_bash, mock_completion):
    """In fallback mode the user's manual command is folded into the SAME user message as the
    instruction (max salience for weak models) without breaking user/assistant alternation."""
    from llm_communicator import history_manager
    history_manager.append_history_turn("", "touch klum", "", source="user", hist_n=1)

    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    msg_gen = MockMessage(content='{"executable": true, "command": "rm klum", "explanation": "delete", "rule_triggered": 1, "response_text": ""}')
    msg_filter = MockMessage(content="DECISION: YES")
    mock_completion.side_effect = [MockResponse(msg_analyze), MockResponse(msg_gen), MockResponse(msg_filter)]
    mock_execute_bash.return_value = "[Success]"

    with patch("llm_communicator.llm_bash.load_config", return_value=("ollama/qwen3:4b-instruct", None, False)):
        agent = BashToolAgent()
        with patch("builtins.input", return_value="y"):
            agent.run_single("delete the file I just made")

    # Some user message carries BOTH the user's command note and the instruction together (folded).
    user_msgs = [m["content"] for m in agent.conversation_history if m["role"] == "user"]
    assert any("touch klum" in c and "delete the file I just made" in c for c in user_msgs)
    mock_execute_bash.assert_called_once_with("rm klum")


@patch("llm_communicator.llm_bash.litellm.completion")
def test_what_did_you_just_do_triggers_resolution(mock_completion):
    """'what did you just do' must trip the context heuristic so the resolver runs and links the
    most recent action (not short-circuit to [])."""
    mock_completion.return_value = MockResponse(MockMessage(content='{"relevant_ids": [2]}'))
    agent = BashToolAgent(api_key="fake-key")
    metadata = [
        {"id": 1, "source": "doit", "prompt": "list", "command": "ls", "suggested_command": ""},
        {"id": 2, "source": "doit", "prompt": "delete klum", "command": "rm klum", "suggested_command": ""},
    ]
    ids = agent._analyze_references("what did you just do", metadata)
    mock_completion.assert_called_once()          # resolver ran (not short-circuited)
    assert ids == [2]                              # linked the most recent action


@patch("llm_communicator.llm_bash.litellm.completion")
def test_plain_command_still_short_circuits_resolution(mock_completion):
    """A plain, independent command must still skip the resolver LLM call."""
    agent = BashToolAgent(api_key="fake-key")
    metadata = [{"id": 1, "source": "doit", "prompt": "list", "command": "ls", "suggested_command": ""}]
    ids = agent._analyze_references("create a file called report.txt", metadata)
    mock_completion.assert_not_called()
    assert ids == []


def test_system_prompt_activity_questions_answered_not_run():
    """Rule 10 (backstop) tells the agent to ANSWER 'what did I/you just do' from context, not run a command."""
    from fixtures import DOIT_SYSTEM_PROMPT
    lowered = DOIT_SYSTEM_PROMPT.lower()
    assert "what did you just do" in lowered or "what did i just do" in lowered
    assert "must not run or generate a command" in lowered


@pytest.mark.parametrize("text,subject", [
    ("what did I just do", "user"),
    ("what did i do", "user"),
    ("summarize what I just did", "user"),
    ("recap what I did", "user"),
    ("remind me what I ran", "user"),
    ("what have I been doing", "user"),
    ("what did you just do", "doit"),
    ("what did doit just run", "doit"),
    ("what just happened", "both"),
    ("what's been going on", "both"),
    ("list the files", None),
    ("delete the file I just made", None),
])
def test_is_activity_query(text, subject):
    from llm_communicator.tools import is_activity_query
    assert is_activity_query(text) == subject


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_activity_query_answered_deterministically(mock_execute_bash, mock_completion, monkeypatch):
    """'what did I just do' is answered from history - NO LLM call, NO command run - with correct
    attribution (the user's manual command -> the user)."""
    from llm_communicator import history_manager
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    history_manager.append_history_turn("", "touch klum", "", source="user", hist_n=1)

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("what did I just do")

    mock_completion.assert_not_called()           # no LLM at all
    mock_execute_bash.assert_not_called()          # no command run (the qwen `ls -l` bug)
    last = history_manager.get_full_turns([2])[0]
    assert last["command"] == ""                   # answered, not a command turn
    assert "touch klum" in last["output"]
    assert "Your most recent command was" in last["output"]


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_activity_query_doit_first_person(mock_execute_bash, mock_completion):
    """'what did you just do' reports doit's own last action in the first person."""
    from llm_communicator import history_manager
    history_manager.append_history_turn("delete klum", "rm -f klum", "0 (SUCCESS)")  # doit turn

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("what did you just do")

    mock_completion.assert_not_called()
    mock_execute_bash.assert_not_called()
    last = history_manager.get_full_turns([2])[0]
    assert "rm -f klum" in last["output"]
    assert "My most recent action was" in last["output"]


@pytest.mark.parametrize("text,subject", [
    ("explain what you just did", "doit"),
    ("explain what I just did", "user"),
    ("explain the command I just performed", "user"),
    ("explain that action", "both"),
    ("I just performed a command, explain what it did", "user"),
])
def test_is_activity_query_explain_variants(text, subject):
    from llm_communicator.tools import is_activity_query
    assert is_activity_query(text) == subject


@patch("llm_communicator.tools.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_explain_query_uses_focused_subcall_not_command(mock_execute_bash, mock_completion, monkeypatch):
    """'explain what I just did' reports the user's recent command and explains it via the focused
    sub-call - it does NOT run a command (the gemma `ls` / capability-rejection bugs)."""
    from llm_communicator import history_manager
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    history_manager.append_history_turn("", "touch klum", "", source="user", hist_n=1)
    # the ONLY litellm call is the focused explain sub-call (tools.litellm)
    mock_completion.return_value = MockResponse(MockMessage(content="It creates an empty file named klum."))

    with patch("llm_communicator.llm_bash.load_config", return_value=("ollama/gemma3:4b", None, False)):
        agent = BashToolAgent()
        agent.run_single("explain what I just did")

    mock_execute_bash.assert_not_called()          # no command run
    last = history_manager.get_full_turns([2])[0]
    assert "touch klum" in last["output"]
    assert "creates an empty file named klum" in last["output"]


def test_activity_report_collapses_consecutive_duplicates():
    from llm_communicator import history_manager
    for n in (1, 2, 3):
        history_manager.append_history_turn("", "touch klum", "", source="user", hist_n=n)
    history_manager.append_history_turn("", "git status", "", source="user", hist_n=4)
    agent = BashToolAgent(api_key="fake-key")
    answer = agent._answer_activity_query("user")
    assert answer.count("touch klum") == 1   # 3 consecutive collapsed to 1 (plus the lead line uses git status)


@patch("llm_communicator.llm_bash.load_config")
def test_synced_user_command_has_ran_marker_not_empty_output(mock_load_config, monkeypatch):
    """A synced user command gets an explicit 'it ran' output (not empty), so the model doesn't read
    empty output as a failure."""
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    agent = BashToolAgent()
    monkeypatch.setenv("DOIT_SHELL_HISTORY", "  5  touch klum\n")
    agent._sync_user_history()

    turn = history_manager.get_full_turns([1])[0]
    assert turn["source"] == "user"
    assert turn["output"] != ""
    assert "ran by the user" in turn["output"].lower()


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_user_turn_replay_includes_ran_marker(mock_execute_bash, mock_completion):
    """The replayed user turn carries the 'it ran' marker so the agent won't treat it as failed."""
    from llm_communicator import history_manager
    history_manager.append_history_turn("", "touch klum", "[Ran by the user directly in the terminal and completed; output not captured by doit.]", source="user", hist_n=1)

    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    tool_call = MockToolCall("d1", "execute_bash_command", {"command": "rm klum", "explanation": "delete"})
    msg_filter = MockMessage(content="DECISION: YES")
    mock_completion.side_effect = [MockResponse(msg_analyze), MockResponse(MockMessage(tool_calls=[tool_call])), MockResponse(msg_filter)]
    mock_execute_bash.return_value = "[Success]"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("delete the file I just made")

    assert any("Ran by the user" in (m.get("content") or "") for m in agent.conversation_history)


def test_parse_cmd_log():
    from llm_communicator.tools import parse_cmd_log
    raw = "0\ttouch klum\n1\tfalse\n2\tls /nope\n"
    assert parse_cmd_log(raw) == [(1, "0", "touch klum"), (2, "1", "false"), (3, "2", "ls /nope")]
    assert parse_cmd_log("") == []


@patch("llm_communicator.llm_bash.load_config")
def test_sync_uses_exit_status_log_success_and_failure(mock_load_config, monkeypatch, tmp_path):
    """When DOIT_CMD_LOG is present, user turns carry REAL success/failure from the exit status."""
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    log = tmp_path / "cmdlog.tsv"
    log.write_text("0\ttouch klum\n2\tcat /nonexistent\n")
    monkeypatch.setenv("DOIT_CMD_LOG", str(log))
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)

    agent = BashToolAgent()
    agent._sync_user_history()

    turns = history_manager.get_full_turns([1, 2])
    assert turns[0]["command"] == "touch klum"
    assert "completed successfully (exit 0)" in turns[0]["output"]
    assert turns[1]["command"] == "cat /nonexistent"
    assert "FAILED (exit 2)" in turns[1]["output"]


@patch("llm_communicator.llm_bash.load_config")
def test_cmd_log_preferred_over_fc_history(mock_load_config, monkeypatch, tmp_path):
    """The exit-status log is preferred over the fc -l fallback when both are present."""
    from llm_communicator import history_manager
    mock_load_config.return_value = ("ollama/gemma3:4b", None, False)
    log = tmp_path / "cmdlog.tsv"
    log.write_text("0\tmkdir data\n")
    monkeypatch.setenv("DOIT_CMD_LOG", str(log))
    monkeypatch.setenv("DOIT_SHELL_HISTORY", "  9  some other command\n")

    agent = BashToolAgent()
    agent._sync_user_history()
    cmds = [t["command"] for t in history_manager.get_history_metadata() if t["source"] == "user"]
    assert cmds == ["mkdir data"]   # from the log, not the fc history


# =====================================================================
# SECTION: Session folder (cmdlog + doit history co-located, PID-mismatch-proof)
# =====================================================================
def test_history_in_pid_folder_named_by_doit_ppid(monkeypatch):
    """doit's history lives in `.doit/history_<DOIT_PPID>/doit.jsonl`, named by the shell-pinned
    DOIT_PPID. It is ALWAYS pid-named (never a bare shared `.doit/doit.jsonl`) and does NOT depend on
    DOIT_CMD_LOG, so a stale/odd DOIT_CMD_LOG can't move or un-isolate it."""
    from llm_communicator import history_manager
    monkeypatch.undo()   # restore the real get_history_file_path (autouse patched it to a tmp file)
    monkeypatch.setenv("DOIT_PPID", "sess777")
    monkeypatch.setenv("DOIT_CMD_LOG", "/somewhere/else/cmdlog_999.tsv")   # deliberately elsewhere
    fp = history_manager.get_history_file_path()
    try:
        assert fp.name == "doit.jsonl"
        assert fp.parent.name == "history_sess777"   # from DOIT_PPID, NOT dirname(DOIT_CMD_LOG)
        assert fp.parent.parent.name == ".doit"
    finally:
        try:
            fp.parent.rmdir()
        except OSError:
            pass


def test_history_pid_folder_falls_back_to_getppid(monkeypatch):
    """With nothing exporting DOIT_PPID (bare run), the folder is still PID-named via os.getppid() -
    so it is never a bare shared `.doit/doit.jsonl`."""
    from llm_communicator import history_manager
    monkeypatch.undo()
    monkeypatch.delenv("DOIT_PPID", raising=False)
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    fp = history_manager.get_history_file_path()
    try:
        assert fp.name == "doit.jsonl"
        assert fp.parent.name == f"history_{__import__('os').getppid()}"
        assert fp.parent.parent.name == ".doit"
    finally:
        try:
            fp.parent.rmdir()
        except OSError:
            pass


# =====================================================================
# SECTION: Output awareness via re-run (no output capture)
# =====================================================================
def test_system_prompt_output_awareness_rerun():
    """Rule 11 tells the agent to RE-RUN an uncaptured (user) command to answer output questions."""
    from fixtures import DOIT_SYSTEM_PROMPT
    lowered = DOIT_SYSTEM_PROMPT.lower()
    assert "output awareness" in lowered
    assert "re-run" in lowered
    assert "exit 0" in lowered
    assert "read-only" in lowered


def test_system_prompt_attribution_defaults():
    """Rule 10 documents whose command an unqualified reference defaults to."""
    from fixtures import DOIT_SYSTEM_PROMPT
    lowered = DOIT_SYSTEM_PROMPT.lower()
    assert "most recent command" in lowered
    assert "you/we" in lowered     # you/we -> doit
    assert "i just did" in lowered  # I -> user


@pytest.mark.parametrize("text,subject", [
    ("what did we just do", "doit"),
    ("what did you just do", "doit"),
    ("what did I just do", "user"),
])
def test_is_activity_query_we_maps_to_doit(text, subject):
    from llm_communicator.tools import is_activity_query
    assert is_activity_query(text) == subject


@patch("llm_communicator.llm_bash.litellm.completion")
@pytest.mark.parametrize("question", [
    "which of these is safe to delete?",
    "why did that fail?",
    "what was the biggest one?",
])
def test_output_question_triggers_resolution(mock_completion, question):
    """An output question about a previous command trips the context heuristic so the resolver runs
    and links the relevant turn (rather than short-circuiting to [])."""
    mock_completion.return_value = MockResponse(MockMessage(content='{"relevant_ids": [1]}'))
    agent = BashToolAgent(api_key="fake-key")
    metadata = [{"id": 1, "source": "user", "prompt": "", "command": "ls -lhS", "suggested_command": ""}]
    ids = agent._analyze_references(question, metadata)
    mock_completion.assert_called_once()
    assert ids == [1]


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_output_question_reruns_user_command(mock_execute_bash, mock_completion):
    """After a successful user command (output NOT captured), an output question replays the user's
    command into context and the agent answers by RE-RUNNING / piping it (here the model builds on
    `ls -lhS`)."""
    from llm_communicator import history_manager
    history_manager.append_history_turn(
        "", "ls -lhS", BashToolAgent._user_cmd_output("0"), source="user", hist_n=1,
    )

    msg_analyze = MockMessage(content='{"relevant_ids": [1]}')
    tool_call = MockToolCall("d1", "execute_bash_command",
                             {"command": "ls -lhS | head -n 1", "explanation": "biggest"})
    msg_filter = MockMessage(content="DECISION: NO")   # read-only -> no [y/N]
    mock_completion.side_effect = [
        MockResponse(msg_analyze),
        MockResponse(MockMessage(tool_calls=[tool_call])),
        MockResponse(msg_filter),
    ]
    mock_execute_bash.return_value = "big.bin"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("which of these is biggest?")

    # The user's command was replayed so the model could re-run it...
    assert any("ls -lhS" in (m.get("content") or "") for m in agent.conversation_history)
    # ...and the agent re-ran/built on it to answer (no output was ever captured).
    mock_execute_bash.assert_called_once_with("ls -lhS | head -n 1")


# =====================================================================
# SECTION: Multi-window / cross-session referencing
# =====================================================================
def _seed_session(h, pid, cwd, turns, last_active="2026-06-30T00:00:00+00:00"):
    """Create a session folder with its doit.jsonl turns + session.json registry entry."""
    d = h.get_session_dir(pid)
    with open(d / "doit.jsonl", "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    (d / "session.json").write_text(json.dumps({
        "pid": pid, "cwd": cwd,
        "created_at": "2026-01-01T00:00:00+00:00", "last_active_at": last_active,
    }), encoding="utf-8")
    return d


def _turn(tid, prompt, command, output="", source="doit"):
    return {"id": tid, "source": source, "hist_n": None, "prompt": prompt, "command": command,
            "suggested_command": "", "output": output, "relevant_ids": []}


@pytest.mark.parametrize("text,expected", [
    ("list my sessions", True),
    ("list the shell numbers", True),
    ("what windows do I have open", True),
    ("show open terminals", True),
    ("what are the current sessions", True),       # "are" BEFORE the noun
    ("what are my current sessions", True),
    ("what are the other sessions", True),
    ("how many sessions are there", True),
    ("other sessions", True),
    ("list the files", False),
    ("sort them by date", False),
    ("do the folder task in the other window", False),   # singular reference, NOT a list
    ("the other terminal", False),
])
def test_is_session_list_query(text, expected):
    from llm_communicator.tools import is_session_list_query
    assert is_session_list_query(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("do the folder task we did in the other window here", True),
    ("the other terminal", True),
    ("do the task from session 12345 here", True),
    ("in window 2", True),
    ("of the files listed by the 130909 session, how many are executable", True),  # number-then-keyword
    ("redo what the 4242 window did here", True),
    ("sort them by date", False),
    ("delete that file", False),
    ("create a folder for each year from 2020 to 2026", False),
])
def test_is_cross_session_reference(text, expected):
    from llm_communicator.tools import is_cross_session_reference
    assert is_cross_session_reference(text) is expected


@pytest.mark.parametrize("text,pid", [
    ("do the task from session 12345 here", "12345"),
    ("of the files listed by the 130909 session, how many are executable", "130909"),  # number-then-keyword
    ("redo what the 4242 window did", "4242"),
    ("window 2", "2"),
    ("the other terminal", None),
    ("create folders 2020 to 2026", None),
])
def test_extract_session_pid(text, pid):
    from llm_communicator.tools import extract_session_pid
    assert extract_session_pid(text) == pid


def test_system_prompt_documents_multi_window():
    from fixtures import DOIT_SYSTEM_PROMPT
    low = DOIT_SYSTEM_PROMPT.lower()
    assert "multi-window" in low or "cross-session" in low
    assert "other window" in low or "other terminal" in low
    assert "re-ground" in low


def test_session_registry_and_listing(monkeypatch, tmp_path):
    """Each session is discoverable by PID + cwd; the current one is flagged; others exclude self."""
    from llm_communicator import history_manager as h
    monkeypatch.undo()                                    # use the real path functions
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    _seed_session(h, "111", "/home/u/llm", [_turn(1, "list", "ls")])
    _seed_session(h, "222", "/home/u/docs", [_turn(1, "make years", "mkdir 2020 2021")])

    alls = h.list_sessions()
    assert {s["pid"] for s in alls} == {"111", "222"}
    current = [s for s in alls if s["is_current"]]
    assert len(current) == 1 and current[0]["pid"] == "111"
    others = h.list_other_sessions()
    assert {s["pid"] for s in others} == {"222"}
    assert others[0]["cwd"] == "/home/u/docs"


@patch("llm_communicator.llm_bash.litellm.completion")
def test_list_sessions_route_is_deterministic(mock_llm, monkeypatch, tmp_path):
    """'list the shell numbers' lists sessions by PID from the registry - no LLM, no command."""
    from llm_communicator import history_manager as h
    import io, contextlib
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    _seed_session(h, "111", "/home/u/llm", [_turn(1, "list", "ls")])
    _seed_session(h, "222", "/home/u/docs", [_turn(1, "years", "mkdir 2020")])

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.run_single("list the shell numbers")
    out = buf.getvalue()
    assert "111" in out and "222" in out
    mock_llm.assert_not_called()


@patch("llm_communicator.llm_bash.resolve_cross_session")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_cross_session_applies_other_window_task(mock_exec, mock_llm, mock_resolve, monkeypatch, tmp_path):
    """An explicit cross-window reference pulls the other session's task and applies it HERE."""
    from llm_communicator import history_manager as h
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    _seed_session(h, "111", "/home/u/llm", [])           # current window, empty history
    _seed_session(h, "222", "/home/u/docs",
                  [_turn(1, "create a folder for each year from 2020 to 2026",
                         "mkdir 2020 2021 2022 2023 2024 2025 2026", output="0 (SUCCESS)")])

    # resolver picks session 222 (patched directly: tools.litellm and llm_bash.litellm are the same
    # module object, so they cannot be mocked separately).
    mock_resolve.return_value = {"pid": "222", "relevant_ids": [1], "confident": True}
    # main agent emits the folder-creation command for the CURRENT dir, then the filter judges it.
    tool_call = MockToolCall("d1", "execute_bash_command",
                             {"command": "mkdir 2020 2021 2022 2023 2024 2025 2026", "explanation": "year folders"})
    mock_llm.side_effect = [MockResponse(MockMessage(tool_calls=[tool_call])),
                            MockResponse(MockMessage(content="DECISION: YES"))]
    mock_exec.return_value = "[Success]"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("do the same folder task we did in the other window here")

    convo = "\n".join(m.get("content") or "" for m in agent.conversation_history
                      if isinstance(m.get("content"), str))
    assert "pid 222" in convo and "mkdir 2020" in convo   # the other window's task was injected
    mock_resolve.assert_called_once()                      # resolver consulted
    mock_exec.assert_called_once_with("mkdir 2020 2021 2022 2023 2024 2025 2026")


@patch("llm_communicator.llm_bash.resolve_cross_session")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_cross_session_by_explicit_pid_skips_resolver(mock_exec, mock_llm, mock_resolve, monkeypatch, tmp_path):
    """Referencing a session by its exact PID resolves deterministically - no fuzzy resolver call."""
    from llm_communicator import history_manager as h
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    _seed_session(h, "111", "/home/u/llm", [])
    _seed_session(h, "222", "/home/u/docs",
                  [_turn(1, "create year folders", "mkdir 2020 2021", output="0 (SUCCESS)")])

    # pid-exact -> resolver NOT used; _analyze_references (llm) picks the turn, then gen + filter
    mock_llm.side_effect = [
        MockResponse(MockMessage(content='{"relevant_ids": [1]}')),   # _analyze_references over 222
        MockResponse(MockMessage(tool_calls=[MockToolCall("d1", "execute_bash_command",
                     {"command": "mkdir 2020 2021", "explanation": "years"})])),
        MockResponse(MockMessage(content="DECISION: YES")),
    ]
    mock_exec.return_value = "[Success]"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("do the folder task from session 222 here")

    mock_resolve.assert_not_called()                       # exact pid -> no fuzzy resolver
    mock_exec.assert_called_once_with("mkdir 2020 2021")


@patch("llm_communicator.llm_bash.resolve_cross_session")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_cross_session_query_number_then_keyword_pid(mock_exec, mock_llm, mock_resolve, monkeypatch, tmp_path):
    """Regression: 'of the files listed by the 130909 session ...' (number-BEFORE-keyword PID)
    resolves to that session and injects its listing - NOT a missing-context rejection."""
    from llm_communicator import history_manager as h
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "131548")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    _seed_session(h, "131548", "/mnt/c/Users/ayb19", [])           # current window, empty
    ls_out = "-rwxr-xr-x 1 u u 0 a.sh\n-rw-r--r-- 1 u u 0 b.txt\n"
    _seed_session(h, "130909", "/home/u/llm_ass3",
                  [_turn(1, "list the files", "ls -la", output=ls_out)])

    # pid-exact (130909) -> no fuzzy resolver; _analyze_references picks the ls turn, then the agent
    # answers from the injected listing (answer_question, no command run here).
    mock_llm.side_effect = [
        MockResponse(MockMessage(content='{"relevant_ids": [1]}')),
        MockResponse(MockMessage(tool_calls=[MockToolCall("a1", "answer_question",
                     {"explanation": "1 of them is executable (a.sh).", "suggested_command": ""})])),
    ]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("of the files listed by the 130909 session, how many of them are executable")

    # Exclude the system message (the rejection phrase lives verbatim in Rule 7 of the prompt).
    convo = "\n".join(m.get("content") or "" for m in agent.conversation_history
                      if isinstance(m.get("content"), str) and m.get("role") != "system")
    assert "pid 130909" in convo and "a.sh" in convo          # the other session's listing was injected
    assert "I do not see any previous command" not in convo   # NOT the missing-context rejection
    mock_resolve.assert_not_called()                          # exact pid
    mock_exec.assert_not_called()                             # answered from the listing, not re-run here


@patch("llm_communicator.llm_bash.litellm.completion")
def test_cd_hoist_updates_session_registry_cwd(mock_llm, monkeypatch, tmp_path):
    """A `cd` is applied by the parent shell only AFTER doit exits, so the registry cwd (recorded at
    the top of the turn) would be stale. The cd-hoist updates session.json to the target immediately,
    so OTHER windows see this session's new location."""
    from llm_communicator import history_manager as h
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    target = tmp_path / "proj"
    target.mkdir()
    monkeypatch.setenv("DOIT_CD_FILE", str(tmp_path / "cdfile"))   # shell integration active

    tool_call = MockToolCall("d1", "execute_bash_command", {"command": f"cd {target}", "explanation": "go"})
    mock_llm.side_effect = [MockResponse(MockMessage(tool_calls=[tool_call])),
                            MockResponse(MockMessage(content="DECISION: NO"))]   # cd doesn't modify fs

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("go to proj")

    meta = h.read_session_meta(h.get_session_dir("111"))
    assert meta.get("cwd") == str(target)                         # registry reflects the NEW dir
    assert (tmp_path / "cdfile").read_text() == str(target)       # cd hoisted to the shell too


@patch("llm_communicator.llm_bash.resolve_cross_session")
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_cross_session_activity_reports_other_window(mock_exec, mock_llm, mock_resolve, monkeypatch, tmp_path):
    """'what did I do in the other window' reports THAT session's activity (not the current one),
    deterministically (no command run, no main LLM)."""
    from llm_communicator import history_manager as h
    import io, contextlib
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    _seed_session(h, "111", "/home/u/llm", [_turn(1, "list", "ls -la")])      # current window
    _seed_session(h, "222", "/home/u/docs", [_turn(1, "make years", "mkdir 2020 2021")])
    mock_resolve.return_value = {"pid": "222", "relevant_ids": [], "confident": True}

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.run_single("what did I do in the other window")
    out = buf.getvalue()

    assert "222" in out and "mkdir 2020 2021" in out      # reports the OTHER window
    assert "ls -la" not in out                            # NOT the current window's command
    mock_exec.assert_not_called()                         # report only - nothing run
    mock_llm.assert_not_called()                          # deterministic (resolver is patched)


@patch("llm_communicator.llm_bash.litellm.completion")
def test_output_question_includes_previous_output_when_resolver_links_nothing(mock_llm, monkeypatch, tmp_path):
    """Output-awareness safety net: a follow-up about previous output ALWAYS gets the most recent
    command-with-real-output in context, even when the reference resolver links nothing (e.g. it got
    confused by output-less user-command noise)."""
    from llm_communicator import history_manager as h
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    seed = [
        _turn(1, "", "ls", source="user"),            # user noise: no real output
        _turn(2, "", "git status", source="user"),
        _turn(3, "", "ls", source="user"),
        _turn(4, "list largest", "ls -laS",
              output="--- STDOUT ---\nbig.iso\nrun.sh\n--- RETURN CODE ---\n0 (SUCCESS)"),  # doit, real output
    ]
    d = h.get_session_dir("111")
    (d / "doit.jsonl").write_text("".join(json.dumps(t) + "\n" for t in seed), encoding="utf-8")
    (d / "session.json").write_text(json.dumps(
        {"pid": "111", "cwd": "/home/u", "created_at": "x", "last_active_at": "y"}), encoding="utf-8")

    mock_llm.side_effect = [
        MockResponse(MockMessage(content='{"relevant_ids": []}')),   # resolver links NOTHING
        MockResponse(MockMessage(tool_calls=[MockToolCall("a1", "answer_question",
                     {"explanation": "big.iso looks safe to delete.", "suggested_command": ""})])),
    ]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    agent.run_single("which of these looks safe to delete?")

    convo = "\n".join(m.get("content") or "" for m in agent.conversation_history
                      if isinstance(m.get("content"), str))
    assert "big.iso" in convo     # the previous doit output is in context despite the resolver linking nothing


@patch("llm_communicator.llm_bash.resolve_cross_session")
@patch("llm_communicator.llm_bash.litellm.completion")
def test_cross_session_activity_query_in_fallback_mode(mock_llm, mock_resolve, monkeypatch, tmp_path):
    """In NON-tool-calling mode, 'what command was recently run in session 222' must report session
    222 (deterministically) - NOT be hijacked by the how-to route's `^what command` pattern."""
    from llm_communicator import history_manager as h
    import io, contextlib
    monkeypatch.undo()
    monkeypatch.setattr(h, "doit_root", lambda: tmp_path / ".doit")
    monkeypatch.setenv("DOIT_PPID", "111")
    monkeypatch.delenv("DOIT_CMD_LOG", raising=False)
    monkeypatch.delenv("DOIT_SHELL_HISTORY", raising=False)
    _seed_session(h, "111", "/home/u/llm", [])
    _seed_session(h, "222", "/mnt/c", [_turn(1, "make years", "mkdir 2020 2021")])

    with patch("llm_communicator.llm_bash.load_config", return_value=("ollama/gemma3:4b", None, False)):
        agent = BashToolAgent()
    assert agent.tool_calling is False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.run_single("what command was recently run in session 222")
    out = buf.getvalue()

    assert "222" in out and "mkdir 2020 2021" in out   # reported the target session's command
    mock_llm.assert_not_called()                        # deterministic report - no main LLM
    mock_resolve.assert_not_called()                    # exact pid - no fuzzy resolver


# =====================================================================
# SECTION: Multi-step command plans (execute_plan)
# =====================================================================
def test_plan_tool_is_offered_in_tool_calling():
    agent = BashToolAgent(api_key="fake-key")
    assert "execute_plan" in [t["function"]["name"] for t in agent._build_tools(include_clarification=True)]
    assert "execute_plan" in [t["function"]["name"] for t in agent._build_tools(include_clarification=False)]


def test_system_prompt_documents_plan_rule():
    from fixtures import DOIT_SYSTEM_PROMPT
    low = DOIT_SYSTEM_PROMPT.lower()
    assert "execute_plan" in DOIT_SYSTEM_PROMPT
    assert "multi-step" in low
    assert "in sequence" in low
    assert "stop if" in low


_PLAN_STEPS = {"overview": "scaffold a project", "steps": [
    {"command": "mkdir proj", "explanation": "create the project dir"},
    {"command": "touch proj/main.py", "explanation": "add an entry file"},
    {"command": "git -C proj init", "explanation": "initialize git"},
]}


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_execute_plan_runs_steps_in_order(mock_exec, mock_llm, monkeypatch):
    """A multi-step plan runs each step IN ORDER (after one confirmation) and records the transcript."""
    from llm_communicator import history_manager
    monkeypatch.setenv("DOIT_PPID", "ptest")
    mock_llm.side_effect = [MockResponse(MockMessage(tool_calls=[MockToolCall("p1", "execute_plan", _PLAN_STEPS)]))]
    mock_exec.return_value = "[Success]"

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("set up a python project")

    assert [c.args[0] for c in mock_exec.call_args_list] == ["mkdir proj", "touch proj/main.py", "git -C proj init"]
    turn = history_manager.get_history_metadata()[-1]
    assert turn["command"] == "mkdir proj; touch proj/main.py; git -C proj init"
    last_out = history_manager.get_full_turns([turn["id"]])[0]["output"]
    assert "all 3 steps succeeded" in last_out


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_execute_plan_stops_on_failure(mock_exec, mock_llm, monkeypatch):
    """If a step fails, the plan STOPS - later steps are not run (no cascade)."""
    from llm_communicator import history_manager
    monkeypatch.setenv("DOIT_PPID", "ptest2")
    mock_llm.side_effect = [MockResponse(MockMessage(tool_calls=[MockToolCall("p1", "execute_plan", _PLAN_STEPS)]))]
    mock_exec.side_effect = ["[Success]", "--- RETURN CODE ---\n1 (FAILED)\n"]   # step 2 fails

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("set up a python project")

    assert [c.args[0] for c in mock_exec.call_args_list] == ["mkdir proj", "touch proj/main.py"]   # step 3 skipped
    out = history_manager.get_full_turns([history_manager.get_history_metadata()[-1]["id"]])[0]["output"]
    assert "STOPPED at step 2/3" in out


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_execute_plan_declined_runs_nothing(mock_exec, mock_llm, monkeypatch):
    """Declining the plan up-front runs no steps at all."""
    monkeypatch.setenv("DOIT_PPID", "ptest3")
    mock_llm.side_effect = [MockResponse(MockMessage(tool_calls=[MockToolCall("p1", "execute_plan", _PLAN_STEPS)]))]

    agent = BashToolAgent(api_key="fake-key")
    agent.tool_calling = True
    with patch("builtins.input", return_value="n"):
        agent.run_single("set up a python project")

    mock_exec.assert_not_called()


# =====================================================================
# SECTION: Command hoisting inside plans (cd / session-state)
# =====================================================================
@pytest.mark.parametrize("cmd,base,expected", [
    ("cd sub", "/a", "/a/sub"),
    ("cd ..", "/a/b", "/a"),
    ("cd /x/y", "/a", "/x/y"),
    ("cd 'my dir'", "/a", "/a/my dir"),
    ("cd proj && ls", "/a", None),     # compound -> not a standalone cd
    ("cd -", "/a", None),
    ("ls", "/a", None),
])
def test_resolve_cd_target(cmd, base, expected):
    from llm_communicator.tools import resolve_cd_target
    assert resolve_cd_target(cmd, base) == expected


def _plan_call(plan):
    return MockResponse(MockMessage(tool_calls=[MockToolCall("p1", "execute_plan", plan)]))


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_plan_threads_cwd_across_cd(mock_exec, mock_llm, monkeypatch, tmp_path):
    """A `cd` step moves the plan's cwd for LATER steps; the cd itself is not subprocessed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proj").mkdir()                       # so `cd proj` validates (mkdir is mocked)
    plan = {"steps": [
        {"command": "mkdir proj", "explanation": "dir"},
        {"command": "cd proj", "explanation": "enter"},
        {"command": "touch main.py", "explanation": "file"},
    ]}
    mock_llm.side_effect = [_plan_call(plan)]
    mock_exec.return_value = "[Success]"
    agent = BashToolAgent(api_key="fake-key"); agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("scaffold")

    by_cmd = {c.args[0]: c.kwargs for c in mock_exec.call_args_list}
    assert set(by_cmd) == {"mkdir proj", "touch main.py"}        # cd tracked, not executed
    assert by_cmd["mkdir proj"]["cwd"] == str(tmp_path)          # before the cd
    assert by_cmd["touch main.py"]["cwd"] == str(tmp_path / "proj")  # after the cd


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_plan_bad_cd_stops(mock_exec, mock_llm, monkeypatch, tmp_path):
    """A `cd` into a missing directory is a failed step that stops the plan."""
    from llm_communicator import history_manager
    monkeypatch.chdir(tmp_path)
    plan = {"steps": [
        {"command": "cd nope", "explanation": "enter missing"},
        {"command": "touch a", "explanation": "file"},
    ]}
    mock_llm.side_effect = [_plan_call(plan)]
    agent = BashToolAgent(api_key="fake-key"); agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("scaffold")

    mock_exec.assert_not_called()                                # nothing ran
    out = history_manager.get_full_turns([history_manager.get_history_metadata()[-1]["id"]])[0]["output"]
    assert "no such directory" in out and "STOPPED at step 1/2" in out


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_plan_compound_cd_runs_as_normal_step(mock_exec, mock_llm, monkeypatch, tmp_path):
    """`cd x && y` is NOT a standalone cd - it runs as one normal subprocess step."""
    monkeypatch.chdir(tmp_path)
    plan = {"steps": [{"command": "cd /tmp && ls", "explanation": "compound"}]}
    mock_llm.side_effect = [_plan_call(plan)]
    mock_exec.return_value = "[Success]"
    agent = BashToolAgent(api_key="fake-key"); agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("x")
    assert [c.args[0] for c in mock_exec.call_args_list] == ["cd /tmp && ls"]


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_plan_export_applies_as_prelude(mock_exec, mock_llm, monkeypatch, tmp_path):
    """An `export` step is tracked and applied to LATER steps via the prelude (not subprocessed alone)."""
    monkeypatch.chdir(tmp_path)
    plan = {"steps": [
        {"command": "export K=1", "explanation": "set var"},
        {"command": 'echo "$K"', "explanation": "use var"},
    ]}
    mock_llm.side_effect = [_plan_call(plan)]
    mock_exec.return_value = "[Success]"
    agent = BashToolAgent(api_key="fake-key"); agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("x")
    by_cmd = {c.args[0]: c.kwargs for c in mock_exec.call_args_list}
    assert set(by_cmd) == {'echo "$K"'}                          # export tracked, not executed
    assert by_cmd['echo "$K"']["prelude"] == ["export K=1"]


@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_plan_hoists_final_cwd_and_state(mock_exec, mock_llm, monkeypatch, tmp_path):
    """On success, the plan's net cwd + session-state are hoisted to the parent shell."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proj").mkdir()
    cdfile = tmp_path / "cdfile"; shfile = tmp_path / "shfile"
    monkeypatch.setenv("DOIT_CD_FILE", str(cdfile))
    monkeypatch.setenv("DOIT_SHELL_FILE", str(shfile))
    plan = {"steps": [
        {"command": "cd proj", "explanation": "enter"},
        {"command": "export K=1", "explanation": "set"},
        {"command": "true", "explanation": "noop"},
    ]}
    mock_llm.side_effect = [_plan_call(plan)]
    mock_exec.return_value = "[Success]"
    agent = BashToolAgent(api_key="fake-key"); agent.tool_calling = True
    with patch("builtins.input", return_value="y"):
        agent.run_single("x")

    assert cdfile.read_text() == str(tmp_path / "proj")          # net cwd hoisted
    assert shfile.read_text() == "export K=1"                    # session-state hoisted


# =====================================================================
# SECTION: Plan-preference guidance (user multi-step vs model chaining)
# =====================================================================
def test_rule13_states_intent_test():
    """Rule 13 distinguishes a user-requested action sequence (-> execute_plan) from chaining/piping
    to answer a question (-> a single execute_bash_command)."""
    from fixtures import DOIT_SYSTEM_PROMPT
    low = DOIT_SYSTEM_PROMPT.lower()
    assert "several distinct actions" in low                 # the "use it" criterion
    assert "answer a question" in low                        # the "don't use it" exclusion
    # existing Rule 13 contract still holds
    assert "execute_plan" in DOIT_SYSTEM_PROMPT
    assert "multi-step" in low and "in sequence" in low and "stop if" in low


def test_fewshot_toolcall_has_plan_and_piped_examples():
    """The tool-calling few-shots demonstrate BOTH a multi-action request -> execute_plan AND a piped
    question -> a single execute_bash_command (teaching the Rule 13 boundary)."""
    from llm_communicator.backup_system_prompts import FEWSHOT_TOOLCALL
    tool_calls = [tc for m in FEWSHOT_TOOLCALL if m.get("role") == "assistant"
                  for tc in m.get("tool_calls", [])]
    names = [tc["function"]["name"] for tc in tool_calls]
    assert "execute_plan" in names                           # multi-action example
    # a piped execute_bash_command example exists (a question answered by one piped command)
    piped = [tc for tc in tool_calls
             if tc["function"]["name"] == "execute_bash_command" and "|" in tc["function"]["arguments"]]
    assert piped, "expected a piped execute_bash_command few-shot"


# =====================================================================
# SECTION: Command plans in NON-tool-calling (fallback) mode
# =====================================================================
_FALLBACK_PLAN_JSON = ('{"executable": true, "rule_triggered": 13, "overview": "scaffold", '
                       '"steps": [{"command": "mkdir proj", "explanation": "dir"}, '
                       '{"command": "touch proj/main.py", "explanation": "file"}, '
                       '{"command": "git -C proj init", "explanation": "git"}], '
                       '"command": "", "response_text": ""}')


def test_fallback_instruction_documents_steps():
    from llm_communicator.backup_system_prompts import FALLBACK_SYSTEM_INSTRUCTION
    assert '"steps"' in FALLBACK_SYSTEM_INSTRUCTION
    assert "13 for a multi-step plan" in FALLBACK_SYSTEM_INSTRUCTION


def test_rule13_fallback_clause_uses_steps():
    from fixtures import DOIT_SYSTEM_PROMPT
    low = DOIT_SYSTEM_PROMPT.lower()
    assert "non-tool-calling mode" in low and "`steps`" in DOIT_SYSTEM_PROMPT


def test_fewshot_fallback_has_steps_example():
    from llm_communicator.backup_system_prompts import FEWSHOT_FALLBACK
    assert any('"steps"' in m.get("content", "") for m in FEWSHOT_FALLBACK if m.get("role") == "assistant")


@patch("llm_communicator.llm_bash.load_config", return_value=("ollama/qwen3:4b-instruct", None, False))
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_fallback_plan_runs_in_order(mock_exec, mock_llm, mock_cfg, monkeypatch):
    """A fallback-JSON `steps` array runs through the same plan runner, in order."""
    from llm_communicator import history_manager
    monkeypatch.setenv("DOIT_PPID", "fbplan1")
    mock_llm.side_effect = [MockResponse(MockMessage(content=_FALLBACK_PLAN_JSON))]
    mock_exec.return_value = "[Success]"

    agent = BashToolAgent()
    assert agent.tool_calling is False
    with patch("builtins.input", return_value="y"):
        agent.run_single("set up a python project")

    assert [c.args[0] for c in mock_exec.call_args_list] == ["mkdir proj", "touch proj/main.py", "git -C proj init"]
    turn = history_manager.get_history_metadata()[-1]
    assert turn["command"] == "mkdir proj; touch proj/main.py; git -C proj init"
    assert "all 3 steps succeeded" in history_manager.get_full_turns([turn["id"]])[0]["output"]


@patch("llm_communicator.llm_bash.load_config", return_value=("ollama/qwen3:4b-instruct", None, False))
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_fallback_plan_stops_on_failure(mock_exec, mock_llm, mock_cfg, monkeypatch):
    from llm_communicator import history_manager
    monkeypatch.setenv("DOIT_PPID", "fbplan2")
    mock_llm.side_effect = [MockResponse(MockMessage(content=_FALLBACK_PLAN_JSON))]
    mock_exec.side_effect = ["[Success]", "--- RETURN CODE ---\n1 (FAILED)\n"]   # step 2 fails

    agent = BashToolAgent()
    with patch("builtins.input", return_value="y"):
        agent.run_single("set up a python project")

    assert [c.args[0] for c in mock_exec.call_args_list] == ["mkdir proj", "touch proj/main.py"]   # step 3 skipped
    out = history_manager.get_full_turns([history_manager.get_history_metadata()[-1]["id"]])[0]["output"]
    assert "STOPPED at step 2/3" in out


@patch("llm_communicator.llm_bash.load_config", return_value=("ollama/qwen3:4b-instruct", None, False))
@patch("llm_communicator.llm_bash.litellm.completion")
@patch("llm_communicator.llm_bash.execute_bash")
def test_fallback_plan_declined_runs_nothing(mock_exec, mock_llm, mock_cfg, monkeypatch):
    monkeypatch.setenv("DOIT_PPID", "fbplan3")
    mock_llm.side_effect = [MockResponse(MockMessage(content=_FALLBACK_PLAN_JSON))]
    agent = BashToolAgent()
    with patch("builtins.input", return_value="n"):
        agent.run_single("set up a python project")
    mock_exec.assert_not_called()
