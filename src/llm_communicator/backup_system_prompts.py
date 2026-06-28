"""
Backup / fallback system prompts for the doit agent.

These are appended to the main system prompt only when the configured model lacks native
tool calling (``tool_calling = False``). The model is then forced to emit a raw JSON block
instead of using tool calls.
"""

FALLBACK_SYSTEM_INSTRUCTION = """IMPORTANT: You do not support native tool calling in this environment. Instead, you MUST respond with a raw JSON block containing exactly the following keys:
- "executable": (boolean) true if a bash command is generated to execute, false if not.
- "command": (string) the single-line or multi-line bash script to execute (only when executable is true, otherwise empty "").
- "explanation": (string) a short explanation of what the command does, or why it is not executable.
- "rule_triggered": (integer) the number of the system instruction rule triggered (1 for command generation, 2 for impossible, 3 for safety violation, 4 for irrelevant input, 5 for capability inquiry, 6 for assume file existence, 8 for clarification request).
- "response_text": (string) the conversational text, warning, or error message to display directly to the user (required when executable is false, e.g., "bash: command not found" for rule 2, or "The command is not safe to execute. <reason>" for rule 3).
- "needs_clarification": (boolean) set to true ONLY when the request is genuinely ambiguous and you need to ask the user before generating a command (Rule 8). When true, also set "executable": false, "command": "", "rule_triggered": 8. You do NOT need to write the question - a separate step authors it.

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

Example response for capability inquiry (Rule 5):
{
  "executable": false,
  "command": "",
  "explanation": "The user is asking about my capabilities",
  "rule_triggered": 5,
  "response_text": "My purpose is to translate natural language descriptions into executable bash commands."
}

Example response for assuming file existence (Rule 6):
{
  "executable": true,
  "command": "cat /etc/passwd",
  "explanation": "Attempting to read the /etc/passwd file",
  "rule_triggered": 6,
  "response_text": ""
}

Example response for an ambiguous bash request that needs clarification (Rule 8). A request to list/sort/find/manipulate files is ALWAYS a bash task - NEVER irrelevant (Rule 4) and you must NOT just guess a command. Sorting or filtering "by date" is the canonical ambiguous case (creation vs. access vs. modification time), so set needs_clarification - even for a follow-up like "sort them by date". You do NOT write the question; a separate step authors it:
{
  "executable": false,
  "command": "",
  "explanation": "Sorting by date is ambiguous: it could mean creation, access, or modification time",
  "rule_triggered": 8,
  "response_text": "",
  "needs_clarification": true
}
"""


# In-context few-shot demonstrations of the clarification DECISION, injected as real
# conversation turns right before the user's instruction (fallback/non-tool-calling mode).
# Weak local models follow concrete in-context examples far better than system-prompt rules.
# Balanced on purpose: ASK, don't-ask, ASK - so the model learns the boundary, not "always ask".
# Few-shot demonstrations of the clarification decision, injected as real conversation turns
# right before the user's instruction. Both tool-calling and fallback models need these:
# without them, models tend to EXPLAIN a command in prose instead of calling a tool / emitting
# the JSON. For AMBIGUOUS examples we show ONLY the ask step (no resolved command) - the model
# copies the assistant's action for a matching input, so ask-only => copying == asking.
FEWSHOT_TOOLCALL = [
    # Clear request -> call execute_bash_command directly (do NOT ask).
    {"role": "user", "content": "create a file called notes.txt"},
    {"role": "assistant", "tool_calls": [{"id": "fs_exec_1", "type": "function",
        "function": {"name": "execute_bash_command", "arguments": '{"command": "touch notes.txt", "explanation": "create an empty file notes.txt"}'}}]},
    {"role": "tool", "tool_call_id": "fs_exec_1", "name": "execute_bash_command", "content": "[Success]"},
    # Ambiguous (which date?) -> call ask_user_clarification (no command).
    {"role": "user", "content": "list the files sorted by date"},
    {"role": "assistant", "tool_calls": [{"id": "fs_clar_2", "type": "function",
        "function": {"name": "ask_user_clarification", "arguments": '{"reason": "by date: creation, access, or modification time?"}'}}]},
    {"role": "tool", "tool_call_id": "fs_clar_2", "name": "ask_user_clarification", "content": "modification time"},
    # Ambiguous (sort key unspecified) -> call ask_user_clarification (no command). LAST = recency.
    {"role": "user", "content": "list files in the cwd and sort them"},
    {"role": "assistant", "tool_calls": [{"id": "fs_clar_1", "type": "function",
        "function": {"name": "ask_user_clarification", "arguments": '{"reason": "sort key unspecified: name, size, or date"}'}}]},
    {"role": "tool", "tool_call_id": "fs_clar_1", "name": "ask_user_clarification", "content": "by name"},
]


FEWSHOT_FALLBACK = [
    {"role": "user", "content": "count how many files are here"},
    {"role": "assistant", "content": '{"executable": true, "command": "ls -1 | wc -l", "explanation": "count files in the current directory", "rule_triggered": 1, "response_text": ""}'},
    {"role": "user", "content": "create a file called notes.txt"},
    {"role": "assistant", "content": '{"executable": true, "command": "touch notes.txt", "explanation": "create an empty file notes.txt", "rule_triggered": 1, "response_text": ""}'},
    {"role": "user", "content": "list the files sorted by date"},
    {"role": "assistant", "content": '{"executable": false, "command": "", "explanation": "Sorting by date is ambiguous: creation, access, or modification time", "rule_triggered": 8, "response_text": "", "needs_clarification": true}'},
    # LAST = recency: the bare-sort ambiguous case.
    {"role": "user", "content": "list files in the cwd and sort them"},
    {"role": "assistant", "content": '{"executable": false, "command": "", "explanation": "Sort key unspecified: by name, size, or date?", "rule_triggered": 8, "response_text": "", "needs_clarification": true}'},
]
