# Single-Turn Doit Agent ACDL Documentation

This directory contains the Agentic Context Description Language (ACDL) specifications for the **Single-Turn Doit Agent**. It maps the structure of the LLM prompt contexts, the interaction dynamics, and execution flows for both tool-supported and non-tool-supported environments.

---

## ACDL Specifications

* **[Tool-Use Scenario](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/data/acdl/single_turn/tool_use.acdl)**: Describes the interaction structure when the underlying LLM natively supports tool calling (function calling).
* **[Non-Tool-Use Scenario](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/data/acdl/single_turn/non_tool_use.acdl)**: Describes the fallback context structure when native tool calling is disabled and the agent relies on structured fallback JSON blocks.

---

## Agent Logic Overview

The entrypoint for the single-turn execution is the [BashToolAgent.run_single](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L481-L593) method. The agent performs a primary LLM completion to translate a natural language query into a shell action, evaluates safety constraints, runs a secondary classifier LLM check for filesystem modification, prompts the user for confirmation if required, and captures execution results.

### 1. Tool-Use Scenario Flow

In environments with native tool support, the primary LLM is configured with [tools_definition](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L244-L258) representing the `execute_bash_command` function.

```mermaid
graph TD
    Start([Start Single Turn]) --> MainCall[1. Primary LLM Call with execute_bash_command tool]
    MainCall --> HasToolCall{Does it call execute_bash_command?}
    
    HasToolCall -- No --> DirectText[Print plain text response directly to user] --> End([End])
    
    HasToolCall -- Yes --> ParseArgs[Parse command & explanation arguments]
    ParseArgs --> FilterCall[2. Secondary LLM Call using DOIT_FILTER_PROMPT]
    FilterCall --> ModifiesCheck{Does command modify filesystem?}
    
    ModifiesCheck -- Yes --> PromptUser[Prompt User for confirmation: [y/N]]
    PromptUser --> ApprovedCheck{Did user approve?}
    ApprovedCheck -- No --> Cancelled[Set result to Cancelled message] --> AppendTool[Append tool role result to history]
    ApprovedCheck -- Yes --> RunBash
    
    ModifiesCheck -- No --> RunBash[Run execute_bash]
    RunBash --> RegexCheck{Fails Python pattern blacklist?}
    RegexCheck -- Yes --> SafetyError[Set result to safety error message] --> AppendTool
    RegexCheck -- No --> Execute[Execute command in bash subshell] --> AppendTool
    
    AppendTool --> PrintOutput[Print shell response] --> End
```

### 2. Non-Tool-Use (Fallback JSON) Scenario Flow

When tool-calling is disabled, the system instructions are appended with [FALLBACK_SYSTEM_INSTRUCTION](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L34-L105). The LLM is forced to respond with a raw JSON block containing `executable`, `command`, `explanation`, and `response_text`.

```mermaid
graph TD
    Start([Start Single Turn]) --> MainCall[1. Primary LLM Call with Fallback JSON instructions]
    MainCall --> ParseJSON[Parse Fallback JSON Response]
    ParseJSON --> ExecutableCheck{Is 'executable' true?}
    
    ExecutableCheck -- No --> PrintDirect[Print 'response_text' directly to user] --> End([End])
    
    ExecutableCheck -- Yes --> FilterCall[2. Secondary LLM Call using DOIT_FILTER_PROMPT]
    FilterCall --> ModifiesCheck{Does command modify filesystem?}
    
    ModifiesCheck -- Yes --> PromptUser[Prompt User for confirmation: [y/N]]
    PromptUser --> ApprovedCheck{Did user approve?}
    ApprovedCheck -- No --> Cancelled[Set result to Cancelled message] --> AppendUser[Append execution output as user role message]
    ApprovedCheck -- Yes --> RunBash
    
    ModifiesCheck -- No --> RunBash[Run execute_bash]
    RunBash --> RegexCheck{Fails Python pattern blacklist?}
    RegexCheck -- Yes --> SafetyError[Set result to safety error message] --> AppendUser
    RegexCheck -- No --> Execute[Execute command in bash subshell] --> AppendUser
    
    AppendUser --> PrintOutput[Print shell response] --> End
```

---

## Detailed Component Analysis

### The Primary LLM Call
* **Prompt context**: 
  * In the Tool-Use scenario, the context comprises [DOIT_SYSTEM_PROMPT](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/fixtures.py#L12-L64) as system (`S`), followed by the user instruction (`U`).
  * In the Non-Tool-Use scenario, the system (`S`) prompt is extended with the [FALLBACK_SYSTEM_INSTRUCTION](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L34-L105) formatting guide.
* **Role execution and mapping**:
  * Tool-Use maps the assistant's decision to a native `tool_calls` object (`A`) and records the execution outcome via a `tool` role message (`T`).
  * Non-Tool-Use receives the JSON block (`A`). Because non-tool models cannot handle standard `tool` messages, the bash execution result is appended as a subsequent user message (`U`), simulating output feedback.

### The Secondary LLM Check (Filesystem Filter)
The helper method [BashToolAgent._filter_bash](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L287-L331) runs a synchronous secondary LLM call to act as a safety judge.
* **Prompt instruction**: Uses [DOIT_FILTER_PROMPT](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/fixtures.py#L66-L87) as system (`S`), which specifies what qualifies as "modifying the file system" (e.g. `mkdir`, `touch`, `rm`, `git commit`, redirection `>`) vs. passive inspection (e.g. `ls`, `grep`, `pwd`, `git status`).
* **Input**: User (`U`) message template: `"Does the following command modify the file system? " + command`.
* **Output formats**: Parses the response looking for `DECISION: YES` (or `YES`, `TRUE`) and `EXPLANATION: <reason>` to determine whether to prompt the user for execution consent.

### Hardcoded Python Blacklist Filter
In addition to the LLM filter, the [execute_bash](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L189-L241) function enforces a Python regex-based safety check against [BANNED_COMMAND_PATTERNS](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L24-L32). If a pattern is matched, execution is blocked immediately, raising a [BashSafetyViolationError](file:///home/ayb19/projects/git-repos/LLM/llm_ass3/src/llm_communicator/llm_bash.py#L107-L109).
