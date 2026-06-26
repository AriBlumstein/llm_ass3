# Multi-Turn Stage 1 Doit Agent ACDL Documentation

This directory contains the Agentic Context Description Language (ACDL) specifications for the **Multi-Turn Stage 1 Doit Agent**. It maps the structure of the LLM prompt contexts, the reference resolution dynamics, and execution flows for both tool-supported and non-tool-supported environments at step-by-step time intervals (`t`).

---

## ACDL Specifications

* **[Tool-Use Scenario](./tool_use.acdl)**: Describes the interaction and reference structure when the underlying LLM natively supports tool calling (function calling).
* **[Non-Tool-Use Scenario](./non_tool_use.acdl)**: Describes the fallback context structure when native tool calling is disabled and the agent relies on structured fallback JSON blocks.

---

## Agent Logic Overview

The entrypoint for the multi-turn execution is the [BashToolAgent.run_single](../../src/llm_communicator/llm_bash.py#L457) method. In this stage, the agent dynamically reconstructs conversation history based on logical dependency chains resolved by the classifier LLM.

### 1. Tool-Use Scenario Flow

```mermaid
graph TD
    Start([Start Turn t]) --> GetHistory[1. Retrieve history metadata]
    GetHistory --> FilterHistory[2. Filter out empty command turns]
    FilterHistory --> RunResolver[3. Call History Resolver with reverse-chronological metadata]
    RunResolver --> TransitiveDeps[4. Resolve transitive dependencies recursively]
    TransitiveDeps --> GetTurns[5. Load full turns for resolved IDs]
    GetTurns --> PopulateHistory[6. Reconstruct Tool-Use history context]
    PopulateHistory --> MainCall[7. Primary LLM Call with execute_bash_command tool]
    MainCall --> HasToolCall{Does it call execute_bash_command?}
    
    HasToolCall -- No --> DirectText[Print plain text response directly to user] --> End([End])
    
    HasToolCall -- Yes --> ParseArgs[Parse command & explanation arguments]
    ParseArgs --> FilterCall[8. Secondary LLM Call using DOIT_FILTER_PROMPT]
    FilterCall --> ModifiesCheck{Does command modify filesystem?}
    
    ModifiesCheck -- Yes --> PromptUser[Prompt User for confirmation: [y/N]]
    PromptUser --> ApprovedCheck{Did user approve?}
    ApprovedCheck -- No --> Cancelled[Set result to Cancelled message] --> AppendTool[Append tool role result to history]
    ApprovedCheck -- Yes --> RunBash
    
    ModifiesCheck -- No --> RunBash[Run execute_bash]
    RunBash --> RegexCheck{Fails Python pattern blacklist?}
    RegexCheck -- Yes --> SafetyError[Set result to safety error message] --> AppendTool
    RegexCheck -- No --> Execute[Execute command in bash subshell] --> AppendTool
    
    AppendTool --> LogTurn[9. Log turn details including stdout & return code to session history] --> End
```

### 2. Non-Tool-Use (Fallback JSON) Scenario Flow

```mermaid
graph TD
    Start([Start Turn t]) --> GetHistory[1. Retrieve history metadata]
    GetHistory --> FilterHistory[2. Filter out empty command turns]
    FilterHistory --> RunResolver[3. Call History Resolver with reverse-chronological metadata]
    RunResolver --> TransitiveDeps[4. Resolve transitive dependencies recursively]
    TransitiveDeps --> GetTurns[5. Load full turns for resolved IDs]
    GetTurns --> PopulateHistory[6. Reconstruct Fallback JSON history context]
    PopulateHistory --> MainCall[7. Primary LLM Call with Fallback JSON instructions]
    MainCall --> ParseJSON[Parse Fallback JSON Response]
    ParseJSON --> ExecutableCheck{Is 'executable' true?}
    
    ExecutableCheck -- No --> PrintDirect[Print 'response_text' directly to user] --> End([End])
    
    ExecutableCheck -- Yes --> FilterCall[8. Secondary LLM Call using DOIT_FILTER_PROMPT]
    FilterCall --> ModifiesCheck{Does command modify filesystem?}
    
    ModifiesCheck -- Yes --> PromptUser[Prompt User for confirmation: [y/N]]
    PromptUser --> ApprovedCheck{Did user approve?}
    ApprovedCheck -- No --> Cancelled[Set result to Cancelled message] --> AppendUser[Append execution output as user role message]
    ApprovedCheck -- Yes --> RunBash
    
    ModifiesCheck -- No --> RunBash[Run execute_bash]
    RunBash --> RegexCheck{Fails Python pattern blacklist?}
    RegexCheck -- Yes --> SafetyError[Set result to safety error message] --> AppendUser
    RegexCheck -- No --> Execute[Execute command in bash subshell] --> AppendUser
    
    AppendUser --> LogTurn[9. Log turn details including stdout & return code to session history] --> End
```

---

## Detailed Component Analysis

### The History Reference Resolver Call
* **Heuristic Bypass**: The query is first checked against context indicators. If no indicator (e.g. `"them"`, `"it"`, `"we created"`, `"we listed"`) is found, reference resolution is skipped and the history is treated as empty (`[]`).
* **Metadata Filtering**: Turns where `"command"` is empty `""` (such as previous warning rejections or text-only responses) are stripped. The resolver only sees turns that executed actual terminal commands.
* **Reverse Chronological Context**: The filtered history metadata list is reversed (most recent first) to prompt the LLM. This causes the resolver to naturally prioritize and match the most recent successful commands.
* **Transitive Dependency Resolution**: If the resolver matches a turn, the agent recursively checks all transitively chained dependencies recorded in the history database to load the entire logic path.

### The Primary LLM Call
* **History Re-construction**: 
  * Only full records of resolved, transitively connected turns are formatted and injected as conversation context.
  * In the Tool-Use scenario, each turn is represented as a alternating sequence of user message (`U`), assistant tool call (`A`), and tool output (`T`).
  * In the Non-Tool-Use scenario, the fallback JSON is used, and the stdout execution response is fed back inside a subsequent user message (`U`).
* **Override Warning Logic**: Under CASE 1 and CASE 2, if context is missing, the prompt instructs the model to return a structured JSON block with `executable: false` and the warning inside `response_text`.
