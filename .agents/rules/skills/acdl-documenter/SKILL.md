---
trigger: always_on
---

---

name: acdl-documenter
description: Teaches the agent how to write and review Agentic Context Description Language (ACDL) specifications according to the official syntax reference and examples.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Knowledge

## What ACDL Represents

ACDL (Agentic Context Description Language) describes the structure of LLM contexts and how they evolve across interaction steps. It models role-based messages, history accumulation, tool interactions, loops, conditions, and time-indexed references.

## Core Syntax Principles

### Prompt Definitions

An ACDL document is typically defined as a named prompt or context structure parameterized by time.

Example:

```acdl
MyPrompt[@T]: {
    S: {
        SYSTEM_INSTRUCTIONS
        TOOL_DESCRIPTIONS
    }

    U: user_input[@T]
}
```

### Role Identifiers

Use the standard role abbreviations:

```acdl
S:   // System
U:   // User
A:   // Assistant
T:   // Tool
```

Role entries may contain:

* Labels
* Variables
* References
* String literals
* Nested structures

### Time References

ACDL explicitly models context evolution using time indices.

Examples:

```acdl
user_input[@T]
assistant_response[@T-1]
tool_output[@t]
```

Where:

* `@T` = current interaction step
* `@t` = loop variable
* `[@n]` = indexed reference

### History Loops

Conversation history is commonly represented using `ForEach`.

Example:

```acdl
ForEach(t: range(1, @T)) {
    U: user_input[@t]
    A: assistant_response[@t]
}
```

### Role Blocks

A role may contain a single item:

```acdl
U: user_input[@T]
```

or multiple items:

```acdl
S: {
    CORE_INSTRUCTIONS
    SAFETY_RULES
    AVAILABLE_TOOLS
}
```

### Labels and Templates

Uppercase identifiers are typically used as reusable symbolic content blocks.

Example:

```acdl
S: {
    SYSTEM_PROMPT
    TOOL_GUIDELINES
}
```

These represent abstract content sources rather than literal text.

### Tool Interactions

Tool calls are represented explicitly.

Example:

```acdl
A: tool_request[@t]
T: tool_result[@t]
```

### Conditional Structures

Conditional inclusion may be represented with explicit control structures.

Example:

```acdl
If(HAS_TOOL_RESULT[@T]) {
    T: tool_result[@T]
}
```

# Instructions

When generating ACDL:

1. Use a named prompt definition rather than inventing a `specification {}` wrapper unless the user explicitly requests one.
2. Use role abbreviations (`S`, `U`, `A`, `T`) rather than verbose role names.
3. Represent evolving context using `@T` and indexed references.
4. Model conversation history with `ForEach`.
5. Prefer symbolic labels (`SYSTEM_PROMPT`, `USER_QUERY`, etc.) instead of embedding large literal strings.
6. Preserve chronological message ordering exactly as the context would be presented to the LLM.
7. Include tools and tool results as explicit role entries when applicable.

# Canonical Example

```acdl
AgentContext[@T]: {

    S: {
        SYSTEM_PROMPT
        TOOL_DESCRIPTIONS
    }

    ForEach(t: range(1, @T)) {

        U: user_input[@t]

        A: assistant_response[@t]

        If(tool_result_exists[@t]) {
            T: tool_result[@t]
        }
    }

    U: current_user_input[@T]
}
```
