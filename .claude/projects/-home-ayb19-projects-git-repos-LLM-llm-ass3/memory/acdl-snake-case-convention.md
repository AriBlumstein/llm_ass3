---
name: acdl-snake-case-convention
description: In this project's ACDL specs, prompt-definition (function) names AND label/variable names must be snake_case, not PascalCase/ALL_CAPS
metadata:
  type: feedback
---

When writing/editing ACDL (`data/**/*.acdl`) for the doit project, use **snake_case** for:
- prompt definition (context/"function") names: `doit_tool_use_agent[@T]`, `history_reference_resolver[@T]`,
  `bash_filter[@T]`, `clarification_author[@T.i]`, `memory_manager[@T]`, `explain_command[@T]`,
  `howto_answer_subcall[@T]` — NOT PascalCase. (This also makes them match the `resp.pid.<name>` references.)
- label / template-function / variable names: `user_ran_note(...)`, `answer_delivered_notice` — NOT ALL_CAPS.

**Why:** the user's convention. (Note: the official ACDL skill suggests PascalCase definitions / ALL_CAPS
labels, but the user overrides that for this project.)

**How to apply:** keep snake_case for any identifier WE define in the ACDL. Real external names stay as
written: the Python class `BashToolAgent`, env vars (`DOIT_CMD_LOG`), shell constructs (`PROMPT_COMMAND`,
`DEBUG`), and `sys.`/`env.`/`resp.` references (already snake_case). ACDL keywords `ForEach`/`If`/`Else`/
`Switch` and time indices `@T`/`@t` are unchanged. Stages 1-4 still use the old PascalCase and should be
swept to snake_case for consistency when convenient. Related: [[doit-next-output-awareness]].
