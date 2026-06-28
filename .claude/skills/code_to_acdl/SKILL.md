---
name: code_to_acdl
description: Generate an ACDL (Agentic Context Description Language) specification from Python agent code that builds an LLM messages array. The inverse of acdl_to_code.
---

# Code to ACDL Documentation Skill

You are an expert at reverse-engineering ACDL (Agentic Context Description Language) specifications from Python agent code. Given Python that builds a prompt/`messages` array for an LLM API call (typically a `build_messages()`-style function), you produce the ACDL spec that describes its context structure.

This is the exact inverse of the `acdl_to_code` skill. Read it together with this one: a spec you emit here, fed through `acdl_to_code`, should round-trip back to equivalent code.

## Your Task

Given Python agent code, produce:
1. A single **named prompt definition** describing the context (e.g. `DoitToolUseAgent[@T]: { ... }`)
2. ACDL comments (optional) noting which source construct each line came from
3. Symbolic labels (ALL_CAPS) standing in for literal prompt strings — not the inlined text
4. Correct role ordering matching the order messages are appended in the code

You describe **structure and evolution of the context**, not the implementation mechanics. The `messages` list, `.append()` calls, and 0-index arithmetic do not appear in ACDL.

---

## Code → ACDL Mapping

### Message Objects → Roles

| Python Code | ACDL |
|-------------|------|
| `{"role": "system", "content": c}` | `S: c` |
| `{"role": "user", "content": c}` | `U: c` |
| `{"role": "assistant", "content": c}` | `A: c` |
| `{"role": "tool", "content": c, "tool_call_id": id}` | `T: c` |

### Parameters/Loops → Time Indices

ACDL is **1-indexed**; Python history access is 0-indexed. Re-introduce the `+1` when reversing (`state.history[t-1]` came from `@t`, so it maps back to `@t`).

| Python Code | ACDL |
|-------------|------|
| `turn` (current-turn parameter) | `@T` |
| `t` (loop variable) | `@t` |
| Literal `1`, `2` | `@1`, `@2` |
| `range(1, turn)` | `range(1, @T)` |
| `state.history[t-1]` | (the `@t`-th turn) |

### Substep Indices (tool loops)

| Python Code | ACDL |
|-------------|------|
| `substep` (second parameter, 0 = no partial) | `@T.I` |
| `state.history[t-1].substeps[i-1]` | `@t.@i` |
| `len(state.history[t-1].substeps)` | `@t.substeps` |
| `range(len(state.history[t-1].substeps))` | `range(1, @t.substeps)` |
| `range(substep)` (partial current turn) | `range(1, @T.I)` |

### Data Lookups → Context Variables (the key reverse judgment)

Choose the prefix by the **source** of the value:

- `env.*` — external input / user-supplied / function parameters (the environment).
- `sys.*` — system- or platform-provided data: tool results, retrieved/stored state, reconstructed history, fixed instruction strings.
- `resp.*` — values produced by a model response (this agent's or another agent's output).

| Python Code | ACDL |
|-------------|------|
| `current_input` (function parameter) | `env.user_input[@T]` |
| `state.history[t-1].user_input` | `env.user_input[@t]` |
| `state.history[0].user_input` | `env.user_input[@1]` |
| `state.history[t-1].tool_response` | `sys.tool_response[@t]` |
| `state.history[t-1].assistant_response` | `resp.reasoning[@t]` / `resp.output[@t]` |
| `state.history[t-1].substeps[i-1].reasoning` | `resp.reasoning[@t.@i]` |
| output of another agent in the pipeline | `resp.<agent_name>[@T].<field>` |

### Constants & Functions → Templates

| Python Code | ACDL |
|-------------|------|
| `SYSTEM_PROMPT = """..."""` (used as content) | `SYSTEM_PROMPT` (label) |
| `def TEMPLATE(arg): return f"..."` then `TEMPLATE(x)` | `TEMPLATE(x)` |
| Inline string literal used as content | promote to an ALL_CAPS label; do NOT inline the text |

### Control Flow → ACDL Control Flow

| Python Code | ACDL |
|-------------|------|
| `for t in range(1, turn): ...` | `ForEach(@t: range(1, @T)) { ... }` |
| `for item in some_list: ...` | `ForEach(item: $some_list) { ... }` |
| `for item in state.history[t-1].items: ...` | `ForEach(item: sys.items[@t]) { ... }` |
| `if cond: ...` | `If cond { ... }` |
| `if cond: ... else: ...` | `If cond { } Else { }` |
| `if x == "a": ... elif x == "b": ... else: ...` (one variable) | `Switch x { Case "a": {...} Case "b": {...} Default: {...} }` |

### Named Variables & Dereference

| Python Code | ACDL |
|-------------|------|
| `C = state.last_compaction_turn` | `Name C := sys.last_compaction_turn[@T]` |
| `range(C, turn)`, `if C > 1:` | `range(@$C, @T)`, `If @$C > 1` |
| `docs = retrieve(current_query, 5)` | `Name docs := retrieve(env.query[@T], 5)` |

### Function Signature → Agent Header

| Python Code | ACDL |
|-------------|------|
| `def build_messages(turn, state, current_input)` | `Agent[@T]: { ... }` |
| `def build_messages(turn, substep, state, ...)` | `Agent[@T.I]: { ... }` |
| `def build_messages(turn, state, ctx, mode)` | `Agent[@T, ctx, mode]: { ... }` |

Name the prompt definition after the agent's role (e.g. `DoitToolUseAgent`), not `Agent`, when the purpose is known.

---

## Reverse-Direction Constructs (no 1:1 forward analog — handle explicitly)

These are where naive line-by-line reversal goes wrong. The forward skill expands ACDL into many `.append()` calls and string concatenations; reversing means **collapsing** those back into structure.

### Collapse string accumulation into one role block

Code that builds a single message by concatenation is **one** multi-item role block, not several messages:

```python
user_content = ""
for doc in docs:
    user_content += DOCUMENT_BLOCK(doc["id"], doc["title"], doc["content"]) + "\n\n"
user_content += QUESTION_HEADER + "\n" + current_query
messages.append({"role": "user", "content": user_content})
```

→

```acdl
U: {
    ForEach(doc: $docs) {
        DOCUMENT_BLOCK(doc.id, doc.title, doc.content)
    }
    QUESTION_HEADER
    env.query[@T]
}
```

### Conditional concatenation → `If` inside the role block

```python
user_content = turn_data.user_input
if turn_data.has_context:
    user_content += "\n" + CONTEXT_HEADER + "\n" + turn_data.context
messages.append({"role": "user", "content": user_content})
```

→

```acdl
U: {
    env.input[@t]
    If sys.has_context[@t] {
        CONTEXT_HEADER
        sys.context[@t]
    }
}
```

### Abstract away mechanics

The `messages = []` list, every `.append(...)`, `+= "\n"` glue, and `[t-1]` index math are implementation noise — they are **not** represented in ACDL. Describe only what content appears, in what role, in what order, under what conditions.

### Inline literals → symbolic labels

A literal prompt string in the code becomes an ALL_CAPS label (`SYSTEM_PROMPT`, `QUESTION_HEADER`), never embedded text. Labels represent abstract content sources.

### One named definition, order preserved

Emit a single named prompt definition. Message order in ACDL must match the order of appends in the code exactly.

---

## Patterns (Python in → ACDL out)

### Pattern 1: Basic History Loop

**Python:**
```python
def build_messages(turn, state, current_input):
    messages = [{"role": "system", "content": INSTRUCTIONS}]
    for t in range(1, turn):
        messages.append({"role": "user", "content": state.history[t-1].user_input})
        messages.append({"role": "assistant", "content": state.history[t-1].assistant_response})
    messages.append({"role": "user", "content": current_input})
    return messages
```

**ACDL:**
```acdl
Agent[@T]: {
    S: INSTRUCTIONS
    ForEach(@t: range(1, @T)) {
        U: env.input[@t]
        A: resp.output[@t]
    }
    U: env.input[@T]
}
```

### Pattern 2: Tool Calls with Substeps (+ partial current turn)

**Python:**
```python
def build_messages(turn, substep, state, initial_task):
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": initial_task}]
    for t in range(1, turn):
        th = state.history[t-1]
        for i in range(len(th.substeps)):
            ss = th.substeps[i]
            messages.append({"role": "assistant", "content": ss.reasoning,
                             "tool_calls": [{"id": tc.id, "function": {"name": tc.name, "arguments": tc.args}}
                                            for tc in ss.tool_calls]})
            for tool in ss.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tool.id, "content": tool.result})
        messages.append({"role": "assistant", "content": th.final_response})
    if substep > 0:
        cur = state.history[turn-1]
        for i in range(substep):
            ...  # same assistant + tool structure
    return messages
```

**ACDL:**
```acdl
Agent[@T.I]: {
    S: INSTRUCTIONS
    U: env.task[@1]
    ForEach(@t: range(1, @T)) {
        ForEach(@i: range(1, @t.substeps)) {
            A: {
                resp.reasoning[@t.@i]
                ForEach(tool: sys.tool_calls[@t.@i]) {
                    tool.invocation
                }
            }
            ForEach(tool: sys.tool_calls[@t.@i]) {
                T: {
                    tool.id
                    tool.result
                }
            }
        }
        A: resp.answer[@t]
    }
    If @T.I > 0 {
        ForEach(@i: range(1, @T.I)) {
            // same A + T substep structure for the in-progress turn
        }
    }
}
```

### Pattern 3: Compaction with Variable Dereference

**Python:**
```python
def build_messages(turn, state, current_input):
    messages = [{"role": "system", "content": INSTRUCTIONS}]
    C = state.last_compaction_turn
    if C > 1:
        messages.append({"role": "user", "content": SUMMARY_HEADER})
        messages.append({"role": "assistant", "content": state.conversation_summary})
    for t in range(C, turn):
        messages.append({"role": "user", "content": state.history[t-1].user_input})
        messages.append({"role": "assistant", "content": state.history[t-1].assistant_response})
    messages.append({"role": "user", "content": current_input})
    return messages
```

**ACDL:**
```acdl
Agent[@T]: {
    S: INSTRUCTIONS
    Name C := sys.last_compaction_turn[@T]
    If @$C > 1 {
        U: SUMMARY_HEADER
        A: sys.conversation_summary[@$C]
    }
    ForEach(@t: range(@$C, @T)) {
        U: env.input[@t]
        A: resp.output[@t]
    }
    U: env.input[@T]
}
```

### Pattern 4: List Iteration with Retrieval

**Python:** (the accumulation example under "Collapse string accumulation" above)

**ACDL:**
```acdl
Agent[@T]: {
    S: INSTRUCTIONS
    U: {
        Name docs := retrieve(env.query[@T], 5)
        ForEach(doc: $docs) {
            DOCUMENT_BLOCK(doc.id, doc.title, doc.content)
        }
        QUESTION_HEADER
        env.query[@T]
    }
}
```

### Pattern 5: Switch/Case

**Python:**
```python
def build_messages(turn, state, current_input, mode):
    messages = [{"role": "system", "content": BASE_INSTRUCTIONS}]
    if mode == "creative":
        messages.append({"role": "system", "content": CREATIVE_ADDON})
    elif mode == "precise":
        messages.append({"role": "system", "content": PRECISE_ADDON})
    else:
        messages.append({"role": "system", "content": DEFAULT_ADDON})
    messages.append({"role": "user", "content": current_input})
    return messages
```

**ACDL:**
```acdl
Agent[@T]: {
    S: BASE_INSTRUCTIONS
    Switch env.mode[@T] {
        Case "creative": { S: CREATIVE_ADDON }
        Case "precise":  { S: PRECISE_ADDON }
        Default:         { S: DEFAULT_ADDON }
    }
    U: env.input[@T]
}
```

### Pattern 6: Conditional Inside Loop

See "Conditional concatenation → `If` inside the role block" above; wrapped in a history `ForEach`:

```acdl
Agent[@T]: {
    S: INSTRUCTIONS
    ForEach(@t: range(1, @T)) {
        U: {
            env.input[@t]
            If sys.has_context[@t] {
                CONTEXT_HEADER
                sys.context[@t]
            }
        }
        A: resp.output[@t]
    }
    U: env.input[@T]
}
```

---

## Guidelines

1. **Name the definition** by the agent's role; parameterize with `@T` (and `@T.I` if there is a substep parameter).
2. **Re-introduce 1-indexing**: `state.history[t-1]` → `@t`. Never emit `t-1` in ACDL.
3. **Collapse, don't transcribe**: concatenations building one message become one role block; only `.append()` calls create new messages.
4. **Prefix by source**: parameters/external → `env.*`, system/tool/stored → `sys.*`, model output → `resp.*`.
5. **Promote literals to labels**: ALL_CAPS symbolic labels, never inlined prompt text.
6. **Template calls stay calls**: `TEMPLATE(args)` in code → `TEMPLATE(args)` in ACDL.
7. **Preserve order** exactly as messages are appended.
8. **Drop mechanics**: no `messages` list, no `.append`, no index math.
9. **Only describe what the code produces** — do not invent roles, branches, or fields the code never builds.
10. **Multi-agent pipelines**: if the code runs several LLM calls (e.g. a resolver, a main agent, a filter), emit one named definition per call and reference earlier outputs as `resp.<agent_name>[@T].<field>` (see `data/multi_turn_stage_1/*.acdl`).

---

## Output Format

Provide:

1. **The named prompt definition(s)** in an ```acdl block, message order preserved.
2. **A short list of the labels you introduced** (e.g. `INSTRUCTIONS`, `QUESTION_HEADER`) and what source string each abstracts, so the mapping back is unambiguous.

```acdl
// One definition per LLM call; reference prior outputs via resp.<agent>[@T].<field>
DoitToolUseAgent[@T]: {
    S: { SYSTEM_PROMPT, TOOLS_DEFINITION }
    ForEach(@t: range(1, @T)) {
        U: env.user_input[@t]
        A: resp.tool_call[@t]
        T: sys.tool_result[@t]
    }
    U: env.user_input[@T]
}
```

---

## Verification Checklist

**After producing the ACDL, walk the source code and confirm the reverse mapping holds:**

1. **Every `messages.append(...)`** maps to exactly one role entry (or is folded into a role block via concatenation) — and nothing extra was invented.
2. **Role of each append** matches its ACDL role (`system`→`S`, `user`→`U`, `assistant`→`A`, `tool`→`T`).
3. **Every `for` loop** became a `ForEach` with the right range; **every `if`/`elif`/`else`** became `If`/`Else`/`Switch`.
4. **Accumulated content** (`x = ...; x += ...; append(x)`) collapsed into a single multi-item role block, NOT multiple messages.
5. **Indices re-incremented**: no `t-1` survives; literals like `history[0]` became `@1`.
6. **Prefixes correct**: each value is `env.`/`sys.`/`resp.` according to its source.
7. **Literals promoted**: no raw prompt strings inlined; all are ALL_CAPS labels.
8. **Order matches** the append order in the function.
9. **Mechanics absent**: no `messages`, `.append`, or index arithmetic leaked into the ACDL.

**Example verification:**

```python
messages.append({"role": "system", "content": INSTRUCTIONS})  # ✓ → S: INSTRUCTIONS
for t in range(1, turn):                                       # ✓ → ForEach(@t: range(1, @T))
    messages.append({"role": "user",
        "content": state.history[t-1].user_input})             # ✓ → U: env.input[@t]   (t-1 → @t)
    messages.append({"role": "assistant",
        "content": state.history[t-1].assistant_response})     # ✓ → A: resp.output[@t]
messages.append({"role": "user", "content": current_input})   # ✓ → U: env.input[@T]
```

If any append or control-flow construct has no corresponding ACDL line — or any ACDL line has no source construct — fix it before finalizing.
