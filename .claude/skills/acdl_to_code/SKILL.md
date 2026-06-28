# ACDL to Code Implementation Skill

You are an expert at implementing LLM agent systems from ACDL (Agentic Context Description Language) specifications. Given an ACDL spec, you generate Python code that builds the prompt/messages array for LLM API calls.

## Your Task

When given an ACDL specification, generate:
1. Template constants and template functions
2. Data structures for history, substeps, and tools
3. Helper/retrieval functions (stubbed with TODO)
4. A `build_messages()` function with comments showing which ACDL line each section implements
5. A brief usage example

---

## ACDL to Code Mapping

### Roles → Message Objects

| ACDL | Python Code |
|------|-------------|
| `S: content` | `{"role": "system", "content": content}` |
| `U: content` | `{"role": "user", "content": content}` |
| `A: content` | `{"role": "assistant", "content": content}` |
| `T: content` | `{"role": "tool", "content": content, "tool_call_id": id}` |

### Time Indices → Parameters/Loops

| ACDL | Python Code |
|------|-------------|
| `@T` | `turn` (current turn parameter) |
| `@t` | `t` (loop variable) |
| `@1`, `@2` | Literal `1`, `2` |
| `range(1, @T)` | `range(1, turn)` |

### Substep Indices (for tool loops)

| ACDL | Python Code |
|------|-------------|
| `@T.I` | `substep` (second parameter, 0 = no partial) |
| `@t.@i` | `state.history[t-1].substeps[i-1]` |
| `@t.substeps` | `len(state.history[t-1].substeps)` |
| `range(1, @t.substeps)` | `range(len(state.history[t-1].substeps))` |
| `range(1, @T.I)` | `range(substep)` (for partial turn) |

### Named Variables & Dereference

| ACDL | Python Code |
|------|-------------|
| `Name C := sys.value[@T]` | `C = state.value` |
| `@$C` | Use `C` as a value (e.g., `range(C, turn)`) |
| `range(@$C, @T)` | `range(C, turn)` |

### Context Variables → Data Lookups

| ACDL | Python Code |
|------|-------------|
| `env.user_input[@t]` | `state.history[t-1].user_input` |
| `env.user_input[@T]` | `current_input` (function parameter) |
| `env.user_input[@1]` | `state.history[0].user_input` or initial param |
| `sys.tool_response[@t]` | `state.history[t-1].tool_response` |
| `resp.reasoning[@t]` | `state.history[t-1].assistant_response` |
| `resp.reasoning[@t.@i]` | `state.history[t-1].substeps[i-1].reasoning` |

### Templates → Constants and Functions

| ACDL | Python Code |
|------|-------------|
| `SYSTEM_PROMPT` | `SYSTEM_PROMPT = """..."""` |
| `TEMPLATE(arg)` | `def TEMPLATE(arg): return f"..."` |
| `TEMPLATE(a, b, c)` | `def TEMPLATE(a, b, c): return f"..."` |

### Control Flow → Python Control Flow

| ACDL | Python Code |
|------|-------------|
| `ForEach(@t: range(1, @T)) { ... }` | `for t in range(1, turn): ...` |
| `ForEach(item: $list) { ... }` | `for item in list_var: ...` |
| `ForEach(item: sys.items[@t]) { ... }` | `for item in state.history[t-1].items: ...` |
| `If condition { ... }` | `if condition: ...` |
| `If cond { } Else { }` | `if cond: ... else: ...` |
| `Switch x { Case "a": {...} Case "b": {...} Default: {...} }` | `if x == "a": ... elif x == "b": ... else: ...` |

### Agent Parameters

| ACDL | Python Code |
|------|-------------|
| `Agent[@T]: { ... }` | `def build_messages(turn, state, ...) -> List[dict]` |
| `Agent[@T.I]: { ... }` | `def build_messages(turn, substep, state, ...) -> List[dict]` |
| `Agent[@T, ctx, mode]: { ... }` | `def build_messages(turn, state, ctx, mode, ...) -> List[dict]` |

---

## Data Structure Patterns

### Basic Agent (no tools)

```python
@dataclass
class TurnHistory:
    user_input: str
    assistant_response: Optional[str] = None

@dataclass
class AgentState:
    history: List[TurnHistory]
```

### Agent with Tool Calls (substeps)

```python
@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    response: Optional[str] = None

@dataclass
class SubStep:
    reasoning: str
    tool_calls: List[ToolCall]

@dataclass
class TurnHistory:
    user_input: str
    substeps: List[SubStep]
    final_response: Optional[str] = None

    def __post_init__(self):
        if self.substeps is None:
            self.substeps = []

@dataclass
class AgentState:
    history: List[TurnHistory]
```

### Agent with Compaction

```python
@dataclass
class AgentState:
    history: List[TurnHistory]
    last_compaction_turn: int = 1
    conversation_summary: Optional[str] = None
```

### Agent with Mode/Config

```python
@dataclass
class TurnHistory:
    user_input: str
    assistant_response: Optional[str] = None
    mode: str = "default"
    mode_changed: bool = False
```

---

## Implementation Patterns

### Pattern 1: Basic History Loop

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

**Python:**
```python
def build_messages(turn: int, state: AgentState, current_input: str) -> List[dict]:
    messages = []

    # S: INSTRUCTIONS
    messages.append({"role": "system", "content": INSTRUCTIONS})

    # ForEach(@t: range(1, @T))
    for t in range(1, turn):
        # U: env.input[@t]
        messages.append({"role": "user", "content": state.history[t-1].user_input})
        # A: resp.output[@t]
        messages.append({"role": "assistant", "content": state.history[t-1].assistant_response})

    # U: env.input[@T]
    messages.append({"role": "user", "content": current_input})

    return messages
```

### Pattern 2: Tool Calls with Substeps

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
    // Partial current turn
    If @T.I > 0 {
        ForEach(@i: range(1, @T.I)) {
            // ... same structure
        }
    }
}
```

**Python:**
```python
def build_messages(turn: int, substep: int, state: AgentState, initial_task: str) -> List[dict]:
    messages = []

    # S: INSTRUCTIONS
    messages.append({"role": "system", "content": INSTRUCTIONS})

    # U: env.task[@1]
    messages.append({"role": "user", "content": initial_task})

    # ForEach(@t: range(1, @T))
    for t in range(1, turn):
        turn_history = state.history[t-1]

        # ForEach(@i: range(1, @t.substeps))
        for i in range(len(turn_history.substeps)):
            substep_data = turn_history.substeps[i]

            # A: { resp.reasoning[@t.@i], tool calls }
            messages.append({
                "role": "assistant",
                "content": substep_data.reasoning,
                "tool_calls": [
                    {"id": tc.id, "function": {"name": tc.name, "arguments": tc.args}}
                    for tc in substep_data.tool_calls
                ]
            })

            # T: for each tool
            for tool in substep_data.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": tool.result
                })

        # A: resp.answer[@t]
        messages.append({"role": "assistant", "content": turn_history.final_response})

    # If @T.I > 0 - partial current turn
    if substep > 0:
        current_turn = state.history[turn-1]
        for i in range(substep):
            substep_data = current_turn.substeps[i]
            messages.append({
                "role": "assistant",
                "content": substep_data.reasoning,
                "tool_calls": [{"id": tc.id, ...} for tc in substep_data.tool_calls]
            })
            for tool in substep_data.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": tool.result
                })

    return messages
```

### Pattern 3: Compaction with Variable Dereference

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

**Python:**
```python
def build_messages(turn: int, state: AgentState, current_input: str) -> List[dict]:
    messages = []

    # S: INSTRUCTIONS
    messages.append({"role": "system", "content": INSTRUCTIONS})

    # Name C := sys.last_compaction_turn[@T]
    C = state.last_compaction_turn

    # If @$C > 1
    if C > 1:
        # U: SUMMARY_HEADER
        messages.append({"role": "user", "content": SUMMARY_HEADER})
        # A: sys.conversation_summary[@$C]
        messages.append({"role": "assistant", "content": state.conversation_summary})

    # ForEach(@t: range(@$C, @T)) - note: starts from C, not 1
    for t in range(C, turn):
        messages.append({"role": "user", "content": state.history[t-1].user_input})
        messages.append({"role": "assistant", "content": state.history[t-1].assistant_response})

    # U: env.input[@T]
    messages.append({"role": "user", "content": current_input})

    return messages
```

### Pattern 4: List Iteration with Retrieval

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

**Python:**
```python
def DOCUMENT_BLOCK(id: str, title: str, content: str) -> str:
    return f"[{id}] {title}\n{content}"

def retrieve(query: str, k: int) -> List[dict]:
    # TODO: Implement retrieval logic
    pass

def build_messages(turn: int, state: AgentState, current_query: str) -> List[dict]:
    messages = []

    # S: INSTRUCTIONS
    messages.append({"role": "system", "content": INSTRUCTIONS})

    # Name docs := retrieve(env.query[@T], 5)
    docs = retrieve(current_query, 5)

    # Build user content
    user_content = ""

    # ForEach(doc: $docs)
    for doc in docs:
        user_content += DOCUMENT_BLOCK(doc["id"], doc["title"], doc["content"]) + "\n\n"

    user_content += QUESTION_HEADER + "\n" + current_query
    messages.append({"role": "user", "content": user_content})

    return messages
```

### Pattern 5: Switch/Case

**ACDL:**
```acdl
Agent[@T]: {
    S: BASE_INSTRUCTIONS
    Switch env.mode[@T] {
        Case "creative": {
            S: CREATIVE_ADDON
        }
        Case "precise": {
            S: PRECISE_ADDON
        }
        Default: {
            S: DEFAULT_ADDON
        }
    }
    U: env.input[@T]
}
```

**Python:**
```python
def build_messages(turn: int, state: AgentState, current_input: str, mode: str) -> List[dict]:
    messages = []

    # S: BASE_INSTRUCTIONS
    messages.append({"role": "system", "content": BASE_INSTRUCTIONS})

    # Switch env.mode[@T]
    if mode == "creative":
        messages.append({"role": "system", "content": CREATIVE_ADDON})
    elif mode == "precise":
        messages.append({"role": "system", "content": PRECISE_ADDON})
    else:
        messages.append({"role": "system", "content": DEFAULT_ADDON})

    # U: env.input[@T]
    messages.append({"role": "user", "content": current_input})

    return messages
```

### Pattern 6: Conditional Inside Loop

**ACDL:**
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

**Python:**
```python
def build_messages(turn: int, state: AgentState, current_input: str) -> List[dict]:
    messages = []

    # S: INSTRUCTIONS
    messages.append({"role": "system", "content": INSTRUCTIONS})

    # ForEach(@t: range(1, @T))
    for t in range(1, turn):
        turn_data = state.history[t-1]

        # U: { env.input[@t], If sys.has_context[@t] {...} }
        user_content = turn_data.user_input
        if turn_data.has_context:
            user_content += "\n" + CONTEXT_HEADER + "\n" + turn_data.context
        messages.append({"role": "user", "content": user_content})

        # A: resp.output[@t]
        messages.append({"role": "assistant", "content": turn_data.assistant_response})

    # U: env.input[@T]
    messages.append({"role": "user", "content": current_input})

    return messages
```

---

## Guidelines

1. **Comment each section** with the ACDL line it implements
2. **Index translation**: ACDL is 1-indexed, Python is 0-indexed (`@t` → `t-1`)
3. **Template functions**: ALL_CAPS with args become `def TEMPLATE(args) -> str`
4. **Retrieval functions**: Stub with `# TODO: Implement`
5. **Preserve order**: Message order must match ACDL specification order
6. **Substep handling**: Use nested loops for `@t.@i` patterns
7. **Partial turns**: `If @T.I > 0` handles mid-turn state (in-progress tool calls)
8. **Variable deref**: `Name X := ...` then `@$X` uses X as a value
9. **Only define what you use**: Do not create data classes, enums, or helper functions that are not actually used in the implementation

---

## Output Format

Always provide:

1. **Imports and template constants**
2. **Template functions** (for `TEMPLATE(args)` patterns)
3. **Data classes** (TurnHistory, SubStep, ToolCall as needed)
4. **Helper functions** (retrieval, etc. with TODO)
5. **`build_messages()` function** with ACDL comments
6. **Usage example**

```python
# Example usage
state = AgentState(history=[...])
messages = build_messages(turn=3, state=state, current_input="Hello")
# Send to LLM API
response = client.chat.completions.create(model="...", messages=messages)
```

---

## Verification Checklist

**After completing your implementation, verify it by going through the ACDL spec line by line:**

1. **Read each line of the ACDL specification** (excluding comments)
2. **For each line, confirm there is corresponding code** in your `build_messages()` function
3. **For each referenced variable, confirm it can be accessed** — either from:
   - Function parameters (e.g., `turn`, `substep`, `current_input`)
   - State fields (e.g., `state.history[t-1].field`)
   - Named variables assigned earlier (e.g., `Name C := ...` then `C`)
   - Computed values (e.g., retrieval function results)
4. **Pay special attention to:**
   - All items inside multi-part messages (e.g., `S: { A, B, C }` — all three must appear)
   - All branches of `Switch/Case` statements (including `Default`)
   - All elements inside `ForEach` loops
   - Conditional blocks (`If`) and their contents
   - Template constants and template function calls

**Example verification:**

```acdl
Agent[@T]: {
    S: {                          // ✓ Check: system message exists
        INSTRUCTIONS              // ✓ Check: INSTRUCTIONS constant used
        CONTEXT_INFO(env.x[@1])   // ✓ Check: CONTEXT_INFO function called with correct arg
    }
    ForEach(@t: range(1, @T)) {   // ✓ Check: for loop with range(1, turn)
        U: env.input[@t]          // ✓ Check: user message with history[t-1].input
        A: resp.output[@t]        // ✓ Check: assistant message with history[t-1].output
    }
    U: env.input[@T]              // ✓ Check: final user message with current_input
}
```

If any ACDL line does not have corresponding code, add it before finalizing your implementation.