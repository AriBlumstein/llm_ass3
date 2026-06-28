"""
Persistent user MEMORY for the doit agent.

Memory lives in the installed repo's `.doit/` directory: `<repo>/.doit/memories.json`. It is
located via THIS MODULE's path (not the current working directory), so it is reachable no matter
which directory doit is run from - that is what lets a memory persist across terminals and
directories within this checkout. The repo's `.doit/` is gitignored, so memories are never
committed. (Trade-off vs a home-dir store: memory is tied to this checkout and would be lost if
the repo is moved or deleted.)

Contrast with history, which is per shell-session and located by walking UP from the cwd
(`<project_root>/.doit/history_<ppid>.jsonl`) - so history is cwd-relative while memory is fixed to
the install.

Stored as a JSON array (not JSONL) because memory needs update/delete, not just append. Each
record: {id, content, created_at, active}. Deletes are tombstones (active=false) so ids stay
stable for `update`/`delete` operations referenced by the memory-manager sub-call.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def get_memory_file_path() -> Path:
    """
    The memory file at `<repo>/.doit/memories.json`. Resolved from this module's location
    (src/llm_communicator/memory_manager.py -> repo root), NOT the cwd, so it is the same file
    from any directory doit is invoked in.
    """
    repo_root = Path(__file__).resolve().parents[2]
    memory_dir = repo_root / ".doit"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / "memories.json"


def _read_all() -> List[Dict[str, Any]]:
    path = get_memory_file_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _write_all(records: List[Dict[str, Any]]) -> None:
    """Atomic rewrite of the whole store (small data)."""
    path = get_memory_file_path()
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, path)


def _next_id(records: List[Dict[str, Any]]) -> int:
    return max((r.get("id", 0) for r in records), default=0) + 1


def load_memories() -> List[Dict[str, Any]]:
    """Active (non-tombstoned) memories, in insertion order."""
    return [r for r in _read_all() if r.get("active", True)]


def add_memory(content: str) -> int:
    """Append a new active memory; returns its id (or -1 for empty content)."""
    content = (content or "").strip()
    if not content:
        return -1
    records = _read_all()
    new_id = _next_id(records)
    records.append({
        "id": new_id,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    })
    _write_all(records)
    return new_id


def update_memory(memory_id: int, content: str) -> bool:
    """Revise an active memory's content in place. Returns True if found."""
    records = _read_all()
    found = False
    for r in records:
        if r.get("id") == memory_id and r.get("active", True):
            r["content"] = (content or "").strip()
            found = True
    if found:
        _write_all(records)
    return found


def delete_memory(memory_id: int) -> bool:
    """Tombstone (active=false) rather than hard-delete, so ids stay stable. True if found."""
    records = _read_all()
    found = False
    for r in records:
        if r.get("id") == memory_id and r.get("active", True):
            r["active"] = False
            found = True
    if found:
        _write_all(records)
    return found


def apply_operations(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply add/update/delete operations (as produced by the memory-manager sub-call). Unknown or
    malformed ops are ignored. Returns the ops that were actually applied (for user feedback).
    """
    applied: List[Dict[str, Any]] = []
    for op in operations or []:
        if not isinstance(op, dict):
            continue
        kind = (op.get("op") or "").lower()
        if kind == "add":
            if add_memory(op.get("content", "")) != -1:
                applied.append(op)
        elif kind == "update" and op.get("id") is not None:
            if update_memory(int(op["id"]), op.get("content", "")):
                applied.append(op)
        elif kind == "delete" and op.get("id") is not None:
            if delete_memory(int(op["id"])):
                applied.append(op)
    return applied


def render_memories() -> str:
    """
    The memory block injected into the system prompt on every invocation. Empty string when there
    are no memories, so the prompt is unchanged for users who have never stored one.
    """
    active = load_memories()
    if not active:
        return ""
    lines = [f"- (memory {r.get('id')}) {r.get('content')}" for r in active]
    return (
        "KNOWN FACTS ABOUT THE USER (persistent memory - apply these when relevant).\n"
        "They are listed OLDEST first; if any two conflict, the MOST RECENT one (last listed, "
        "highest id) is authoritative and overrides the older one:\n"
        + "\n".join(lines)
    )


def clear_memories() -> None:
    """Delete the entire memory store (used by tests / a future reset command)."""
    path = get_memory_file_path()
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
