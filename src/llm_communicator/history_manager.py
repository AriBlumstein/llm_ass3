import json
import os
from pathlib import Path
from typing import Any, Dict, List

def find_project_root() -> Path:
    """
    Search upwards from the current working directory to locate the project/repository root.
    Looks for markers like '.git', 'pyproject.toml', 'doit.cfg', or '.doit'.
    If no marker is found, returns the current working directory.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (
            (parent / ".git").exists()
            or (parent / "pyproject.toml").exists()
            or (parent / "doit.cfg").exists()
            or (parent / ".doit").exists()
        ):
            return parent
    return cwd

def get_history_file_path() -> Path:
    """
    Get the path to the history file for the current shell session (isolated by PPID).
    Stored in a hidden directory in the resolved project root directory.
    """
    ppid = os.environ.get("DOIT_PPID") or str(os.getppid())
    project_root = find_project_root()
    history_dir = project_root / ".doit"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / f"history_{ppid}.jsonl"

def append_history_turn(prompt: str, command: str, output: str, relevant_ids: List[int] = None, suggested_command: str = "") -> None:
    """
    Appends a new conversation/execution turn to the session history.

    `suggested_command` holds a command that was proposed by an `answer_question` turn but
    NOT executed, so a later "execute it" follow-up can resolve and run it.
    """
    path = get_history_file_path()

    # Calculate the next incrementing ID
    next_id = 1
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_turn = json.loads(lines[-1].strip())
                    next_id = last_turn.get("id", 0) + 1
        except Exception:
            pass

    turn = {
        "id": next_id,
        "prompt": prompt,
        "command": command,
        "suggested_command": suggested_command,
        "output": output,
        "relevant_ids": relevant_ids if relevant_ids is not None else []
    }
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn) + "\n")

def get_history_metadata(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves metadata (id, prompt, command) for the last N turns.
    Omits execution outputs to keep context token count minimal.
    """
    path = get_history_file_path()
    if not path.exists():
        return []
        
    metadata = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turn = json.loads(line)
                metadata.append({
                    "id": turn["id"],
                    "prompt": turn["prompt"],
                    "command": turn["command"],
                    "suggested_command": turn.get("suggested_command", "")
                })
    except Exception:
        return []
        
    return metadata[-limit:]

def get_full_turns(ids: List[int]) -> List[Dict[str, Any]]:
    """
    Retrieves full turn logs (including execution outputs) for specific target IDs.
    Returns them in the order of the requested IDs.
    """
    path = get_history_file_path()
    if not path.exists() or not ids:
        return []
        
    id_set = set(ids)
    matched_turns = {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turn = json.loads(line)
                if turn.get("id") in id_set:
                    matched_turns[turn["id"]] = turn
    except Exception:
        return []
        
    # Return in the order of IDs specified by the analyzer
    return [matched_turns[turn_id] for turn_id in ids if turn_id in matched_turns]

def get_latest_suggested_command() -> "tuple[int, str] | None":
    """
    Return (turn_id, suggested_command) for the most recent turn that proposed a command via
    an answer (Rule 9) but did not execute it, or None if no such turn exists. Used by the
    deterministic "execute that" route to run the last suggestion without asking the model.
    """
    path = get_history_file_path()
    if not path.exists():
        return None

    latest = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turn = json.loads(line)
                suggested = turn.get("suggested_command")
                if suggested:
                    latest = (turn.get("id"), suggested)
    except Exception:
        return None

    return latest


def clear_history() -> None:
    """
    Deletes the current shell session's history file entirely.
    """
    path = get_history_file_path()
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
