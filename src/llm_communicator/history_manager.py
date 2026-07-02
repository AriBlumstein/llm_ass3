import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

def doit_root() -> Path:
    """The install-relative `<repo>/.doit` directory that holds every session folder + memory."""
    return Path(__file__).resolve().parents[2] / ".doit"


def current_session_id() -> str:
    """
    The current shell session's id = its shell PID. `doit-init.sh` PINS this to the interactive
    shell's `$$` and exports it as `DOIT_PPID` (the launcher preserves it); it MUST be the
    shell-exported value, not re-derived, because the launcher's own `$PPID` is not always this shell
    (e.g. the VS Code terminal). Falls back to `os.getppid()` for a bare run without shell integration.
    """
    return os.environ.get("DOIT_PPID") or str(os.getppid())


def get_session_dir(session_id: Optional[str] = None) -> Path:
    """The per-session folder `<repo>/.doit/history_<pid>/` (created if missing). Holds `doit.jsonl`
    (doit's own turns), `cmdlog.tsv` (the shell recorder), and `session.json` (the registry entry)."""
    sid = session_id or current_session_id()
    session_dir = doit_root() / f"history_{sid}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_history_file_path(session_dir: Optional[Path] = None) -> Path:
    """doit's own history file `<session dir>/doit.jsonl` (defaults to the current session). Always
    PID-named and install-relative, so sessions never collide regardless of the cwd."""
    return (session_dir or get_session_dir()) / "doit.jsonl"


def _read_turns(path: Path) -> List[Dict[str, Any]]:
    """All turns from a `doit.jsonl` file, in order; tolerant of blank/corrupt lines. [] if absent."""
    if not path.exists():
        return []
    turns: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return turns

def append_history_turn(prompt: str, command: str, output: str, relevant_ids: List[int] = None, suggested_command: str = "", source: str = "doit", hist_n: int = None, cwd: str = None) -> None:
    """
    Appends a new conversation/execution turn to the session history.

    `suggested_command` holds a command that was proposed by an `answer_question` turn but
    NOT executed, so a later "execute it" follow-up can resolve and run it.

    `source` is "doit" for the agent's own turns, or "user" for a command the user ran DIRECTLY in
    the terminal (synced from shell history). `hist_n` is the shell-history index of a user command
    (the de-dup high-water mark); it is None for doit turns. `cwd` is the directory a user command
    ran in (from the shell recorder), so doit can re-run it there later; None when unknown.
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
        "source": source,
        "hist_n": hist_n,
        "cwd": cwd,
        "prompt": prompt,
        "command": command,
        "suggested_command": suggested_command,
        "output": output,
        "relevant_ids": relevant_ids if relevant_ids is not None else []
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn) + "\n")


def get_last_user_hist_n() -> int:
    """
    The highest shell-history index (`hist_n`) among already-imported `source:"user"` turns, or 0.
    Used as the de-dup high-water mark when syncing the user's recent shell commands - we only import
    commands with a higher index. Lives in the history itself, so `clear_history` resets it.
    """
    path = get_history_file_path()
    if not path.exists():
        return 0
    last = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turn = json.loads(line)
                if turn.get("source") == "user" and isinstance(turn.get("hist_n"), int):
                    last = max(last, turn["hist_n"])
    except Exception:
        return 0
    return last

def get_history_metadata(limit: int = 10, session_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Retrieves metadata (id, prompt, command) for the last N turns. Omits execution outputs to keep
    context token count minimal. `session_dir` defaults to the current session; pass another
    session's dir to read its metadata (cross-session referencing).
    """
    metadata = []
    for turn in _read_turns(get_history_file_path(session_dir)):
        try:
            metadata.append({
                "id": turn["id"],
                "source": turn.get("source", "doit"),
                "cwd": turn.get("cwd"),
                "prompt": turn["prompt"],
                "command": turn["command"],
                "suggested_command": turn.get("suggested_command", "")
            })
        except Exception:
            continue
    return metadata[-limit:]

def get_full_turns(ids: List[int], session_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Retrieves full turn logs (including execution outputs) for specific target IDs, in the order of
    the requested IDs. `session_dir` defaults to the current session; pass another session's dir to
    read its turns (cross-session referencing).
    """
    if not ids:
        return []
    id_set = set(ids)
    matched_turns = {t["id"]: t for t in _read_turns(get_history_file_path(session_dir))
                     if t.get("id") in id_set}
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


def get_latest_output_turn_id(session_dir: Optional[Path] = None) -> Optional[int]:
    """
    The id of the most recent turn that ran a command AND has real captured output - i.e. a doit turn
    (user turns only store a status placeholder, never real output). Used to GUARANTEE that a
    follow-up question about "the previous output / that command / why it failed" has the actual
    output in context, even if the LLM reference resolver linked nothing (or an output-less user
    command among the noise). None when there is no such turn.
    """
    latest = None
    for t in _read_turns(get_history_file_path(session_dir)):
        if t.get("source", "doit") == "doit" and t.get("command") and t.get("output"):
            latest = t.get("id")
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


# --- Session registry (multi-window awareness) -----------------------------------------------------
# Each session writes a small session.json into its OWN folder, so other windows are discoverable
# (with cwd + recency) for explicit cross-session references. Writes are per-session -> no contention.

def _session_meta_path(session_dir: Optional[Path] = None) -> Path:
    return (session_dir or get_session_dir()) / "session.json"


def write_session_meta(cwd: Optional[str] = None) -> None:
    """Record/refresh THIS session's registry entry {pid, cwd, created_at, last_active_at}. Called at
    the top of each run; best-effort (never raises into the caller)."""
    try:
        sid = current_session_id()
        path = _session_meta_path(get_session_dir(sid))
        now = datetime.now(timezone.utc).isoformat()
        created = now
        if path.exists():
            try:
                created = json.loads(path.read_text(encoding="utf-8")).get("created_at", now)
            except Exception:
                pass
        if cwd is None:
            try:
                cwd = os.getcwd()
            except Exception:
                cwd = "(unknown)"
        meta = {"pid": sid, "cwd": cwd, "created_at": created, "last_active_at": now}
        tmp = path.parent / (path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, path)
    except Exception:
        pass


def read_session_meta(session_dir: Path) -> Dict[str, Any]:
    """The {pid, cwd, created_at, last_active_at} for a session folder, or {} if missing/unreadable."""
    path = _session_meta_path(session_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pid_is_live(pid: str) -> Optional[bool]:
    """True/False if the shell PID is still a running process; None when we can't tell (no /proc,
    e.g. macOS/Windows) so callers fall back to recency."""
    try:
        if not Path("/proc").exists():
            return None
        return Path(f"/proc/{pid}").exists()
    except Exception:
        return None


def _recent_turn_summary(session_dir: Path, n: int = 6) -> List[str]:
    """A short 'id, who, prompt -> command' digest of a session's last few command/answer turns, for
    the cross-session resolver (which returns the relevant ids) and the clarify menu."""
    out: List[str] = []
    for t in _read_turns(get_history_file_path(session_dir)):
        cmd = t.get("command") or t.get("suggested_command")
        if not cmd:
            continue
        prompt = (t.get("prompt") or "").strip()
        who = "user" if t.get("source") == "user" else "doit"
        out.append(f"[id {t.get('id')}, {who}] {prompt + ' -> ' if prompt else ''}{cmd}")
    return out[-n:]


def list_sessions(include_current: bool = True, recent_turns: int = 3) -> List[Dict[str, Any]]:
    """
    All known sessions (one per `.doit/history_*/` folder), newest-active first. Each entry:
    {pid, cwd, last_active_at, alive (bool|None), is_current, recent: [..]}. `alive` distinguishes a
    still-open window from a folder left by a closed one (None where we can't check). Used by the
    "list the shell numbers" route and cross-session resolution.
    """
    base = doit_root()
    if not base.exists():
        return []
    cur = current_session_id()
    sessions: List[Dict[str, Any]] = []
    try:
        candidates = sorted(base.glob("history_*"))
    except Exception:
        candidates = []
    for d in candidates:
        if not d.is_dir():
            continue
        sid = d.name[len("history_"):]
        if not sid:
            continue
        meta = read_session_meta(d)
        sessions.append({
            "pid": sid,
            "cwd": meta.get("cwd"),
            "last_active_at": meta.get("last_active_at"),
            "alive": _pid_is_live(sid),
            "is_current": sid == cur,
            "recent": _recent_turn_summary(d, recent_turns),
        })
    if not include_current:
        sessions = [s for s in sessions if not s["is_current"]]
    sessions.sort(key=lambda s: s.get("last_active_at") or "", reverse=True)
    return sessions


def list_other_sessions(recent_turns: int = 3) -> List[Dict[str, Any]]:
    """`list_sessions` minus the current session - the candidates for a cross-session reference."""
    return list_sessions(include_current=False, recent_turns=recent_turns)
