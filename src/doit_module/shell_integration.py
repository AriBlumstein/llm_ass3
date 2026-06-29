"""
First-run shell-integration bootstrap.

Persisting `cd`/`export` requires the `doit` SHELL FUNCTION (in src/doit-init.sh) to be loaded by
the user's shell - a program on PATH can't change its parent shell. This module, on an
interactive run that is NOT already going through that function, offers to add the one `source`
line to the user's shell rc. It keeps asking on each run until the user accepts (or until the line
is present). It never edits anything in a non-interactive context (scripts/CI), and only supports
bash and zsh.

Limitation it can't escape: the run that installs the line is still the bare executable, so THAT
invocation can't persist - persistence begins in the next shell.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def _interactive() -> bool:
    """True only when both stdin and stdout are TTYs (so we can safely prompt)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _init_script_path() -> Path:
    """Absolute path to src/doit-init.sh (this file lives in src/doit_module/)."""
    return Path(__file__).resolve().parent.parent / "doit-init.sh"


def _rc_path_for_shell(shell_name: str) -> Optional[Path]:
    """
    The shell rc file to install into, or None for an unsupported shell. bash and zsh only. On
    Git Bash/MSYS a login shell reads ~/.bash_profile; on Linux/WSL interactive shells read
    ~/.bashrc - prefer whichever already exists, else create ~/.bashrc.
    """
    home = Path.home()
    if "zsh" in shell_name:
        return home / ".zshrc"
    if "bash" in shell_name:
        bashrc = home / ".bashrc"
        if bashrc.exists():
            return bashrc
        bash_profile = home / ".bash_profile"
        if bash_profile.exists():
            return bash_profile
        return bashrc
    return None


def ensure_shell_integration(input_fn=input) -> None:
    """
    Offer to install the doit shell-function integration. No-op when already integrated, when
    non-interactive, or when the shell is unsupported. Best-effort: never raise into the caller.
    """
    try:
        # Already running through the shell function -> integrated, nothing to do.
        if os.environ.get("DOIT_CD_FILE"):
            return
        # Never prompt or edit dotfiles in a non-interactive context (pipes, scripts, CI).
        if not _interactive():
            return

        shell = os.path.basename(os.environ.get("SHELL", "")).lower()
        rc = _rc_path_for_shell(shell)
        if rc is None:
            return  # unsupported shell (fish, etc.) - stay quiet rather than nag

        init = _init_script_path()
        try:
            existing = rc.read_text(encoding="utf-8") if rc.exists() else ""
        except Exception:
            return

        if str(init) in existing:
            # Already accepted; just not loaded in THIS shell yet. Remind, don't re-ask.
            print(f"[doit] Shell integration is installed but not loaded in this shell. "
                  f"Open a new terminal, or run: source {rc}")
            return

        # Ask. If declined, we write nothing, so this prompt returns on the next run too.
        print("[doit] Persistent navigation (cd / export) AND user-awareness (so doit can see your "
              "recent terminal commands) need a one-line shell hook.")
        try:
            answer = input_fn(f"      Add it to {rc} now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if answer in ("n", "no"):
            print(f"[doit] Skipped (I'll ask again next time). To enable it yourself, add:\n"
                  f"       source {init}")
            return

        line = (
            "\n# doit shell integration (persistent cd / export / alias) — added by doit\n"
            f'[ -f "{init}" ] && source "{init}"\n'
        )
        try:
            with open(rc, "a", encoding="utf-8") as f:
                f.write(line)
            print(f"[doit] Installed into {rc}. Open a new terminal (or run: source {rc}) to "
                  "enable persistent navigation.")
        except Exception as e:
            print(f"[doit] Could not write {rc}: {e}")
    except Exception:
        # Bootstrap must never break a normal run.
        return
