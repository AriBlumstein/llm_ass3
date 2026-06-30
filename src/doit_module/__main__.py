#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from llm_communicator import history_manager
from llm_communicator.llm_bash import BashToolAgent, BashSafetyViolationError
from doit_module.shell_integration import ensure_shell_integration


def reset_all_histories() -> int:
    """
    Wipe EVERY session's history by deleting all `<repo>/.doit/history_*` folders (each holds a
    session's `doit.jsonl`, `cmdlog.tsv`, and `session.json`). Memories live in
    `<repo>/.doit/memories.json` - a sibling of these folders, NOT inside them - so they are left
    completely untouched. Returns how many session folders were removed.
    """
    removed = 0
    for session_dir in history_manager.doit_root().glob("history_*"):
        if session_dir.is_dir():
            shutil.rmtree(session_dir, ignore_errors=True)
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(
        prog="doit",
        description="Translate natural language to shell commands and execute them."
    )
    parser.add_argument(
        "-n", "--new",
        action="store_true",
        help="Delete all history and start a new session"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ALL sessions' histories (every window), but keep memories. Then exit."
    )
    parser.add_argument(
        "instruction",
        type=str,
        nargs="?",
        default=None,
        help="The natural language instruction to execute"
    )
    args = parser.parse_args()

    # --reset is a standalone maintenance action: wipe EVERY session's history (not just this
    # window's) while leaving memories intact, then exit. Handled before the instruction check so it
    # works on its own.
    if args.reset:
        try:
            removed = reset_all_histories()
            print(f"Reset complete: cleared history for {removed} session(s). Memories were kept.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)

    # `instruction` is required EXCEPT with -n/--new (which may be used alone to just clear history).
    if not args.instruction:
        if args.new:
            try:
                BashToolAgent(force_new=True)
                print("Session history cleared and new session started.")
                sys.exit(0)
            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)
        # No instruction and no -n: emit argparse's standard required-argument error (exit 2).
        parser.error("the following arguments are required: instruction")

    # On an interactive run that isn't already going through the doit shell function, offer to
    # install the integration that makes cd/export persist. Keeps asking until accepted; no-op when
    # already integrated or non-interactive.
    ensure_shell_integration()

    try:
        llm = BashToolAgent(force_new=args.new)
        llm.run_single(args.instruction)
    except BashSafetyViolationError as e:
        print(f"The command is not safe to execute: {e}")
    except subprocess.TimeoutExpired:
        print("Command execution terminated due to exceeding 20.0s timeout limit")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()