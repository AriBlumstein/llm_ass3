#!/usr/bin/env python3

import argparse
import subprocess
import sys
from llm_communicator.llm_bash import BashToolAgent, BashSafetyViolationError
from doit_module.shell_integration import ensure_shell_integration


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
        "instruction",
        type=str,
        nargs="?",
        default=None,
        help="The natural language instruction to execute"
    )
    args = parser.parse_args()

    if not args.instruction:
        if args.new:
            try:
                BashToolAgent(force_new=True)
                print("Session history cleared and new session started.")
                sys.exit(0)
            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)

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