#!/usr/bin/env python3

import argparse
import subprocess
import sys
from llm_communicator.llm_bash import BashToolAgent, BashSafetyViolationError


def main():
    parser = argparse.ArgumentParser(description="Translate natural language to shell commands and execute them.")
    parser.add_argument("instruction", type=str, nargs="?", help="The natural language instruction to execute")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")

    args = parser.parse_args()

    max_iter = args.max_iterations

    try:
        llm = BashToolAgent()
        if args.instruction:
            llm.run_single(args.instruction)
        else:
            llm.run_agent_loop(max_iterations=max_iter)
    except BashSafetyViolationError as e:
        print(f"The command is not safe to execute: {e}")
    except subprocess.TimeoutExpired:
        print("Command execution terminated due to exceeding 15.0s timeout limit")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()