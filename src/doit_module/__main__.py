#!/usr/bin/env python3

import argparse
import subprocess
from llm_communicator.openai_bash import BashToolAgent, BashSafetyViolationError



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")

    args = parser.parse_args()

    max_iter = args.max_iterations

    
    try:
        llm = BashToolAgent()
        llm.run_agent_loop(max_iterations=max_iter)
    except BashSafetyViolationError:
        print("The command is not safe to execute")
    except subprocess.TimeoutExpired:
        print("Command execution terminated due to exceeding 15.0s timeout limit")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

    