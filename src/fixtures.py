import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the path to the root directory's .env file relative to this script
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
MODEL_NAME = "gpt-4o-mini"
