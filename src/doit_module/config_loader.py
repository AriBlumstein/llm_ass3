import os
from pathlib import Path
import configparser
from typing import Optional, Tuple

# We can import MODEL_NAME from fixtures as a fallback default
try:
    from fixtures import MODEL_NAME
except ImportError:
    MODEL_NAME = "gpt-5.4-nano"

DEFAULT_MODEL = MODEL_NAME
DEFAULT_TOOL_CALLING = True

def load_config(config_path: Optional[Path] = None) -> Tuple[str, Optional[str], bool]:
    """
    Loads model configuration from doit.cfg in the project root directory.
    
    Returns:
        Tuple[model_name, api_base, tool_calling]
        - model_name (str): LiteLLM model name (e.g. 'openai/gpt-4o-mini', 'ollama/qwen3:4b-instruct')
        - api_base (str | None): Custom API base URL if specified, else None
        - tool_calling (bool): Whether the model supports native function calling
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "doit.cfg"
    
    # Fallbacks
    model_name = DEFAULT_MODEL
    api_base = None
    tool_calling = DEFAULT_TOOL_CALLING
    
    if config_path.exists():
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            if "model" in config:
                model_section = config["model"]
                model_name = model_section.get("name", fallback=DEFAULT_MODEL).strip()
                api_base_val = model_section.get("api_base", fallback=None)
                if api_base_val:
                    api_base = api_base_val.strip()
                tool_calling = config.getboolean("model", "tool_calling", fallback=DEFAULT_TOOL_CALLING)
        except Exception as e:
            # If parsing fails, print warning and use defaults
            print(f"Warning: Failed to parse {config_path}: {e}. Using defaults.", flush=True)
            
    return model_name, api_base, tool_calling
