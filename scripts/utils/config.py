import json
from pathlib import Path
# TODO: implement config loading  (better with default)

def load_config(config_path):
    """Load configuration from config file with default fallbacks."""
    
    # Load config from file if it exists
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            # Update defaults with values from config file
            config = file_config
            print(f"Loaded configuration from: {config_file}")
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration values")
    else:
        raise FileNotFoundError(f"Config file {config_file} not found")
    
    return config