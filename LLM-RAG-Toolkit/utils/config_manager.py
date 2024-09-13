import os
import logging
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load environment variables
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

        # Load additional configurations
        self.max_segment_tokens = self.config.get("max_segment_tokens", 500)
        self.overlap_sentences = self.config.get("overlap_sentences", 1)
        self.file_paths = self.config.get("file_paths", [])

        # Ensure critical variables are set
        self._validate_config()

    def _validate_config(self):
        critical_vars = ["OPENAI_API_KEY", "max_segment_tokens", "overlap_sentences", "file_paths"]
        for var in critical_vars:
            if not getattr(self, var):
                raise ValueError(f"Critical configuration {var} is not set.")

    def __getattr__(self, name):
        return self.config.get(name)

def load_config(config_path="config/config.yaml"):
    config = Config(config_path)
    
    # Ensure the logging directory exists
    log_dir = os.path.dirname(config.logging['file'])
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if config.DEBUG else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logging['file']),
            logging.StreamHandler()
        ]
    )

    return config

# Create a global config object
config = load_config()
