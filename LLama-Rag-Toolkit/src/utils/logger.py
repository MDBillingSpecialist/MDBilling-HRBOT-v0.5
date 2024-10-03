import logging
from typing import Dict

class StreamlitLogger(logging.Logger):
    def __init__(self):
        super().__init__("StreamlitLogger")
        self.logs = []
        self.progress = 0
        self.api_usage = {"total_tokens": 0, "total_cost": 0}

    def info(self, msg):
        self.logs.append(f"INFO: {msg}")
        super().info(msg)

    def warning(self, msg):
        self.logs.append(f"WARNING: {msg}")
        super().warning(msg)

    def error(self, msg):
        self.logs.append(f"ERROR: {msg}")
        super().error(msg)

    def get_logs(self):
        return "\n".join(self.logs)

    def set_progress(self, progress):
        self.progress = progress

    def get_progress(self):
        return self.progress

    def log_api_usage(self, tokens: int, cost: float):
        self.api_usage["total_tokens"] += tokens
        self.api_usage["total_cost"] += cost
        self.info(f"API call: {tokens} tokens used, cost: ${cost:.4f}")

    def get_api_usage(self) -> Dict[str, float]:
        return self.api_usage

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger