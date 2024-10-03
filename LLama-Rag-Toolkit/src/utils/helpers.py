import hashlib
from datetime import datetime
import re

def generate_doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def get_current_time() -> str:
    return datetime.now().isoformat()

def sanitize_name(name):
    # Remove any non-alphanumeric characters and convert to lowercase
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
    # Hash the sanitized name to ensure uniqueness
    hashed_name = hashlib.md5(sanitized.encode()).hexdigest()
    return f"kb_{hashed_name[:10]}"