import os
import logging
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"

def initialize_settings():
    Settings.llm = OpenAI(model=DEFAULT_MODEL)
    Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env file.")
    
    logger.info(f"LLM initialized with model: {Settings.llm.model}")

def get_current_llm_model():
    return Settings.llm.model if Settings.llm else None