import os
from dotenv import load_dotenv
import logging
import sys
from openai import OpenAI as OpenAIClient
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

def load_config():
    load_dotenv()
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables.")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    Settings.llm = OpenAI(model="gpt-4")
    Settings.embed_model = OpenAIEmbedding()

    return OpenAIClient(api_key=OPENAI_API_KEY), logger