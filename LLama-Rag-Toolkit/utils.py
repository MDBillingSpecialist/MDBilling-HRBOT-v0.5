import hashlib
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_llm(openai_client, query, model="gpt-4"):
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        raise

def generate_doc_id(file_content):
    return hashlib.md5(file_content).hexdigest()

def get_current_time():
    return datetime.datetime.now().isoformat()