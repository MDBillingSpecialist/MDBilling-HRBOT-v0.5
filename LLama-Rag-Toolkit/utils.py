import hashlib
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import logging
import streamlit as st
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_llm(query, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        raise

def generate_doc_id(file_content):
    return hashlib.md5(file_content.encode('utf-8')).hexdigest()

def get_current_time():
    return datetime.datetime.now().isoformat()

def initialize_session_state():
    if 'documents' not in st.session_state:
        st.session_state['documents'] = {}
    if 'parsed_documents' not in st.session_state:
        st.session_state['parsed_documents'] = None
    if 'knowledge_base' not in st.session_state:
        st.session_state['knowledge_base'] = None
    if 'dataset' not in st.session_state:
        st.session_state['dataset'] = None
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = {'uploaded_docs': 0, 'parsed_docs': 0, 'total_chunks': 0, 'kb_size': 0}