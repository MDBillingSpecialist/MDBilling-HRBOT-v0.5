import streamlit as st
from config import load_config
from ui import (
    initialize_session_state,
    handle_upload_documents,
    handle_parse_documents,
    handle_create_knowledge_base,
    handle_generate_questions_and_evaluate,
)
import sys
import os

def main():
    st.set_page_config(page_title="LLM-RAG Document Processing")
    openai_client, logger = load_config()
    st.session_state['openai_client'] = openai_client

    st.title("Document Processing and Q&A Evaluation System")
    st.sidebar.header("Navigation")

    initialize_session_state()

    step = st.sidebar.radio("Select Step", [
        "1. Upload Documents",
        "2. Parse Documents",
        "3. Create Knowledge Base",
        "4. Generate Questions and Evaluate"
    ])

    if step == "1. Upload Documents":
        handle_upload_documents()
    elif step == "2. Parse Documents":
        handle_parse_documents()
    elif step == "3. Create Knowledge Base":
        handle_create_knowledge_base()
    elif step == "4. Generate Questions and Evaluate":
        handle_generate_questions_and_evaluate()

if __name__ == '__main__':
    main()
