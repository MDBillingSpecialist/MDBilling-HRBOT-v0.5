import streamlit as st
from .document_ui import render_upload_documents
from .kb_ui import render_create_knowledge_base
from .query_ui import render_query_rag
from .question_generation_ui import render_generate_questions
from .evaluation_ui import render_evaluate_rag

def initialize_session():
    if 'step' not in st.session_state:
        st.session_state.step = 'Upload Documents'

def render_ui():
    st.title("LLama RAG Toolkit")
    
    st.sidebar.title("Navigation")
    step = st.sidebar.radio(
        "Go to",
        ('Upload Documents', 'Create Knowledge Base', 'Generate Questions', 'Query RAG', 'Evaluate RAG')
    )
    
    if step == 'Upload Documents':
        render_upload_documents()
    elif step == 'Create Knowledge Base':
        render_create_knowledge_base()
    elif step == 'Generate Questions':
        render_generate_questions()
    elif step == 'Query RAG':
        render_query_rag()
    elif step == 'Evaluate RAG':
        render_evaluate_rag()

if __name__ == "__main__":
    render_ui()