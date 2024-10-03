import streamlit as st
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from utils import generate_doc_id, get_current_time
import tempfile
import os

def process_documents(uploaded_files):
    if not uploaded_files:
        st.warning("Please upload some documents.")
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())

        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()

    # Convert to LlamaIndex Document objects
    llama_documents = [
        Document(
            text=doc.text,
            metadata={
                "file_name": doc.metadata.get("file_name", ""),
                "doc_id": generate_doc_id(doc.text),
                "created_at": get_current_time()
            }
        ) for doc in documents
    ]

    return llama_documents

def display_parsed_documents():
    if 'parsed_documents' in st.session_state and st.session_state['parsed_documents']:
        st.subheader("Parsed Documents")
        for i, doc in enumerate(st.session_state['parsed_documents']):
            st.write(f"Document {i+1}:")
            st.json({
                "text": doc.text[:100] + "...",
                "doc_id": doc.metadata.get('doc_id', 'N/A'),
                "file_name": doc.metadata.get('file_name', 'N/A'),
            })

def display_parser_config():
    st.sidebar.subheader("Document Parser Configuration")
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    return chunk_size, chunk_overlap