import os
import tempfile
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from utils import generate_doc_id, get_current_time

def upload_document(uploaded_file):
    file_content = uploaded_file.read()
    doc_id = generate_doc_id(file_content)

    if doc_id not in st.session_state['documents']:
        st.session_state['documents'][doc_id] = {
            'content': file_content,
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'upload_time': get_current_time()
        }
        st.session_state['metrics']['uploaded_docs'] += 1
        return True
    return False

def parse_document(doc_id, doc_info):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, doc_info['name'])
            with open(file_path, 'wb') as f:
                f.write(doc_info['content'])
            reader = SimpleDirectoryReader(input_dir=tmpdirname)
            documents = reader.load_data()
            st.session_state['parsed_documents'][doc_id] = documents
            st.session_state['metrics']['parsed_docs'] += 1
            st.session_state['metrics']['total_chunks'] += len(documents)
            return True
    except Exception as e:
        st.error(f"Failed to parse '{doc_info['name']}': {e}")
        return False