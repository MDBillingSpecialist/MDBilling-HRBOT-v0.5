import streamlit as st
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from utils import generate_doc_id, get_current_time
import tempfile
import os

def process_documents(uploaded_files, buffer_size=1, num_children=5):
    if not uploaded_files:
        st.warning("Please upload some documents.")
        return None

    all_nodes = []

    # Create a semantic node parser
    embed_model = OpenAIEmbedding()
    node_parser = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        embed_model=embed_model,
        num_children=num_children
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())

            reader = SimpleDirectoryReader(input_dir=temp_dir)
            documents = reader.load_data()
            
            for doc in documents:
                doc_id = generate_doc_id(doc.text)
                if 'documents' not in st.session_state:
                    st.session_state['documents'] = {}
                
                if doc_id not in st.session_state['documents']:
                    st.session_state['documents'][doc_id] = {
                        'content': doc.text,
                        'name': uploaded_file.name,
                        'size': len(doc.text),
                        'upload_time': get_current_time()
                    }
                    if 'metrics' not in st.session_state:
                        st.session_state['metrics'] = {'uploaded_docs': 0, 'parsed_docs': 0, 'total_chunks': 0, 'kb_size': 0}
                    st.session_state['metrics']['uploaded_docs'] += 1

                    # Parse the document into nodes
                    nodes = node_parser.get_nodes_from_documents([doc])
                    
                    # Add metadata to each node
                    for node in nodes:
                        node.metadata['doc_id'] = doc_id
                        node.metadata['file_name'] = uploaded_file.name

                    all_nodes.extend(nodes)

    if 'metrics' in st.session_state:
        st.session_state['metrics']['total_chunks'] = len(all_nodes)
    
    st.info(f"Documents processed. Total nodes: {len(all_nodes)}")
    return all_nodes

def display_parsed_documents():
    if 'parsed_documents' in st.session_state and st.session_state['parsed_documents']:
        st.subheader("Parsed Documents")
        for i, node in enumerate(st.session_state['parsed_documents']):
            st.write(f"Node {i+1}:")
            st.json({
                "text": node.text[:100] + "...",
                "doc_id": node.metadata.get('doc_id', 'N/A'),
                "file_name": node.metadata.get('file_name', 'N/A'),
            })

def display_parser_config():
    st.sidebar.subheader("Document Parser Configuration")
    buffer_size = st.sidebar.slider("Buffer Size", min_value=1, max_value=10, value=1, help="Number of sentences to look at when deciding where to split")
    num_children = st.sidebar.slider("Number of Children", min_value=1, max_value=20, value=5, help="Maximum number of child nodes that can be created from a parent node")
    return buffer_size, num_children