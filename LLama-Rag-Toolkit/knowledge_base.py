import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
import json
import os
import chromadb
from tqdm import tqdm
from api_logger import add_api_call
import pickle

# Set up global settings
Settings.llm = OpenAI(model="gpt-4o-mini")  # Updated to use gpt-4o-mini
Settings.embed_model = OpenAIEmbedding()

def create_knowledge_base():
    if 'parsed_documents' not in st.session_state or not st.session_state['parsed_documents']:
        print("No knowledge base nodes found")
        return False

    transformations = create_transformations()

    try:
        nodes = st.session_state['parsed_documents']

        for transformation in transformations:
            nodes = transformation.process_nodes(nodes)
            # Log the API call
            add_api_call(f"Processing nodes with {transformation.__class__.__name__}", f"Processed {len(nodes)} nodes")

        # Create a persistent Chroma client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        st.session_state['knowledge_base'] = VectorStoreIndex(nodes, storage_context=storage_context)
        # Log the API call
        add_api_call("Creating VectorStoreIndex", f"Created index with {len(nodes)} nodes")
        
        print(f"Knowledge base created successfully with {len(nodes)} nodes")
        return True
    except Exception as e:
        print(f"Error creating knowledge base: {e}")
        return False

def create_transformations():
    transformations = [
        TitleExtractor(nodes=5),
        KeywordExtractor(keywords=10),
        QuestionsAnsweredExtractor(questions=3),
    ]
    return transformations

def save_knowledge_base_summary(nodes):
    summary = {
        "total_nodes": len(nodes),
        "document_summaries": []
    }
    
    for node in nodes:
        summary["document_summaries"].append({
            "node_id": node.node_id,
            "chunk": node.get_content()[:100] + "...",
            "metadata": node.metadata,
        })
    
    os.makedirs("knowledge_base", exist_ok=True)
    with open("knowledge_base/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def export_knowledge_base():
    if st.session_state.get('knowledge_base') is None:
        st.error("No knowledge base to export.")
        return False

    export_dir = "exported_knowledge_base"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    try:
        st.session_state['knowledge_base'].storage_context.persist(persist_dir=export_dir)
        st.session_state['knowledge_base'].save_to_disk(export_dir)
        st.success(f"Knowledge base exported to {export_dir}")
        return export_dir
    except Exception as e:
        st.error(f"Error exporting knowledge base: {str(e)}")
        return False

def load_knowledge_base(path):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(storage_context)
        return index
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return None

def create_rag_query_engine(knowledge_base, model="gpt-4o-mini"):
    """Creates a RAG query engine from the knowledge base."""
    llm = OpenAI(model=model)
    query_engine = knowledge_base.as_query_engine(
        retriever_mode="default",
        response_mode="compact",
        llm=llm
    )
    return query_engine

def query_knowledge_base(query_engine, query):
    """Queries the knowledge base using the RAG query engine."""
    response = query_engine.query(query)
    return response

def save_knowledge_base_state(filename="knowledge_base_state.pkl"):
    """Save the entire knowledge base state."""
    state_to_save = {
        'knowledge_base': st.session_state.get('knowledge_base'),
        'parsed_documents': st.session_state.get('parsed_documents'),
        'synthetic_qa_pairs': st.session_state.get('synthetic_qa_pairs'),
        'rag_dataset': st.session_state.get('rag_dataset')
    }
    with open(filename, 'wb') as f:
        pickle.dump(state_to_save, f)
    st.success(f"Knowledge base state saved to {filename}")

def load_knowledge_base_state(filename="knowledge_base_state.pkl"):
    """Load the entire knowledge base state."""
    try:
        with open(filename, 'rb') as f:
            loaded_state = pickle.load(f)
        for key, value in loaded_state.items():
            st.session_state[key] = value
        st.success(f"Knowledge base state loaded from {filename}")
        return True
    except FileNotFoundError:
        st.error(f"No saved state found at {filename}")
        return False