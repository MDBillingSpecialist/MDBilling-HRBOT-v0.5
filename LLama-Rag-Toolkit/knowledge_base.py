import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import json
import os

def create_knowledge_base():
    all_nodes = []
    for doc_list in st.session_state['parsed_documents'].values():
        all_nodes.extend(doc_list)

    vector_store = ChromaVectorStore(chroma_collection=st.session_state['chroma_collection'])
    index = VectorStoreIndex.from_documents(all_nodes, vector_store=vector_store)
    st.session_state['knowledge_base'] = index
    st.session_state['metrics']['kb_size'] = len(all_nodes)
    
    # Save knowledge base summary
    save_knowledge_base_summary(all_nodes)
    
    return True

def save_knowledge_base_summary(nodes):
    summary = {
        "total_nodes": len(nodes),
        "document_summaries": []
    }
    
    for node in nodes:
        summary["document_summaries"].append({
            "doc_id": node.doc_id,
            "chunk": node.text[:100] + "...",  # First 100 characters of each chunk
            "metadata": node.metadata
        })
    
    os.makedirs("knowledge_base", exist_ok=True)
    with open("knowledge_base/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def export_knowledge_base():
    if 'knowledge_base' not in st.session_state or st.session_state['knowledge_base'] is None:
        st.error("Knowledge base has not been created yet.")
        return

    # Export the index
    st.session_state['knowledge_base'].storage_context.persist("knowledge_base/index")
    
    st.success("Knowledge base exported successfully.")
    return "knowledge_base/index"