import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
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
import sqlite3
from datetime import datetime
import re
import hashlib
import shutil

# Set up global settings
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(api_key=os.environ.get("OPENAI_API_KEY"))

def sanitize_name(name):
    # Remove any non-alphanumeric characters and convert to lowercase
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
    # Hash the sanitized name to ensure uniqueness
    hashed_name = hashlib.md5(sanitized.encode()).hexdigest()
    return f"kb_{hashed_name[:10]}"

def create_knowledge_base(name, progress_callback=None, chunk_size=1000, chunk_overlap=200):
    if 'parsed_documents' not in st.session_state or not st.session_state['parsed_documents']:
        raise ValueError("No parsed documents found. Please process documents first.")

    try:
        documents = st.session_state['parsed_documents']
        total_docs = len(documents)

        sanitized_name = sanitize_name(name)

        # Create a persistent Chroma client
        chroma_client = chromadb.PersistentClient(path=f"./chroma_db_{sanitized_name}")
        chroma_collection = chroma_client.create_collection(sanitized_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create a semantic splitter node parser
        node_parser = SemanticSplitterNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            llm=Settings.llm
        )

        # Create transformations
        transformations = create_transformations()

        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=transformations,
            node_parser=node_parser,
        )

        for i, doc in enumerate(tqdm(documents, desc="Processing documents")):
            # Process the document through the ingestion pipeline
            nodes = pipeline.run(documents=[doc])
            
            # Add the nodes to the vector store
            vector_store.add(nodes)
            
            if progress_callback:
                progress = (i + 1) / total_docs * 100
                progress_callback(progress)

        knowledge_base = VectorStoreIndex.from_vector_store(vector_store)
        
        # Save a summary of the knowledge base
        save_knowledge_base_summary(knowledge_base.docstore.docs.values())
        
        print(f"Knowledge base '{name}' created successfully with {total_docs} documents")
        return knowledge_base
    except Exception as e:
        print(f"Error creating knowledge base: {str(e)}")
        raise

def create_transformations():
    transformations = [
        TitleExtractor(nodes=5, llm=Settings.llm),
        KeywordExtractor(keywords=10, llm=Settings.llm),
        QuestionsAnsweredExtractor(questions=3, llm=Settings.llm),
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

def export_knowledge_base(knowledge_base, name):
    export_dir = f"exported_knowledge_base_{name}"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    try:
        knowledge_base.storage_context.persist(persist_dir=export_dir)
        knowledge_base.save_to_disk(export_dir)
        st.success(f"Knowledge base '{name}' exported to {export_dir}")
        return export_dir
    except Exception as e:
        st.error(f"Error exporting knowledge base: {str(e)}")
        return False

def load_knowledge_base(name):
    sanitized_name = sanitize_name(name)
    chroma_path = f"chroma_db_{sanitized_name}"
    export_path = f"exported_knowledge_base_{sanitized_name}"
    
    try:
        # Check for directories with partial matches
        chroma_dirs = [d for d in os.listdir() if d.startswith("chroma_db_")]
        export_dirs = [d for d in os.listdir() if d.startswith("exported_knowledge_base_")]
        
        st.write(f"Debug: All Chroma directories: {chroma_dirs}")
        st.write(f"Debug: All export directories: {export_dirs}")
        
        # Use partial matching instead of exact matching
        matching_chroma = [d for d in chroma_dirs if name in d or sanitized_name in d]
        matching_export = [d for d in export_dirs if name in d or sanitized_name in d]
        
        st.write(f"Debug: Matching Chroma directories: {matching_chroma}")
        st.write(f"Debug: Matching export directories: {matching_export}")
        
        if matching_chroma:
            chroma_path = matching_chroma[0]
            sanitized_name = chroma_path[len("chroma_db_"):]
        elif matching_export:
            export_path = matching_export[0]
            sanitized_name = export_path[len("exported_knowledge_base_"):]
        else:
            # If no match found, try to find the closest match
            all_dirs = chroma_dirs + export_dirs
            closest_match = min(all_dirs, key=lambda x: len(set(x) - set(name)))
            st.write(f"Debug: No exact match found. Closest match: {closest_match}")
            if closest_match.startswith("chroma_db_"):
                chroma_path = closest_match
                sanitized_name = chroma_path[len("chroma_db_"):]
            else:
                export_path = closest_match
                sanitized_name = export_path[len("exported_knowledge_base_"):]
        
        st.write(f"Debug: Final Chroma path: {chroma_path}")
        st.write(f"Debug: Final export path: {export_path}")
        
        if os.path.exists(chroma_path):
            # Load from Chroma
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            chroma_collection = chroma_client.get_collection(sanitized_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            st.success(f"Loaded knowledge base from Chroma: {chroma_path}")
        elif os.path.exists(export_path):
            # Load from exported directory
            storage_context = StorageContext.from_defaults(persist_dir=export_path)
            index = load_index_from_storage(storage_context)
            st.success(f"Loaded knowledge base from exported directory: {export_path}")
        else:
            raise FileNotFoundError(f"Knowledge base not found in Chroma ({chroma_path}) or exported directory ({export_path})")
        
        # Update last_modified in database
        conn = sqlite3.connect('knowledge_bases.db')
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute("UPDATE knowledge_bases SET last_modified = ? WHERE name = ?", (now, name))
        conn.commit()
        conn.close()
        
        return index
    except Exception as e:
        error_msg = f"Error loading knowledge base: {str(e)}"
        st.error(error_msg)
        st.write(f"Detailed error: {e}")
        st.write(f"Current working directory: {os.getcwd()}")
        st.write("Directories in current working directory:")
        st.write(list_directories())
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

def save_knowledge_base(knowledge_base, name):
    """Saves the knowledge base to disk and updates the database."""
    export_dir = f"exported_knowledge_base_{name}"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    try:
        knowledge_base.storage_context.persist(persist_dir=export_dir)
        knowledge_base.save_to_disk(export_dir)
        
        # Update database
        conn = sqlite3.connect('knowledge_bases.db')
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute("INSERT OR REPLACE INTO knowledge_bases (name, created_at, last_modified) VALUES (?, ?, ?)",
                  (name, now, now))
        conn.commit()
        conn.close()
        
        st.success(f"Knowledge base '{name}' saved to {export_dir}")
        return export_dir
    except Exception as e:
        st.error(f"Error saving knowledge base: {str(e)}")
        return False

def create_advanced_rag_query_engine(knowledge_base, model="gpt-4o", retriever_mode="hybrid"):
    """Creates an advanced RAG query engine from the knowledge base using GPT-4."""
    llm = OpenAI(model=model)
    query_engine = knowledge_base.as_query_engine(
        retriever_mode=retriever_mode,
        response_mode="tree_summarize",
        llm=llm,
        verbose=True
    )
    return query_engine

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect('knowledge_bases.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_bases
                 (id INTEGER PRIMARY KEY, name TEXT, created_at TEXT, last_modified TEXT)''')
    conn.commit()
    conn.close()

def delete_knowledge_base(name):
    """Deletes a knowledge base from disk and database."""
    try:
        shutil.rmtree(f"exported_knowledge_base_{name}")
        shutil.rmtree(f"./chroma_db_{name}")
        
        # Remove from database
        conn = sqlite3.connect('knowledge_bases.db')
        c = conn.cursor()
        c.execute("DELETE FROM knowledge_bases WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        
        st.success(f"Knowledge base '{name}' deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting knowledge base: {str(e)}")

def get_all_knowledge_bases():
    """Retrieves all knowledge bases from the database."""
    conn = sqlite3.connect('knowledge_bases.db')
    c = conn.cursor()
    c.execute("SELECT name, created_at, last_modified FROM knowledge_bases")
    knowledge_bases = c.fetchall()
    conn.close()
    return knowledge_bases

def sync_knowledge_bases():
    """Synchronize the SQLite database with existing Chroma databases."""
    conn = sqlite3.connect('knowledge_bases.db')
    c = conn.cursor()
    
    # Get all existing knowledge bases from the database
    c.execute("SELECT name FROM knowledge_bases")
    existing_kbs = set(row[0] for row in c.fetchall())
    
    # Scan the directory for Chroma databases
    chroma_dbs = [d for d in os.listdir() if d.startswith("chroma_db_")]
    
    for chroma_db in chroma_dbs:
        kb_name = chroma_db[len("chroma_db_"):]
        if kb_name not in existing_kbs:
            # Add the knowledge base to the SQLite database
            now = datetime.now().isoformat()
            c.execute("INSERT INTO knowledge_bases (name, created_at, last_modified) VALUES (?, ?, ?)",
                      (kb_name, now, now))
            print(f"Added knowledge base '{kb_name}' to the database.")
    
    conn.commit()
    conn.close()

# Add this new function to get the original name from a sanitized name
def get_original_name(sanitized_name):
    conn = sqlite3.connect('knowledge_bases.db')
    c = conn.cursor()
    c.execute("SELECT name FROM knowledge_bases WHERE name LIKE ?", (f"%{sanitized_name}%",))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# Add this function to list all directories in the current working directory
def list_directories():
    return [d for d in os.listdir() if os.path.isdir(d)]