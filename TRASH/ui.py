import streamlit as st
from document_processing import process_documents, display_parsed_documents, display_parser_config
from knowledge_base import (
    create_knowledge_base, save_knowledge_base, load_knowledge_base, 
    create_rag_query_engine, query_knowledge_base, create_advanced_rag_query_engine,
    init_db, get_all_knowledge_bases, delete_knowledge_base, sanitize_name, sync_knowledge_bases,
    get_original_name, list_directories
)
from question_generation import generate_synthetic_data, log_dataset_structure, save_rag_dataset, load_rag_dataset, save_for_fine_tuning
from api_logger import add_api_call
import logging
import io
import uuid
import os
import shutil

def initialize_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'step' not in st.session_state:
        st.session_state.step = 'upload_documents'

def render_ui():
    initialize_session()
    
    if st.session_state.step == 'upload_documents':
        render_upload_documents()
    elif st.session_state.step == 'create_knowledge_base':
        render_create_knowledge_base()
    elif st.session_state.step == 'generate_questions':
        render_generate_questions()
    elif st.session_state.step == 'query_rag':
        query_rag()
    elif st.session_state.step == 'test_knowledge_base':
        test_knowledge_base()

def render_upload_documents():
    # ... (code for uploading documents)
    if documents_uploaded:
        st.session_state.step = 'create_knowledge_base'
        st.experimental_rerun()

def render_create_knowledge_base():
    # ... (code for creating knowledge base)
    if knowledge_base_created:
        st.session_state.step = 'generate_questions'
        st.experimental_rerun()

def render_generate_questions():
    # ... (code for generating questions)
    if questions_generated:
        st.session_state.step = 'query_rag'
        st.experimental_rerun()

def render_query_rag():
    # ... (code for querying RAG)
    st.header("3. Query Knowledge Base using RAG")
    model = st.selectbox("Select Model for RAG", [
        "gpt-4o-mini",
        "gpt-4o",
        "o1-mini",
        "o1-preview"
    ])
    
    if 'rag_query_engine' not in st.session_state:
        st.session_state['rag_query_engine'] = create_rag_query_engine(st.session_state['knowledge_base'], model)
    
    query = st.text_input("Enter your query:")
    
    if st.button("Submit Query"):
        if query:
            with st.spinner("Generating response..."):
                response = query_knowledge_base(st.session_state['rag_query_engine'], query)
                st.subheader("RAG Response:")
                st.write(response)
                st.subheader("Source Nodes:")
                for node in response.source_nodes:
                    st.write(f"- {node.node.get_content()[:200]}...")
        else:
            st.warning("Please enter a query.")

def render_sidebar():
    st.sidebar.title("LLama RAG Toolkit")
    
    # Initialize database and sync knowledge bases
    init_db()
    sync_knowledge_bases()
    
    # Manage knowledge bases
    st.sidebar.header("Manage Knowledge Bases")
    kb_action = st.sidebar.radio("Select action", ["Create New KB", "Load Existing KB", "Delete KB"])
    
    if kb_action == "Create New KB":
        create_new_knowledge_base()
    elif kb_action == "Load Existing KB":
        load_existing_knowledge_base()
    elif kb_action == "Delete KB":
        delete_existing_knowledge_base()
    
    # Generate Questions (only if knowledge base exists)
    if 'current_kb' in st.session_state and st.session_state['current_kb'] is not None:
        generate_questions()
    
    # Query Knowledge Base (only if knowledge base exists)
    if 'current_kb' in st.session_state and st.session_state['current_kb'] is not None:
        query_rag()
    
    # Test Knowledge Base (only if knowledge base exists)
    if 'current_kb' in st.session_state and st.session_state['current_kb'] is not None:
        if st.sidebar.button("Test Knowledge Base with GPT-4"):
            st.session_state.step = 'test_knowledge_base'
            st.experimental_rerun()

def create_new_knowledge_base():
    st.header("Create New Knowledge Base")
    
    kb_name = st.text_input("Enter a name for the new knowledge base:")
    
    # Step 1: Upload and Process Documents
    st.subheader("1.1 Upload and Process Documents")
    uploaded_files = st.file_uploader("Upload Documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
    
    if uploaded_files and st.button("Process and Parse Documents"):
        with st.spinner("Processing and parsing documents..."):
            processed_docs = process_documents(uploaded_files)
            if processed_docs:
                st.session_state['parsed_documents'] = processed_docs
                st.success(f"Documents processed and parsed. {len(processed_docs)} documents created.")
                display_parsed_documents()
            else:
                st.error("Failed to process documents.")

    # Step 2: Create Knowledge Base
    if 'parsed_documents' in st.session_state and st.session_state['parsed_documents']:
        st.subheader("1.2 Create Knowledge Base")
        
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        
        if st.button("Create Knowledge Base"):
            if kb_name:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress):
                    progress_bar.progress(progress / 100)
                    status_text.text(f"Creating knowledge base... {progress:.2f}% complete")

                try:
                    with st.spinner("Creating knowledge base..."):
                        new_kb = create_knowledge_base(kb_name, update_progress, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        st.session_state['current_kb'] = new_kb
                        st.session_state['current_kb_name'] = kb_name
                        save_knowledge_base(new_kb, kb_name)
                        st.success(f"Knowledge base '{kb_name}' created and saved successfully.")
                except Exception as e:
                    st.error(f"Failed to create knowledge base: {str(e)}")
            else:
                st.error("Please enter a name for the knowledge base.")
    else:
        st.info("Please upload and process documents before creating a knowledge base.")

def load_existing_knowledge_base():
    st.header("Load Existing Knowledge Base")
    knowledge_bases = get_all_knowledge_bases()
    
    if not knowledge_bases:
        st.info("No knowledge bases found in the database. Checking for existing Chroma databases...")
        sync_knowledge_bases()
        knowledge_bases = get_all_knowledge_bases()
    
    if not knowledge_bases:
        st.info("No knowledge bases found. Create a new one first.")
        return
    
    kb_options = [f"{kb[0]} (Created: {kb[1]}, Last Modified: {kb[2]})" for kb in knowledge_bases]
    selected_kb = st.selectbox("Select a knowledge base to load:", kb_options)
    selected_kb_name = selected_kb.split(" (")[0]
    
    if st.button("Load Knowledge Base"):
        loaded_kb = load_knowledge_base(selected_kb_name)
        if loaded_kb:
            st.session_state['current_kb'] = loaded_kb
            st.session_state['current_kb_name'] = selected_kb_name
            st.success(f"Knowledge base '{selected_kb_name}' loaded successfully.")
        else:
            st.error("Failed to load knowledge base.")
            st.write("Debugging information:")
            st.write(f"Current working directory: {os.getcwd()}")
            st.write("Directories in current working directory:")
            st.write(list_directories())
            
            # Try to find a matching directory
            chroma_dirs = [d for d in os.listdir() if d.startswith("chroma_db_")]
            export_dirs = [d for d in os.listdir() if d.startswith("exported_knowledge_base_")]
            
            sanitized_name = sanitize_name(selected_kb_name)
            matching_chroma = [d for d in chroma_dirs if sanitized_name in d]
            matching_export = [d for d in export_dirs if sanitized_name in d]
            
            if matching_chroma or matching_export:
                st.write("Found potential matching directories:")
                st.write(f"Chroma: {matching_chroma}")
                st.write(f"Exported: {matching_export}")
                
                if st.button("Update database with correct name"):
                    new_name = matching_chroma[0][len("chroma_db_"):] if matching_chroma else matching_export[0][len("exported_knowledge_base_"):]
                    update_kb_name_in_db(selected_kb_name, new_name)
                    st.success(f"Updated database. Please try loading the knowledge base again with the name: {new_name}")
            else:
                st.write("No matching directories found.")

def delete_existing_knowledge_base():
    st.header("Delete Knowledge Base")
    knowledge_bases = get_all_knowledge_bases()
    
    if not knowledge_bases:
        st.info("No knowledge bases found.")
        return
    
    kb_options = [f"{kb[0]} (Created: {kb[1]}, Last Modified: {kb[2]})" for kb in knowledge_bases]
    selected_kb = st.selectbox("Select a knowledge base to delete:", kb_options)
    selected_kb_name = selected_kb.split(" (")[0]
    
    if st.button("Delete Knowledge Base"):
        if st.session_state.get('current_kb_name') == selected_kb_name:
            st.session_state.pop('current_kb', None)
            st.session_state.pop('current_kb_name', None)
        
        delete_knowledge_base(selected_kb_name)
        st.experimental_rerun()

def generate_questions():
    st.header("2. Generate Synthetic Questions and Answers")
    model = st.selectbox("Select Model", [
        "gpt-4o-mini",
        "gpt-4o",
        "o1-mini",
        "o1-preview"
    ])
    questions_per_node = st.number_input("Questions per node", min_value=1, value=5)
    
    if st.button("Generate Synthetic Data"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_output = st.empty()
        api_calls_container = st.empty()
        
        def update_progress(progress):
            progress = min(100, max(0, progress))  # Ensure progress is between 0 and 100
            progress_bar.progress(progress / 100)  # Convert to [0, 1] range for Streamlit
            status_text.text(f"Generating and checking relevance of Q&A pairs... {progress:.2f}% complete")
        
        # Capture log output
        with st.spinner("Generating synthetic data..."):
            log_capture_string = io.StringIO()
            ch = logging.StreamHandler(log_capture_string)
            ch.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs
            logger = logging.getLogger()  # Capture root logger to get OpenAI logs
            logger.addHandler(ch)
            
            num_qa_pairs, dataset = generate_synthetic_data(questions_per_node, model, update_progress)
            
            logger.removeHandler(ch)
            log_contents = log_capture_string.getvalue()
            log_output.text_area("Log Output:", log_contents, height=500)
        
        progress_bar.progress(100)
        status_text.text("Question and answer generation complete!")
        
        st.success(f"Generated {num_qa_pairs} relevant question-answer pairs.")
        log_dataset_structure(dataset)
        
        # Save RAG dataset
        if st.button("Save RAG Dataset"):
            save_rag_dataset(st.session_state['rag_dataset'])
        
        # Save for fine-tuning
        if st.button("Save Dataset for Fine-tuning"):
            save_for_fine_tuning(dataset)
        
        # Display API calls
        api_calls_container.text_area("API Calls:", value=get_api_calls_text(), height=500)

def get_api_calls_text():
    if 'api_calls' not in st.session_state:
        return "No API calls logged yet."
    return "\n\n".join([f"Request: {call['request']}\nResponse: {call['response']}" for call in st.session_state.api_calls[-10:]])  # Show only the last 10 calls

def render_api_calls():
    st.header("API Calls and Responses")
    st.text_area("API Calls:", value=get_api_calls_text(), height=300, key="api_calls_display")

def render_main_content():
    st.title("LLama RAG Toolkit")
    render_sidebar()
    render_api_calls()

def render_ui():
    render_main_content()

def query_rag():
    st.header("3. Query Knowledge Base using RAG")
    st.subheader(f"Current Knowledge Base: {st.session_state.get('current_kb_name', 'None')}")
    
    model = st.selectbox("Select Model for RAG", [
        "gpt-4o-mini",
        "gpt-4o",
        "o1-mini",
        "o1-preview"
    ])
    
    if 'rag_query_engine' not in st.session_state or st.session_state.get('current_kb_name') != st.session_state.get('last_queried_kb'):
        st.session_state['rag_query_engine'] = create_rag_query_engine(st.session_state['current_kb'], model)
        st.session_state['last_queried_kb'] = st.session_state.get('current_kb_name')
    
    query = st.text_input("Enter your query:")
    
    if st.button("Submit Query"):
        if query:
            with st.spinner("Generating response..."):
                response = query_knowledge_base(st.session_state['rag_query_engine'], query)
                st.subheader("RAG Response:")
                st.write(response)
                st.subheader("Source Nodes:")
                for node in response.source_nodes:
                    st.write(f"- {node.node.get_content()[:200]}...")
        else:
            st.warning("Please enter a query.")

def test_knowledge_base():
    st.header("4. Test Knowledge Base with GPT-4")
    st.subheader(f"Current Knowledge Base: {st.session_state.get('current_kb_name', 'None')}")
    
    model = st.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"])
    retriever_mode = st.selectbox("Select Retriever Mode", ["hybrid", "embedding", "keyword"])
    
    if 'advanced_rag_query_engine' not in st.session_state or \
       st.session_state.get('current_kb_name') != st.session_state.get('last_tested_kb') or \
       st.session_state.get('last_tested_model') != model or \
       st.session_state.get('last_tested_retriever') != retriever_mode:
        st.session_state['advanced_rag_query_engine'] = create_advanced_rag_query_engine(
            st.session_state['current_kb'], 
            model=model, 
            retriever_mode=retriever_mode
        )
        st.session_state['last_tested_kb'] = st.session_state.get('current_kb_name')
        st.session_state['last_tested_model'] = model
        st.session_state['last_tested_retriever'] = retriever_mode
    
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Ask GPT-4"):
        if query:
            with st.spinner("Generating response..."):
                response = st.session_state['advanced_rag_query_engine'].query(query)
                st.subheader("GPT-4 Response:")
                st.write(response.response)
                st.subheader("Source Nodes:")
                for node in response.source_nodes:
                    st.write(f"- {node.node.get_content()[:200]}...")
                st.subheader("Thought Process:")
                st.write(response.metadata['thought_process'])
        else:
            st.warning("Please enter a question.")