import streamlit as st
from document_processing import process_documents, display_parsed_documents, display_parser_config
from knowledge_base import create_knowledge_base, export_knowledge_base, load_knowledge_base, create_rag_query_engine, query_knowledge_base, save_knowledge_base_state, load_knowledge_base_state
from question_generation import generate_synthetic_data, log_dataset_structure, save_rag_dataset, load_rag_dataset, save_for_fine_tuning
from api_logger import add_api_call
import logging
import io
import uuid

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
        render_query_rag()

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
    
    # Add save and load options
    if st.sidebar.button("Save Current State"):
        save_knowledge_base_state()
    
    if st.sidebar.button("Load Saved State"):
        if load_knowledge_base_state():
            st.experimental_rerun()
    
    # Step 1: Choose action
    st.sidebar.header("1. Choose Action")
    action = st.sidebar.radio("Select an action", ["Create New Knowledge Base", "Load Existing Knowledge Base"])

    if action == "Create New Knowledge Base":
        create_new_knowledge_base()
    else:
        load_existing_knowledge_base()

    # Step 2: Generate Questions (only if knowledge base exists)
    if 'knowledge_base' in st.session_state and st.session_state['knowledge_base'] is not None:
        generate_questions()

    # Step 3: Query Knowledge Base (only if knowledge base exists)
    if 'knowledge_base' in st.session_state and st.session_state['knowledge_base'] is not None:
        query_rag()

def create_new_knowledge_base():
    st.header("Create New Knowledge Base")
    
    # Step 1: Upload and Process Documents
    st.subheader("1.1 Upload and Process Documents")
    uploaded_files = st.file_uploader("Upload Documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
    
    buffer_size, num_children = display_parser_config()
    
    if uploaded_files and st.button("Process and Parse Documents"):
        processed_docs = process_documents(uploaded_files, buffer_size, num_children)
        if processed_docs:
            st.session_state['parsed_documents'] = processed_docs
            st.success(f"Documents processed and parsed.")
            display_parsed_documents()

    # Step 2: Create Knowledge Base
    if 'parsed_documents' in st.session_state and st.session_state['parsed_documents']:
        st.subheader("1.2 Create Knowledge Base")
        if st.button("Create Knowledge Base"):
            if create_knowledge_base():
                st.success("Knowledge base created successfully.")
                export_path = export_knowledge_base()
                st.success(f"Knowledge base created and exported to: {export_path}")

def load_existing_knowledge_base():
    st.header("Load Existing Knowledge Base")
    kb_path = st.text_input("Enter path to knowledge base")
    if st.button("Load Knowledge Base"):
        if kb_path:
            loaded_index = load_knowledge_base(kb_path)
            if loaded_index:
                st.session_state['knowledge_base'] = loaded_index
                st.success(f"Knowledge base loaded from: {kb_path}")
        else:
            st.error("Please enter a path to the knowledge base.")

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