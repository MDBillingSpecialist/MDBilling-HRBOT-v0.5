import streamlit as st
import pandas as pd
import chromadb
from document_processing import upload_document, parse_document
from knowledge_base import create_knowledge_base, export_knowledge_base
from question_generation import generate_synthetic_data, load_dataset, evaluate_responses, save_evaluation_results, export_qa_dataset

# Add the functions: initialize_session_state, handle_upload_documents, 
# handle_parse_documents, handle_create_knowledge_base, 
# and handle_generate_questions_and_evaluate here
# (The content of these functions should be similar to what was in the original app2.py)

def initialize_session_state():
    if 'documents' not in st.session_state:
        st.session_state['documents'] = {}
    if 'parsed_documents' not in st.session_state:
        st.session_state['parsed_documents'] = {}
    if 'knowledge_base' not in st.session_state:
        st.session_state['knowledge_base'] = None
    if 'dataset' not in st.session_state:
        st.session_state['dataset'] = load_dataset()
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = {'uploaded_docs': 0, 'parsed_docs': 0, 'total_chunks': 0, 'kb_size': 0, 'total_nodes': 0}
    if 'chroma_client' not in st.session_state:
        st.session_state['chroma_client'] = chromadb.Client()
    if 'chroma_collection' not in st.session_state:
        collection_name = "my_collection"
        st.session_state['chroma_collection'] = st.session_state['chroma_client'].get_or_create_collection(collection_name)

def handle_upload_documents():
    st.header("Step 1: Upload Documents")
    uploaded_files = st.file_uploader("Upload Documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if upload_document(uploaded_file):
                st.success(f"Uploaded {uploaded_file.name}")
            else:
                st.warning(f"Document '{uploaded_file.name}' is already uploaded.")
    
    display_uploaded_documents()

def display_uploaded_documents():
    if st.session_state['documents']:
        st.subheader("Uploaded Documents")
        doc_df = pd.DataFrame([{'Name': doc_info['name'], 'Size': doc_info['size'], 'Upload Time': doc_info['upload_time']} for doc_info in st.session_state['documents'].values()])
        st.dataframe(doc_df)

def handle_parse_documents():
    st.header("Step 2: Parse Documents")
    if not st.session_state['documents']:
        st.error("Please upload documents first.")
    else:
        if st.button("Parse All Documents"):
            for doc_id, doc_info in st.session_state['documents'].items():
                if doc_id not in st.session_state['parsed_documents']:
                    if parse_document(doc_id, doc_info):
                        st.success(f"Parsed {doc_info['name']}")
        
        if st.session_state['parsed_documents']:
            st.subheader("Parsing Metrics")
            st.write(f"Parsed Documents: {st.session_state['metrics']['parsed_docs']}")
            st.write(f"Total Chunks: {st.session_state['metrics']['total_chunks']}")

def handle_create_knowledge_base():
    st.header("Step 3: Create Knowledge Base")
    if not st.session_state['parsed_documents']:
        st.error("Please parse documents first.")
    else:
        if st.button("Create Knowledge Base"):
            if create_knowledge_base():
                st.success("Knowledge base created successfully!")
                export_path = export_knowledge_base()
                st.success(f"Knowledge base exported to: {export_path}")
        
        if st.session_state['knowledge_base']:
            st.subheader("Knowledge Base Metrics")
            st.write(f"Knowledge Base Size: {st.session_state['metrics']['kb_size']}")

def handle_generate_questions_and_evaluate():
    st.header("Step 4: Generate Questions and Evaluate")
    
    if not st.session_state.get('parsed_documents'):
        st.error("Please parse documents first before generating synthetic data.")
        return
    
    parser_choice = st.selectbox("Choose a parsing method", ["Simple", "Semantic", "Hierarchical"])
    questions_per_chunk = st.number_input("Number of questions per chunk", min_value=1, value=5)
    
    model_options = [
        "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo"
    ]
    selected_model = st.selectbox("Choose AI Model", model_options)
    
    if st.button("Generate Synthetic Data"):
        new_questions = generate_synthetic_data(questions_per_chunk, parser_choice, selected_model)
        st.success(f"Generated {new_questions} new questions using {selected_model}.")
        
        if new_questions > 0:
            # Export the QA dataset
            qa_export_path = export_qa_dataset(st.session_state['dataset'])
            st.success(f"QA dataset exported to: {qa_export_path}")
    
    if 'dataset' in st.session_state and st.session_state['dataset'].examples:
        st.write(f"Available questions: {len(st.session_state['dataset'].examples)}")

        max_questions = len(st.session_state['dataset'].examples)
        num_questions = st.slider("Number of questions to evaluate", min_value=1, max_value=min(max_questions, 10), value=5)
        faithfulness_threshold = st.slider("Faithfulness Threshold", min_value=0.0, max_value=1.0, value=0.5)
        experiment_name = st.text_input("Experiment Name", value="default_experiment")

        if st.button("Evaluate Responses"):
            with st.spinner("Evaluating responses..."):
                query_engine = st.session_state['knowledge_base'].as_query_engine()
                all_evaluations, correct_relevancy, faithful_responses, total_questions = evaluate_responses(
                    query_engine, st.session_state['dataset'], num_questions, faithfulness_threshold
                )

                summary_df, all_evaluations_df, reports_dir = save_evaluation_results(
                    all_evaluations, correct_relevancy, faithful_responses, total_questions, 
                    faithfulness_threshold, experiment_name
                )

                st.subheader("Evaluation Results")
                st.dataframe(all_evaluations_df)
                st.subheader("Summary")
                st.dataframe(summary_df)
                st.success(f"Detailed reports saved in: {reports_dir}")
    else:
        st.warning("No questions available. Please generate synthetic data first.")