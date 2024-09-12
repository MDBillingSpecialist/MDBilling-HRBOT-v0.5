import os
import sys
import streamlit as st
import logging
import json

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from utils.config_manager import config
from src.document_processing.document_processor import DocumentProcessor
from src.document_processing.content_segmenter import segment_pdf, post_process_segments
from src.rag_system.rag_builder import RAGBuilder
from src.data_generation.synthetic_data_generator import generate_synthetic_data, analyze_dataset
from src.model_management.fine_tuner import fine_tune_model

logger = logging.getLogger(__name__)

def upload_documents():
    st.header("Step 1: Upload Documents")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "png", "jpg", "jpeg", "tiff"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.join(config.file_paths['input_directory'], uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File {uploaded_file.name} has been uploaded successfully.")
                
                if 'uploaded_documents' not in st.session_state:
                    st.session_state.uploaded_documents = []
                st.session_state.uploaded_documents.append(file_path)
            except Exception as e:
                st.error(f"Error uploading file {uploaded_file.name}: {str(e)}")
        
        if st.button("Proceed to Document Processing"):
            st.session_state.current_stage = "process_documents"
            st.rerun()

def process_documents():
    st.header("Step 2: Process Documents")
    if 'uploaded_documents' not in st.session_state or not st.session_state.uploaded_documents:
        st.warning("Please upload documents first.")
        return

    processor = DocumentProcessor()
    
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []

    for doc_path in st.session_state.uploaded_documents:
        if doc_path not in [doc.get('file_path') for doc in st.session_state.processed_documents]:
            try:
                processed_document = processor.process_document(doc_path)
                st.session_state.processed_documents.append(processed_document)
                
                st.success(f"Document processed successfully: {os.path.basename(doc_path)}")
                
                # Display metadata
                st.subheader("Metadata")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", processed_document['metadata'].get('word_count', 'N/A'))
                    st.metric("Character Count", processed_document['metadata'].get('character_count', 'N/A'))
                with col2:
                    st.metric("Average Word Length", f"{processed_document['metadata'].get('average_word_length', 'N/A'):.2f}")
                    st.metric("Page Count", processed_document['metadata'].get('page_count', 'N/A'))

                # Display image information
                st.subheader("Image Information")
                for img in processed_document['metadata'].get('image_info', []):
                    with st.expander(f"Image on page {img['page_number']}"):
                        st.json(img)

                # Display semantic data
                st.subheader("Semantic Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Main Topics")
                    st.write(processed_document['semantic_data'].get('main_topics', []))
                with col2:
                    st.write("Key Entities")
                    st.write(processed_document['semantic_data'].get('key_entities', []))
                
                st.write("Summary")
                st.write(processed_document['semantic_data'].get('summary', 'No summary available'))

            except Exception as e:
                st.error(f"Error processing document {os.path.basename(doc_path)}: {str(e)}")

    if st.button("Proceed to Document Segmentation"):
        st.session_state.current_stage = "segment_documents"
        st.rerun()

def segment_documents():
    st.header("Step 3: Segment Documents")
    if 'processed_documents' not in st.session_state or not st.session_state.processed_documents:
        st.warning("Please process documents first.")
        return

    if 'segmented_documents' not in st.session_state:
        st.session_state.segmented_documents = []

    for doc in st.session_state.processed_documents:
        if doc['file_path'] not in [seg_doc.get('file_path') for seg_doc in st.session_state.segmented_documents]:
            try:
                raw_segments = segment_pdf(doc['file_path'], doc.get('toc', {}))
                processed_segments = post_process_segments(raw_segments)
                
                segmented_doc = doc.copy()
                segmented_doc['segments'] = processed_segments
                st.session_state.segmented_documents.append(segmented_doc)
                
                st.success(f"Document segmented successfully: {os.path.basename(doc['file_path'])}")
                
                # Display segments
                st.subheader("Document Segments")
                for i, segment in enumerate(processed_segments[:5]):  # Display first 5 segments
                    with st.expander(f"Segment {i+1}: {segment['title']}"):
                        st.write(f"Content: {segment['content'][:200]}...")
                        st.metric("Tokens", segment['tokens'])
                
                st.info(f"Total segments: {len(processed_segments)}")

            except Exception as e:
                st.error(f"Error segmenting document {os.path.basename(doc['file_path'])}: {str(e)}")

    if st.button("Proceed to RAG System Building"):
        st.session_state.current_stage = "build_rag"
        st.rerun()

def build_rag():
    st.header("Step 4: Build RAG System")
    if 'segmented_documents' not in st.session_state or not st.session_state.segmented_documents:
        st.warning("Please segment documents first.")
        return

    rag_builder = RAGBuilder()
    
    if 'rag_systems' not in st.session_state:
        st.session_state.rag_systems = []

    for doc in st.session_state.segmented_documents:
        if 'file_path' not in doc:
            st.error(f"Missing file path for document: {doc.get('file_name', 'Unknown')}")
            continue

        if doc['file_path'] not in [rag.get('file_path') for rag in st.session_state.rag_systems]:
            try:
                rag_system = rag_builder.build_rag_system(doc)
                rag_system['file_path'] = doc['file_path']  # Ensure file_path is in the rag_system
                st.session_state.rag_systems.append(rag_system)
                st.success(f"RAG system built successfully for: {os.path.basename(doc['file_path'])}")
                
                # Display RAG system information
                st.subheader("RAG System Information")
                st.write(f"Number of chunks: {len(rag_system['chunks'])}")
                st.write(f"Embedding dimension: {rag_system['embeddings'].shape[1]}")
                
                # Display a sample query and retrieval
                sample_query = "What is the main topic of this document?"
                retrieved_chunks = rag_builder.get_relevant_chunks(sample_query, rag_system['chunks'])
                st.write("Sample Query:", sample_query)
                st.write("Top retrieved chunk:")
                st.json(retrieved_chunks[0] if retrieved_chunks else "No chunks retrieved")
                
            except Exception as e:
                st.error(f"Error building RAG system for {os.path.basename(doc['file_path'])}: {str(e)}")

    if st.button("Proceed to Synthetic Data Generation"):
        st.session_state.current_stage = "generate_synthetic_data"
        st.rerun()

def generate_synthetic_data_stage():
    st.header("Step 5: Generate Synthetic Data")
    if 'rag_systems' not in st.session_state or not st.session_state.rag_systems:
        st.warning("Please build RAG systems first.")
        return

    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = []

    for rag_system in st.session_state.rag_systems:
        if rag_system['file_path'] not in [data['file_path'] for data in st.session_state.synthetic_data]:
            try:
                synthetic_data = generate_synthetic_data(rag_system)
                st.session_state.synthetic_data.append({
                    'file_path': rag_system['file_path'],
                    'data': synthetic_data
                })
                st.success(f"Synthetic data generated successfully for: {os.path.basename(rag_system['file_path'])}")
                
                # Display some statistics about the generated data
                output_file = os.path.join(config.file_paths['output_folder'], f"{os.path.basename(rag_system['file_path'])}_synthetic_data.jsonl")
                analysis = analyze_dataset(output_file)
                st.write(f"Dataset Analysis for {os.path.basename(rag_system['file_path'])}:")
                st.json(analysis)
            except Exception as e:
                st.error(f"Error generating synthetic data for {os.path.basename(rag_system['file_path'])}: {str(e)}")

    if st.button("Proceed to Model Fine-tuning"):
        st.session_state.current_stage = "fine_tune_model"
        st.rerun()

def fine_tune_model_stage():
    st.header("Step 6: Fine-tune Model")
    if 'synthetic_data' not in st.session_state or not st.session_state.synthetic_data:
        st.warning("Please generate synthetic data first.")
        return

    if 'fine_tuned_models' not in st.session_state:
        st.session_state.fine_tuned_models = []

    for data in st.session_state.synthetic_data:
        if data['file_path'] not in [model['file_path'] for model in st.session_state.fine_tuned_models]:
            try:
                fine_tuned_model_id = fine_tune_model(data['data'])
                if fine_tuned_model_id:
                    st.session_state.fine_tuned_models.append({
                        'file_path': data['file_path'],
                        'model_id': fine_tuned_model_id
                    })
                    st.success(f"Model fine-tuned successfully for: {os.path.basename(data['file_path'])}. Job ID: {fine_tuned_model_id}")
                else:
                    st.error(f"Fine-tuning failed for {os.path.basename(data['file_path'])}. Please check the logs for more information.")
            except Exception as e:
                st.error(f"Error fine-tuning model for {os.path.basename(data['file_path'])}: {str(e)}")

    if st.button("Complete Process"):
        st.session_state.current_stage = "completed"
        st.rerun()

def main():
    st.title("LLM-RAG-Toolkit")

    # Ensure necessary directories exist
    os.makedirs(config.file_paths['input_directory'], exist_ok=True)
    os.makedirs(config.file_paths['output_folder'], exist_ok=True)

    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "upload_documents"

    stages = {
        "upload_documents": upload_documents,
        "process_documents": process_documents,
        "segment_documents": segment_documents,
        "build_rag": build_rag,
        "generate_synthetic_data": generate_synthetic_data_stage,
        "fine_tune_model": fine_tune_model_stage,
        "completed": lambda: st.success("All steps completed!")
    }

    # Display progress
    st.sidebar.title("Progress")
    for stage in stages:
        if st.session_state.current_stage == stage:
            st.sidebar.markdown(f"**{stage.replace('_', ' ').title()}**")
        elif stage in st.session_state:
            st.sidebar.markdown(f"~~{stage.replace('_', ' ').title()}~~")
        else:
            st.sidebar.markdown(stage.replace('_', ' ').title())

    # Run the current stage
    stages[st.session_state.current_stage]()

if __name__ == "__main__":
    main()