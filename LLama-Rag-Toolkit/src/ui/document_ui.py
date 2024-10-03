import streamlit as st
import asyncio
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile
from src.document_processing.document_processor import process_single_file
from src.utils.logger import StreamlitLogger
from config.settings import get_current_llm_model

streamlit_logger = StreamlitLogger()

async def process_documents_async(uploaded_files: List[UploadedFile]):
    try:
        documents = []
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            file_name = uploaded_file.name
            doc = await process_single_file(file_content, file_name)
            if doc:
                documents.append(doc)
        
        if not documents:
            st.error("No documents were successfully processed.")
            return None

        return documents
    except Exception as e:
        st.error(f"Failed to process documents: {str(e)}")
        streamlit_logger.error(f"Error in document processing: {str(e)}")
        return None

def render_upload_documents():
    st.title("Document Upload and Processing")

    st.sidebar.info(f"Current LLM model: {get_current_llm_model()}")

    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf"])
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                documents = asyncio.run(process_documents_async(uploaded_files))
                if documents:
                    st.session_state['processed_documents'] = documents
                    st.success(f"Successfully processed {len(documents)} documents.")
                    st.info("You can now proceed to the 'Create Knowledge Base' step.")
                else:
                    st.error("Failed to process documents. Please check the logs for more information.")

    with st.expander("View Logs"):
        st.text_area("Log Output", streamlit_logger.get_logs(), height=200)

if __name__ == "__main__":
    render_upload_documents()