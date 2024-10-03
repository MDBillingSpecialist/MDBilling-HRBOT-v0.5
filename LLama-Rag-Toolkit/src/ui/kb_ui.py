import streamlit as st
import asyncio
from src.knowledge_base.kb_manager import KnowledgeBaseManager
import os

def render_create_knowledge_base():
    st.title("Create Knowledge Base")

    if 'processed_documents' not in st.session_state:
        st.error("Please upload and process documents first.")
        return

    index_name = st.text_input("Enter a name for the knowledge base:")
    
    if st.button("Create Knowledge Base"):
        if not index_name:
            st.error("Please enter a name for the knowledge base.")
        else:
            with st.spinner("Creating knowledge base..."):
                index = asyncio.run(KnowledgeBaseManager.create_kb(index_name, st.session_state['processed_documents']))
                if index:
                    st.success(f"Successfully created knowledge base '{index_name}' with {len(st.session_state['processed_documents'])} documents.")
                    st.session_state['current_index'] = index
                    st.session_state['index_name'] = index_name
                    
                    # Add download button for Azure-compatible index
                    azure_index_path = f"{index_name}_azure_index.json"
                    if os.path.exists(azure_index_path):
                        with open(azure_index_path, "rb") as file:
                            st.download_button(
                                label="Download Azure AI Studio Index",
                                data=file,
                                file_name=azure_index_path,
                                mime="application/json"
                            )
                        st.info("You can use the downloaded file to import your index into Azure AI Studio.")
                else:
                    st.error("Failed to create knowledge base. Please check the logs for more information.")

    if 'current_index' in st.session_state:
        st.header("Query Your Knowledge Base")
        query = st.text_input("Enter your query:")
        if query:
            if st.button("Search"):
                with st.spinner("Searching..."):
                    result = KnowledgeBaseManager.query_index(st.session_state['current_index'], query)
                    st.write(result)

if __name__ == "__main__":
    render_create_knowledge_base()