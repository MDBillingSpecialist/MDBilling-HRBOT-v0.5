import streamlit as st
from src.knowledge_base.kb_manager import KnowledgeBaseManager

def render_query_rag():
    st.title("Query Knowledge Base")

    if 'current_index' not in st.session_state:
        st.error("Please create a Knowledge Base first.")
        return

    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            with st.spinner("Generating response..."):
                response = KnowledgeBaseManager.query_index(st.session_state['current_index'], query)
                st.subheader("Response:")
                st.write(response.response)
                st.subheader("Source Nodes:")
                for node in response.source_nodes:
                    st.write(f"- {node.node.get_content()[:200]}...")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    render_query_rag()