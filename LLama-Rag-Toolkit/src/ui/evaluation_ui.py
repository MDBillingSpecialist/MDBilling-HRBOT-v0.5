import streamlit as st
from src.evaluation.evaluation import evaluate_rag_system

def render_evaluate_rag():
    st.title("Evaluate RAG System")
    
    if 'current_index' not in st.session_state:
        st.error("Please create a Knowledge Base first.")
        return
    
    if 'synthetic_qa_pairs' not in st.session_state:
        st.error("Please generate synthetic questions first.")
        return
    
    if st.button("Evaluate RAG System"):
        with st.spinner("Evaluating RAG system..."):
            evaluation_results = evaluate_rag_system(
                st.session_state['current_index'],
                st.session_state['synthetic_qa_pairs']
            )
            
            st.subheader("Evaluation Results")
            st.write(f"Average Relevancy Score: {evaluation_results['average_relevancy_score']:.2f}")
            st.write(f"Average Faithfulness Score: {evaluation_results['average_faithfulness_score']:.2f}")
            
            st.subheader("Detailed Results")
            for i, result in enumerate(evaluation_results['detailed_results']):
                with st.expander(f"Query {i+1}"):
                    st.write(f"Query: {result['query']}")
                    st.write(f"Expected Answer: {result['expected_answer']}")
                    st.write(f"Generated Answer: {result['generated_answer']}")
                    st.write(f"Relevancy Score: {result['relevancy_score']:.2f}")
                    st.write(f"Faithfulness Score: {result['faithfulness_score']:.2f}")

if __name__ == "__main__":
    render_evaluate_rag()