import streamlit as st
from src.question_generation.generator import generate_synthetic_data, save_rag_dataset
from src.evaluation.evaluation import evaluate_rag_system, save_evaluation_results

def render_generate_questions():
    st.title("Generate Synthetic Questions")
    
    if 'current_index' not in st.session_state:
        st.error("Please create a Knowledge Base first.")
        return

    questions_per_node = st.slider("Questions per node", min_value=1, max_value=10, value=3)
    model = st.selectbox("Select OpenAI model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"])
    
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic questions..."):
            num_qa_pairs, dataset = generate_synthetic_data(
                st.session_state['current_index'],
                questions_per_node,
                model
            )
            
            if dataset:
                st.success(f"Generated {num_qa_pairs} Q&A pairs")
                st.session_state['synthetic_qa_pairs'] = dataset
                
                if st.button("Save RAG Dataset"):
                    save_rag_dataset(dataset, f"{st.session_state['index_name']}_rag_dataset.json")
                    st.success("RAG dataset saved successfully.")
            else:
                st.error("Failed to generate synthetic data")

    st.title("Evaluate RAG System")
    if 'synthetic_qa_pairs' in st.session_state and st.session_state['synthetic_qa_pairs']:
        eval_model = st.selectbox("Select evaluation model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"], key="eval_model")
        num_questions = st.slider("Number of questions to evaluate", min_value=1, max_value=len(st.session_state['synthetic_qa_pairs']), value=min(10, len(st.session_state['synthetic_qa_pairs'])))
        
        if st.button("Evaluate RAG System"):
            with st.spinner("Evaluating RAG system..."):
                evaluation_results = evaluate_rag_system(
                    st.session_state['current_index'],
                    st.session_state['synthetic_qa_pairs'][:num_questions],
                    eval_model
                )
                
                st.subheader("Evaluation Results")
                st.write(f"Total Questions: {evaluation_results['total_questions']}")
                st.write(f"Relevancy Accuracy: {evaluation_results['relevancy_accuracy']:.2f}%")
                st.write(f"Faithfulness Accuracy: {evaluation_results['faithfulness_accuracy']:.2f}%")
                st.write(f"Answer Relevancy Accuracy: {evaluation_results['answer_relevancy_accuracy']:.2f}%")
                st.write(f"Hallucination Rate: {evaluation_results['hallucination_rate']:.2f}%")
                
                st.write(f"Average Relevancy Score: {evaluation_results['average_relevancy_score']:.2f}")
                st.write(f"Average Faithfulness Score: {evaluation_results['average_faithfulness_score']:.2f}")
                st.write(f"Average Answer Relevancy Score: {evaluation_results['average_answer_relevancy_score']:.2f}")
                
                st.session_state['evaluation_results'] = evaluation_results
                
        if 'evaluation_results' in st.session_state:
            if st.button("Save Evaluation Results"):
                experiment_name = st.text_input("Enter experiment name:", value=st.session_state['index_name'])
                summary_df, all_evaluations_df, reports_dir = save_evaluation_results(st.session_state['evaluation_results'], experiment_name)
                st.success(f"Evaluation results saved in {reports_dir}")
                st.dataframe(summary_df)
                with st.expander("Show detailed results"):
                    st.dataframe(all_evaluations_df)
    else:
        st.warning("Please generate synthetic data first before evaluating.")

if __name__ == "__main__":
    render_generate_questions()