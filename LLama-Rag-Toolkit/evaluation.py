import streamlit as st
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from utils import query_llm

def evaluate_responses(num_questions, faithfulness_threshold):
    query_engine = st.session_state['knowledge_base'].as_query_engine()
    relevancy_evaluator = RelevancyEvaluator()
    faithfulness_evaluator = FaithfulnessEvaluator()

    correct_relevancy = 0
    faithful_responses = 0
    all_evaluations = []

    for example in st.session_state['dataset'].examples[:num_questions]:
        question = example.query
        expected_response = example.reference_answer

        response_text = query_llm(st.session_state['openai_client'], question)
        
        contexts = [""]

        relevancy_result = relevancy_evaluator.evaluate(
            query=question,
            response=response_text,
            contexts=contexts
        )
        faithfulness_result = faithfulness_evaluator.evaluate(
            query=question,
            response=response_text,
            contexts=contexts
        )
        is_hallucination = faithfulness_result.score is not None and faithfulness_result.score < faithfulness_threshold

        eval_record = {
            "Query": question,
            "Generated Response": response_text,
            "Expected Response": expected_response,
            "Relevancy Score": relevancy_result.score,
            "Faithfulness Score": faithfulness_result.score,
            "Is Hallucination": is_hallucination
        }
        all_evaluations.append(eval_record)

        if relevancy_result.score and relevancy_result.score > 0.5:
            correct_relevancy += 1
        if not is_hallucination:
            faithful_responses += 1

    return all_evaluations, correct_relevancy, faithful_responses, num_questions