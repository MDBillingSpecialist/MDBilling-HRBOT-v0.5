import streamlit as st
import pandas as pd
import os
import datetime
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.llama_dataset import LabelledRagDataset

def evaluate_responses(query_engine, dataset, num_questions, faithfulness_threshold):
    llm = OpenAI(temperature=0, model="gpt-4o-mini")
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)

    correct_relevancy = 0
    faithful_responses = 0
    all_evaluations = []

    for example in dataset.examples[:num_questions]:
        question = example.query
        expected_response = example.reference_answer

        response = query_engine.query(question)
        response_text = str(response)
        
        contexts = [node.node.get_content() for node in response.source_nodes] if response.source_nodes else [""]

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
        is_hallucination = faithfulness_result.score is None or faithfulness_result.score < faithfulness_threshold

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

def save_evaluation_results(all_evaluations, correct_relevancy, faithful_responses, total_questions, faithfulness_threshold, experiment_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = f"evaluation_reports/{experiment_name}_{timestamp}"
    os.makedirs(reports_dir, exist_ok=True)

    relevancy_accuracy = (correct_relevancy / total_questions) * 100
    faithfulness_accuracy = (faithful_responses / total_questions) * 100
    hallucination_rate = 100 - faithfulness_accuracy

    summary_data = {
        "Total Evaluations": [total_questions],
        "Relevancy Accuracy (%)": [relevancy_accuracy],
        "Faithfulness Accuracy (%)": [faithfulness_accuracy],
        "Hallucination Rate (%)": [hallucination_rate],
        "Faithfulness Threshold": [faithfulness_threshold],
    }
    summary_df = pd.DataFrame(summary_data)

    summary_filename = os.path.join(reports_dir, "evaluation_summary.html")
    summary_df.to_html(summary_filename, index=False)

    evaluations_filename = os.path.join(reports_dir, "all_evaluations.html")
    all_evaluations_df = pd.DataFrame(all_evaluations)
    all_evaluations_df.to_html(evaluations_filename, index=False)

    return summary_df, all_evaluations_df, reports_dir

def evaluate_rag_system(rag_dataset, query_engine):
    if not rag_dataset or len(rag_dataset) == 0:
        return {"error": "Empty dataset"}

    RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
    rag_evaluator = RagEvaluatorPack(
        query_engine=query_engine,
        rag_dataset=rag_dataset,
    )
    benchmark_df = rag_evaluator.run()
    return benchmark_df