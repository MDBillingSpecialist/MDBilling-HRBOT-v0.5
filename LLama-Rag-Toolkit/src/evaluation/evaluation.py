import streamlit as st
import pandas as pd
import os
import datetime
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, AnswerRelevancyEvaluator
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core import VectorStoreIndex
from typing import List, Dict

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

def evaluate_rag_system(index: VectorStoreIndex, synthetic_qa_pairs: List[Dict[str, str]], model: str = "gpt-4o-mini"):
    if not index or not synthetic_qa_pairs:
        return {"error": "Invalid index or empty dataset"}

    llm = OpenAI(temperature=0, model=model)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=llm)

    query_engine = index.as_query_engine()
    
    results = []
    for qa_pair in synthetic_qa_pairs:
        query = qa_pair['question']
        expected_answer = qa_pair['answer']
        
        response = query_engine.query(query)
        response_text = str(response)
        
        contexts = [node.node.get_content() for node in response.source_nodes] if response.source_nodes else [""]

        relevancy_result = relevancy_evaluator.evaluate(
            query=query,
            response=response_text,
            contexts=contexts
        )
        faithfulness_result = faithfulness_evaluator.evaluate(
            query=query,
            response=response_text,
            contexts=contexts
        )
        answer_relevancy_result = answer_relevancy_evaluator.evaluate(
            query=query,
            response=response_text,
            ground_truth=expected_answer
        )

        results.append({
            "query": query,
            "expected_answer": expected_answer,
            "generated_answer": response_text,
            "relevancy_score": relevancy_result.score,
            "faithfulness_score": faithfulness_result.score,
            "answer_relevancy_score": answer_relevancy_result.score,
            "contexts": contexts  # Add this line to include contexts in the results
        })

    # Calculate metrics
    avg_relevancy = sum(r['relevancy_score'] for r in results if r['relevancy_score'] is not None) / len(results)
    avg_faithfulness = sum(r['faithfulness_score'] for r in results if r['faithfulness_score'] is not None) / len(results)
    avg_answer_relevancy = sum(r['answer_relevancy_score'] for r in results if r['answer_relevancy_score'] is not None) / len(results)
    
    correct_relevancy = sum(1 for r in results if r['relevancy_score'] is not None and r['relevancy_score'] > 0.5)
    correct_faithfulness = sum(1 for r in results if r['faithfulness_score'] is not None and r['faithfulness_score'] > 0.5)
    correct_answer_relevancy = sum(1 for r in results if r['answer_relevancy_score'] is not None and r['answer_relevancy_score'] > 0.5)

    total_questions = len(results)
    relevancy_accuracy = (correct_relevancy / total_questions) * 100
    faithfulness_accuracy = (correct_faithfulness / total_questions) * 100
    answer_relevancy_accuracy = (correct_answer_relevancy / total_questions) * 100
    hallucination_rate = 100 - faithfulness_accuracy

    return {
        "detailed_results": results,
        "average_relevancy_score": avg_relevancy,
        "average_faithfulness_score": avg_faithfulness,
        "average_answer_relevancy_score": avg_answer_relevancy,
        "relevancy_accuracy": relevancy_accuracy,
        "faithfulness_accuracy": faithfulness_accuracy,
        "answer_relevancy_accuracy": answer_relevancy_accuracy,
        "hallucination_rate": hallucination_rate,
        "total_questions": total_questions
    }

def save_evaluation_results(evaluation_results, experiment_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = f"evaluation_reports/{experiment_name}_{timestamp}"
    os.makedirs(reports_dir, exist_ok=True)

    summary_data = {
        "Total Evaluations": [evaluation_results['total_questions']],
        "Relevancy Accuracy (%)": [evaluation_results['relevancy_accuracy']],
        "Faithfulness Accuracy (%)": [evaluation_results['faithfulness_accuracy']],
        "Answer Relevancy Accuracy (%)": [evaluation_results['answer_relevancy_accuracy']],
        "Hallucination Rate (%)": [evaluation_results['hallucination_rate']],
        "Average Relevancy Score": [evaluation_results['average_relevancy_score']],
        "Average Faithfulness Score": [evaluation_results['average_faithfulness_score']],
        "Average Answer Relevancy Score": [evaluation_results['average_answer_relevancy_score']],
    }
    summary_df = pd.DataFrame(summary_data)

    summary_filename = os.path.join(reports_dir, "evaluation_summary.html")
    summary_df.to_html(summary_filename, index=False)

    evaluations_filename = os.path.join(reports_dir, "all_evaluations.html")
    all_evaluations_df = pd.DataFrame(evaluation_results['detailed_results'])
    all_evaluations_df.to_html(evaluations_filename, index=False)

    return summary_df, all_evaluations_df, reports_dir