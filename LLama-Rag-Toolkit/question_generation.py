import streamlit as st
import pandas as pd
import os
import datetime
import json
from llama_index.core.node_parser import SimpleFileNodeParser, SemanticSplitterNodeParser, HierarchicalNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

def generate_synthetic_data(questions_per_chunk, parser_choice="Simple", model="gpt-4o-mini"):
    try:
        st.info(f"Generating synthetic data using {parser_choice} parser and {model} model...")
        
        # Choose the appropriate parser based on user selection
        if parser_choice == "Simple":
            node_parser = SimpleFileNodeParser.from_defaults()
        elif parser_choice == "Semantic":
            embed_model = OpenAIEmbedding()
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95, 
                embed_model=embed_model
            )
        elif parser_choice == "Hierarchical":
            node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            )
        else:
            raise ValueError(f"Unknown parser choice: {parser_choice}")

        all_nodes = []
        for doc_id, doc_list in st.session_state['parsed_documents'].items():
            st.text(f"Processing document: {doc_id}")
            nodes = node_parser.get_nodes_from_documents(doc_list)
            all_nodes.extend(nodes)

        st.text(f"Total nodes generated: {len(all_nodes)}")

        llm = OpenAI(model=model)

        st.text("Generating dataset...")
        from llama_index.core.llama_dataset.generator import RagDatasetGenerator
        dataset_generator = RagDatasetGenerator(
            nodes=all_nodes,
            num_questions_per_chunk=questions_per_chunk,
            llm=llm
        )

        new_dataset = dataset_generator.generate_dataset_from_nodes()
        
        if 'dataset' not in st.session_state:
            st.session_state['dataset'] = LabelledRagDataset(examples=[])
        
        st.session_state['dataset'].examples.extend(new_dataset.examples)
        return len(new_dataset.examples)
        
    except Exception as e:
        st.error(f"Error generating synthetic data: {str(e)}")
        return 0

def save_dataset(dataset: LabelledRagDataset, filename="generated_dataset.json"):
    dataset.save_json(filename)

def load_dataset(filename="generated_dataset.json") -> LabelledRagDataset:
    try:
        return LabelledRagDataset.from_json(filename)
    except FileNotFoundError:
        return LabelledRagDataset(examples=[])

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

    # Save summary to HTML
    summary_filename = os.path.join(reports_dir, "evaluation_summary.html")
    summary_df.to_html(summary_filename, index=False)

    # Save all evaluations to HTML
    evaluations_filename = os.path.join(reports_dir, "all_evaluations.html")
    all_evaluations_df = pd.DataFrame(all_evaluations)
    all_evaluations_df.to_html(evaluations_filename, index=False)

    return summary_df, all_evaluations_df, reports_dir

def export_qa_dataset(dataset: LabelledRagDataset, filename="qa_dataset.json"):
    qa_pairs = []
    for example in dataset.examples:
        qa_pairs.append({
            "question": example.query,
            "answer": example.reference_answer,
            "contexts": example.reference_contexts
        })
    
    with open(filename, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    
    return filename