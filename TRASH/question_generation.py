import os
import streamlit as st
import logging
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai import OpenAI
from typing import List, Dict, Any
from api_logger import add_api_call
import openai
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable OpenAI library logging
openai.log = "debug"

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_synthetic_data(questions_per_node: int, model: str, progress_callback=None):
    """Generates synthetic questions and answers based on the knowledge base."""
    logger.info("Starting synthetic data generation")
    st.write("Starting synthetic data generation...")

    if 'knowledge_base' not in st.session_state or st.session_state['knowledge_base'] is None:
        logger.error("Knowledge base not found")
        st.error("Knowledge base not found. Please create a knowledge base first.")
        return 0, None

    # Initialize the OpenAI LLM
    llm = OpenAI(model=model)
    logger.info(f"Initialized OpenAI LLM with model: {model}")
    st.write(f"Using OpenAI model: {model}")

    # Get nodes from the knowledge base and convert to list
    nodes = list(st.session_state['knowledge_base'].docstore.docs.values())
    logger.info(f"Retrieved {len(nodes)} nodes from the knowledge base")
    st.write(f"Retrieved {len(nodes)} nodes from the knowledge base")

    # Create the dataset generator
    dataset_generator = RagDatasetGenerator(
        nodes=nodes,
        llm=llm,
        num_questions_per_chunk=questions_per_node
    )
    logger.info(f"Created RagDatasetGenerator with {questions_per_node} questions per node")
    st.write(f"Created RagDatasetGenerator with {questions_per_node} questions per node")

    # Generate the dataset
    logger.info("Generating dataset from nodes")
    st.write("Generating dataset from nodes...")
    rag_dataset = dataset_generator.generate_dataset_from_nodes()
    logger.info(f"Generated dataset with {len(rag_dataset.examples)} examples")
    st.write(f"Generated dataset with {len(rag_dataset.examples)} examples")

    # Convert the dataset to a list of dictionaries and check relevance
    dataset = []
    total_examples = len(rag_dataset.examples)
    for i, example in enumerate(rag_dataset.examples):
        qa_pair = {
            "question": example.query,
            "answer": example.reference_answer,
            "context": example.reference_contexts[0] if example.reference_contexts else ""
        }
        logger.info(f"Checking relevance for Q&A pair {i+1}")
        st.write(f"Checking relevance for Q&A pair {i+1}...")
        if check_relevance(qa_pair, llm):
            dataset.append(qa_pair)
            logger.info(f"Q&A pair {i+1} is relevant")
            st.write(f"Q&A pair {i+1} is relevant")
        else:
            logger.info(f"Q&A pair {i+1} is not relevant")
            st.write(f"Q&A pair {i+1} is not relevant")
        if progress_callback:
            progress = min(100, (i + 1) / total_examples * 100)
            progress_callback(progress)

    update_session_state('synthetic_qa_pairs', dataset)
    update_session_state('rag_dataset', rag_dataset)
    
    logger.info(f"Finished generating {len(dataset)} relevant Q&A pairs")
    st.write(f"Finished generating {len(dataset)} relevant Q&A pairs")
    return len(dataset), dataset

def check_relevance(qa_pair: Dict[str, str], llm: OpenAI) -> bool:
    """Check if the generated Q&A pair is relevant to the context."""
    prompt = f"""
    Given the following context, question, and answer, determine if the Q&A pair is relevant and appropriate.
    
    Context: {qa_pair['context']}
    
    Question: {qa_pair['question']}
    
    Answer: {qa_pair['answer']}
    
    Is this Q&A pair relevant to the context and appropriate? Answer with 'Yes' or 'No' and provide a brief explanation.
    """
    
    try:
        response = llm.complete(prompt)
        
        # Log the full prompt and response
        logger.info(f"Relevance Check Prompt:\n{prompt}")
        logger.info(f"Relevance Check Response:\n{response}")
        
        add_api_call("Relevance Check", f"Prompt: {prompt}\nResponse: {response}")
        
        # Check if the response text starts with 'yes'
        is_relevant = response.text.lower().startswith('yes')
        logger.info(f"Relevance check result: {'Relevant' if is_relevant else 'Not relevant'}")
        return is_relevant
    except Exception as e:
        logger.error(f"Error in check_relevance: {str(e)}")
        return False

def update_session_state(key, value):
    """Updates the Streamlit session state."""
    if isinstance(key, list):
        key = tuple(tuple(item.items()) for item in key)
    st.session_state[key] = value

def log_dataset_structure(dataset: List[Dict[str, str]]):
    """Logs the structure of the generated dataset."""
    st.subheader("Generated Dataset Structure")
    st.text(f"Number of QA pairs: {len(dataset)}")
    if dataset:
        st.text("Example QA pair:")
        st.json(dataset[0])

def save_rag_dataset(rag_dataset: LabelledRagDataset, filename: str = "rag_dataset.json"):
    """Saves the RAG dataset to a JSON file."""
    rag_dataset.save_json(filename)
    st.success(f"RAG dataset saved to {filename}")

def load_rag_dataset(filename: str = "rag_dataset.json") -> LabelledRagDataset:
    """Loads a RAG dataset from a JSON file."""
    return LabelledRagDataset.from_json(filename)

def format_for_fine_tuning(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Formats the dataset for fine-tuning."""
    formatted_data = []
    for item in dataset:
        formatted_data.append({
            "prompt": f"Question: {item['question']}\nContext: {item['context']}\nAnswer:",
            "completion": f" {item['answer']}"
        })
    return formatted_data

def save_for_fine_tuning(dataset: List[Dict[str, str]], filename: str = "fine_tuning_data.jsonl"):
    """Saves the formatted dataset for fine-tuning."""
    import json
    formatted_data = format_for_fine_tuning(dataset)
    with open(filename, 'w') as f:
        for item in formatted_data:
            json.dump(item, f)
            f.write('\n')
    st.success(f"Fine-tuning dataset saved to {filename}")

@lru_cache(maxsize=None)
def cached_generate_synthetic_data(questions_per_node: int, model: str):
    """Cached version of generate_synthetic_data."""
    return generate_synthetic_data(questions_per_node, model)

def generate_questions():
    # ... (previous code)
    
    if st.button("Generate Synthetic Data"):
        # Use the cached version
        num_qa_pairs, dataset = cached_generate_synthetic_data(questions_per_node, model)
        
        # ... (rest of the function)
