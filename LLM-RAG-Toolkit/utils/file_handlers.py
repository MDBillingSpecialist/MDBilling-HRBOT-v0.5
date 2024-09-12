import json
import random
import logging
from typing import List, Dict, Any
from utils.config_manager import config

logger = logging.getLogger(__name__)

# Paths to the input and output files
input_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/synthetic_data_v4.jsonl'
train_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/train_data.jsonl'
val_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/val_data.jsonl'
test_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/test_data.jsonl'

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def convert_jsonl_and_split(input_file: str, train_file: str, val_file: str, test_file: str) -> None:
    """
    Convert JSONL file and split it into train, validation, and test sets.
    """
    try:
        data = []
        with open(input_file, 'r') as infile:
            for line in infile:
                data_entry = json.loads(line.strip())
                processed_entry = process_data_entry(data_entry)
                if processed_entry:
                    data.append(processed_entry)

        if not data:
            logger.warning("No valid data found to process.")
            return

        # Shuffle and split the data
        random.shuffle(data)
        train_size = int(len(data) * config.training['train_ratio'])
        val_size = int(len(data) * config.training['val_ratio'])

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        write_jsonl(train_file, train_data)
        write_jsonl(val_file, val_data)
        write_jsonl(test_file, test_data)

        logger.info("Data conversion and splitting completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred during conversion: {e}", exc_info=True)
        raise

def process_data_entry(data_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single data entry and return a formatted chat completion.
    """
    segment = data_entry.get('segment')
    question = data_entry.get('question')

    if not segment or not question:
        logger.warning(f"Skipping entry with missing 'segment' or 'question': {data_entry}")
        return None

    responses = data_entry.get('responses', {})
    if not responses:
        logger.warning(f"Skipping entry with missing 'responses': {data_entry}")
        return None

    if data_entry.get('multi_turn', False):
        messages = [
            {"role": "system", "content": config.system_message},
        ]
        for turn in data_entry['conversation']:
            messages.append({"role": turn['role'], "content": turn['content']})
    else:
        best_response = max(responses.values(), key=lambda x: x['similarity_score'])['response']
        messages = [
            {"role": "system", "content": config.system_message},
            {"role": "user", "content": f"Segment: {segment}\nQuestion: {question}"},
            {"role": "assistant", "content": best_response}
        ]

    return {"messages": messages}

def write_jsonl(file_path: str, dataset: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    """
    with open(file_path, 'w') as outfile:
        for entry in dataset:
            json.dump(entry, outfile)
            outfile.write('\n')

def validate_jsonl(file_path: str) -> bool:
    """
    Validate the structure of a JSONL file.
    """
    try:
        with open(file_path, 'r') as infile:
            for i, line in enumerate(infile):
                try:
                    entry = json.loads(line.strip())
                    if not validate_entry(entry):
                        logger.error(f"Invalid entry structure in line {i + 1}")
                        return False
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding error in line {i + 1}: {e}")
                    return False

        logger.info(f"Validation successful for file: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"An error occurred during validation: {e}", exc_info=True)
        return False

def validate_entry(entry: Dict[str, Any]) -> bool:
    """
    Validate the structure of a single entry in the JSONL file.
    """
    if 'messages' not in entry:
        logger.error("Missing 'messages' key")
        return False

    messages = entry['messages']
    if not isinstance(messages, list) or len(messages) < 2:
        logger.error("'messages' should be a list with at least 2 items")
        return False

    required_roles = ['system', 'user']
    roles = [msg['role'] for msg in messages[:2]]
    if roles != required_roles:
        logger.error(f"Incorrect message roles. Expected {required_roles}, got {roles}")
        return False

    for msg in messages:
        if not isinstance(msg.get('content'), str):
            logger.error("Message content is not a string")
            return False

    return True

# Run the updated conversion function
convert_jsonl_and_split(input_file, train_file, val_file, test_file)

# Validate the resulting files
validate_jsonl(train_file)
validate_jsonl(val_file)
validate_jsonl(test_file)