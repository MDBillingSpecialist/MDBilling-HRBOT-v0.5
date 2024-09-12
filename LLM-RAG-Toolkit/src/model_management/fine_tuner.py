import os
import time
import logging
from typing import Dict, Any, List  # Add List to the import
from openai import OpenAI
from utils.config_manager import config

logger = logging.getLogger(__name__)

client = OpenAI(api_key=config.OPENAI_API_KEY)

def estimate_cost(train_file_path: str, val_file_path: str, price_per_token: float = config.training['price_per_token']) -> float:
    """
    Estimate the cost of fine-tuning based on the number of tokens in the training and validation files.
    """
    try:
        with open(train_file_path, 'r') as train_file, open(val_file_path, 'r') as val_file:
            train_tokens = sum(len(line.split()) for line in train_file)
            val_tokens = sum(len(line.split()) for line in val_file)
        
        total_tokens = train_tokens + val_tokens
        estimated_cost = total_tokens * price_per_token
        return estimated_cost
    except Exception as e:
        logger.error(f"Failed to estimate cost: {e}", exc_info=True)
        raise

def upload_file(file_path: str, purpose: str) -> str:
    """
    Upload a file to OpenAI for fine-tuning.
    """
    try:
        logger.info(f"Uploading file {file_path} for purpose: {purpose}")
        with open(file_path, "rb") as file:
            response = client.files.create(file=file, purpose=purpose)
        logger.info(f"File uploaded successfully: {response.id}")
        return response.id
    except Exception as e:
        logger.error(f"Failed to upload file {file_path}: {e}", exc_info=True)
        raise

def create_fine_tuning_job(train_file_id: str, val_file_id: str) -> str:
    """
    Create a fine-tuning job with the uploaded files.
    """
    try:
        logger.info("Creating fine-tuning job.")
        response = client.fine_tuning.jobs.create(
            training_file=train_file_id,
            validation_file=val_file_id,
            model=config.training['model_name'],
            hyperparameters={"n_epochs": config.training['n_epochs']}
        )
        logger.info(f"Fine-tuning job created successfully: {response.id}")
        return response.id
    except Exception as e:
        logger.error(f"Failed to create fine-tuning job: {e}", exc_info=True)
        raise

def monitor_fine_tuning(fine_tune_id: str, poll_interval: int = 60) -> None:
    """
    Monitor the progress of a fine-tuning job.
    """
    while True:
        try:
            status = client.fine_tuning.jobs.retrieve(fine_tune_id)
            logger.info(f"Status: {status.status}")
            print(f"Status: {status.status}")

            if status.status in ['succeeded', 'failed']:
                print(f"Fine-tuned model: {getattr(status, 'fine_tuned_model', 'Not yet completed')}")
                logger.info(f"Fine-tuned model: {getattr(status, 'fine_tuned_model', 'Not yet completed')}")

                if status.result_files:
                    print(f"Result Files: {status.result_files}")
                    logger.info(f"Result Files: {status.result_files}")
                else:
                    print("No result files yet.")
                    logger.info("No result files yet.")

                if status.status == 'failed':
                    print(f"Error: {status.error}")
                    logger.error(f"Error: {status.error}")

                break
            else:
                if status.trained_tokens is not None:
                    print(f"Trained tokens: {status.trained_tokens}")
                    logger.info(f"Trained tokens: {status.trained_tokens}")
                else:
                    print("Token information is not available yet.")
                    logger.info("Token information is not available yet.")
                
                print("Fine-tuning is still in progress...")
                logger.info("Fine-tuning is still in progress...")
                time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Error during monitoring: {e}", exc_info=True)
            print(f"Error during monitoring: {e}")
            time.sleep(poll_interval)

def get_fine_tuned_model_info(model_id: str) -> Dict[str, Any]:
    """
    Retrieve information about a fine-tuned model.
    """
    try:
        response = client.models.retrieve(model_id)
        logger.info(f"Model Info: {response}")
        return response
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}", exc_info=True)
        raise

def get_fine_tuning_job_info(job_id: str) -> Dict[str, Any]:
    """
    Retrieve information about a fine-tuning job.
    """
    try:
        response = client.fine_tuning.jobs.retrieve(job_id)
        logger.info(f"Fine-Tuning Job Info: {response}")
        return response
    except Exception as e:
        logger.error(f"Error retrieving fine-tuning job info: {e}", exc_info=True)
        raise

def list_fine_tune_events(job_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    List events for a fine-tuning job.
    """
    try:
        response = client.fine_tuning.jobs.list_events(job_id, limit=limit)
        logger.info("Fine-Tuning Job Events retrieved successfully")
        return list(response)
    except Exception as e:
        logger.error(f"Error retrieving fine-tune events: {e}", exc_info=True)
        raise

def fine_tune_model(synthetic_data: Dict[str, Any]) -> str:
    try:
        # Assuming synthetic_data contains file paths for train and validation data
        train_file_id = upload_file(synthetic_data['train_file'], "fine-tune")
        val_file_id = upload_file(synthetic_data['val_file'], "fine-tune")
        
        job_id = create_fine_tuning_job(train_file_id, val_file_id)
        logger.info(f"Fine-tuning job created with ID: {job_id}")
        
        # Monitor the job until it's completed
        status = "running"
        while status == "running":
            status = monitor_fine_tuning(job_id)
            logger.info(f"Fine-tuning status: {status}")
        
        if status == "succeeded":
            logger.info("Fine-tuning completed successfully")
            return job_id
        else:
            logger.error(f"Fine-tuning failed with status: {status}")
            return None
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
        return None
