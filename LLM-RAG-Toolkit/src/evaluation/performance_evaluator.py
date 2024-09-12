import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your fine-tuned model ID
fine_tuned_model_id = "ft:gpt-4o-mini-2024-07-18:personal::A4X2MjrJ"

# Recent successful fine-tuning job ID
fine_tune_job_id = "ftjob-Y2YiiDDsFd8pQFu64dvhCFf7"

def get_fine_tuned_model_info(model_id):
    try:
        response = client.models.retrieve(model_id)
        print(f"Model Info: {response}")
        return response
    except Exception as e:
        print(f"Error retrieving model info: {e}")

def get_fine_tuning_job_info(job_id):
    try:
        response = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Fine-Tuning Job Info:")
        print(f"  ID: {response.id}")
        print(f"  Status: {response.status}")
        print(f"  Created at: {response.created_at}")
        print(f"  Finished at: {response.finished_at}")
        print(f"  Model: {response.fine_tuned_model}")
        print(f"  Training file: {response.training_file}")
        return response
    except Exception as e:
        print(f"Error retrieving fine-tuning job info: {e}")

def list_fine_tune_events(job_id):
    try:
        response = client.fine_tuning.jobs.list_events(job_id, limit=10)
        print("Fine-Tuning Job Events:")
        for event in response:
            print(f"  {event.created_at}: {event.message}")
        return response
    except Exception as e:
        print(f"Error retrieving fine-tune events: {e}")

if __name__ == "__main__":
    # Retrieve information about your fine-tuned model
    fine_tune_info = get_fine_tuned_model_info(fine_tuned_model_id)

    # Retrieve fine-tuning job information
    job_info = get_fine_tuning_job_info(fine_tune_job_id)

    # List fine-tuning events
    list_fine_tune_events(fine_tune_job_id)
