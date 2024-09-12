import os
import logging
from typing import List, Dict, Any
from src.document_processing.document_loader import DocumentLoader
from src.document_processing.structure_analyzer import extract_toc, save_toc_to_json
from src.document_processing.content_segmenter import segment_pdf, save_segments_to_json
from src.document_processing.metadata_extractor import extract_metadata, save_metadata_to_json
from src.data_generation.synthetic_data_generator import QAGenerator, process_segment, analyze_dataset
from src.model_management.fine_tuner import estimate_cost, create_fine_tuning_job, monitor_fine_tuning
from utils.config_manager import config
from utils.logging_config import logger
import concurrent.futures

logger = logging.getLogger(__name__)

class WorkflowManager:
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.qa_generator = QAGenerator()
        logger.info("WorkflowManager initialized")

    def process_documents(self, input_directory: str) -> List[Dict[str, Any]]:
        logger.info(f"Processing documents in {input_directory}")
        processed_documents = []
        for filename in os.listdir(input_directory):
            file_path = os.path.join(input_directory, filename)
            try:
                document = self.document_loader.load_document(file_path)
                processed_document = self.process_single_document(document)
                processed_documents.append(processed_document)
                logger.info(f"Successfully processed {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
        return processed_documents

    def process_documents_parallel(self, input_directory: str, max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process all documents in the input directory in parallel."""
        processed_documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_document, os.path.join(input_directory, filename)): filename 
                              for filename in os.listdir(input_directory)}
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    processed_document = future.result()
                    processed_documents.append(processed_document)
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}", exc_info=True)
        return processed_documents

    def process_single_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document through all stages."""
        file_path = document['file_path']
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]

        # Extract TOC and structure
        toc_data = extract_toc(file_path)
        toc_output_file = os.path.join(config.file_paths['output_folder'], f"{base_name}_toc.json")
        save_toc_to_json(toc_data, toc_output_file)

        # Segment the document
        segments = segment_pdf(file_path, toc_output_file)
        segments_output_file = os.path.join(config.file_paths['output_folder'], f"{base_name}_segments.json")
        save_segments_to_json(segments, segments_output_file)

        # Extract metadata
        metadata = extract_metadata(file_path)
        metadata_output_file = os.path.join(config.file_paths['output_folder'], f"{base_name}_metadata.json")
        save_metadata_to_json(metadata, metadata_output_file)

        # Generate synthetic Q&A data
        qa_output_file = os.path.join(config.file_paths['output_folder'], f"{base_name}_qa_data.jsonl")
        for segment in segments:
            process_segment(self.qa_generator, segment['title'], segment['content'], 
                            config.generation_parameters['n_questions'], qa_output_file)

        return {
            "file_name": file_name,
            "toc_file": toc_output_file,
            "segments_file": segments_output_file,
            "metadata_file": metadata_output_file,
            "qa_file": qa_output_file
        }

    def prepare_training_data(self, processed_documents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Prepare training data for fine-tuning."""
        all_qa_data = []
        for doc in processed_documents:
            with open(doc['qa_file'], 'r') as f:
                all_qa_data.extend(f.readlines())

        # Shuffle and split the data
        random.shuffle(all_qa_data)
        split_index = int(len(all_qa_data) * config.training['train_ratio'])

        train_file = os.path.join(config.file_paths['output_folder'], "train_data.jsonl")
        val_file = os.path.join(config.file_paths['output_folder'], "val_data.jsonl")

        with open(train_file, 'w') as f:
            f.writelines(all_qa_data[:split_index])
        with open(val_file, 'w') as f:
            f.writelines(all_qa_data[split_index:])

        return {"train_file": train_file, "val_file": val_file}

    def run_fine_tuning(self, training_files: Dict[str, str]) -> str:
        """Run the fine-tuning process."""
        estimated_cost = estimate_cost(training_files['train_file'], training_files['val_file'])
        logger.info(f"Estimated cost for fine-tuning: ${estimated_cost:.2f}")

        confirm = input("Do you want to proceed with the fine-tuning? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Fine-tuning process aborted by user.")
            return

        train_file_id = upload_file(training_files['train_file'], "fine-tune")
        val_file_id = upload_file(training_files['val_file'], "fine-tune")
        fine_tune_id = create_fine_tuning_job(train_file_id, val_file_id)
        logger.info(f"Fine-tuning job created with ID: {fine_tune_id}")

        monitor_fine_tuning(fine_tune_id)
        return fine_tune_id

    def run_workflow(self):
        """Run the entire workflow."""
        try:
            processed_documents = self.process_documents(config.file_paths['input_directory'])
            training_files = self.prepare_training_data(processed_documents)
            fine_tune_id = self.run_fine_tuning(training_files)

            if fine_tune_id:
                logger.info(f"Workflow completed successfully. Fine-tuning job ID: {fine_tune_id}")
            else:
                logger.info("Workflow completed without fine-tuning.")

            # Analyze the generated dataset
            analyze_dataset(training_files['train_file'])

        except Exception as e:
            logger.error(f"Error in workflow execution: {e}", exc_info=True)

def main():
    logger.info("Starting LLM-RAG-Toolkit workflow")
    workflow = WorkflowManager()
    workflow.run_workflow()
    logger.info("LLM-RAG-Toolkit workflow completed")

if __name__ == "__main__":
    main()
