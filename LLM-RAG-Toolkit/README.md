# LLM-RAG-Toolkit

LLM-RAG-Toolkit is a comprehensive solution for processing documents, building Retrieval-Augmented Generation (RAG) systems, generating synthetic data, and fine-tuning language models.

## Features

- Document Processing: Support for PDF and image files
- Content Segmentation: Intelligent segmentation of document content
- RAG System Building: Create embeddings and build efficient retrieval systems
- Synthetic Data Generation: Generate high-quality synthetic Q&A pairs
- Model Fine-tuning: Fine-tune language models on processed data
- Web Interface: User-friendly Streamlit interface for easy interaction

## Installation

1. Clone the repository:
   ```
      git clone https://github.com/MDBillingspecilist/LLM-RAG-Toolkit.git
      cd LLM-RAG-Toolkit

   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key in the `config.yaml` file or as an environment variable.

## Usage

1. Start the Streamlit web interface:
   ```
   streamlit run src/web_interface/app.py
   ```

2. Follow the steps in the web interface to:
   - Upload documents
   - Process documents
   - Segment content
   - Build RAG systems
   - Generate synthetic data
   - Fine-tune models

## Project Structure

- `src/`: Source code for the toolkit
  - `document_processing/`: Document processing modules
  - `rag_system/`: RAG system building modules
  - `data_generation/`: Synthetic data generation modules
  - `model_management/`: Model fine-tuning modules
  - `web_interface/`: Streamlit web interface
- `utils/`: Utility functions and configurations
- `tests/`: Unit tests
- `data/`: Directory for input and output data
  - `raw/`: Raw input documents
  - `processed/`: Processed and generated data

## Configuration

Adjust the settings in `config/config.yaml` to customize the toolkit's behavior, including file paths, model parameters, and processing options.

## Testing

Run the unit tests using: