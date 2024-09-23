# LLama RAG Toolkit Documentation

## Table of Contents
1. Introduction
2. System Architecture
3. Module Descriptions
4. Workflow
5. Configuration
6. Best Practices
7. Troubleshooting
8. API Reference

## 1. Introduction
The LLama RAG Toolkit is a comprehensive solution for implementing a Retrieval-Augmented Generation (RAG) system using LlamaIndex and Streamlit. It provides a user-friendly interface for processing documents, creating a knowledge base, generating synthetic questions, and evaluating the system's performance.

## 2. System Architecture
The toolkit is built on a modular architecture, with each component handling a specific part of the RAG pipeline:

- Document Processing: Handles file uploads and initial parsing
- Knowledge Base: Creates and manages the vector store index
- Question Generation: Produces synthetic questions based on the knowledge base
- Evaluation: Assesses the quality of generated answers

The system uses Streamlit for the user interface, LlamaIndex for document indexing and retrieval, and OpenAI's language models for question generation and answering.

## 3. Module Descriptions

### 3.1 document_processing.py
This module is responsible for handling document uploads and initial parsing. It uses LlamaIndex's SimpleDirectoryReader and SimpleNodeParser to process documents and create initial nodes.

Key functions:
- `process_documents()`: Handles file uploads and parsing
- `display_parsed_documents()`: Shows the parsed document structure

### 3.2 knowledge_base.py
This module manages the creation and operations of the knowledge base. It uses LlamaIndex's VectorStoreIndex and various extractors to create a sophisticated knowledge base.

Key functions:
- `create_knowledge_base()`: Generates the knowledge base from parsed documents
- `create_transformations()`: Sets up the ingestion pipeline with various extractors
- `export_knowledge_base()`: Saves the knowledge base to disk
- `load_knowledge_base()`: Loads a previously saved knowledge base

### 3.3 question_generation.py
This module handles the generation of synthetic questions based on the knowledge base. It uses OpenAI's language models to create question-answer pairs.

Key functions:
- `generate_synthetic_data()`: Produces synthetic questions and answers
- `generate_dataset()`: Creates a LabelledRagDataset
- `process_examples()`: Applies custom prompts to generated questions

### 3.4 evaluation.py
This module is responsible for evaluating the performance of the RAG system. It uses LlamaIndex's evaluation metrics to assess the quality of generated answers.

Key functions:
- `evaluate_responses()`: Evaluates generated responses against reference answers
- `evaluate_rag_system()`: Runs a comprehensive evaluation of the RAG system

## 4. Workflow
1. Document Processing: Users upload documents, which are parsed into nodes.
2. Knowledge Base Creation: Parsed nodes are processed through an ingestion pipeline to create a sophisticated knowledge base.
3. Question Generation: Synthetic questions are generated based on the knowledge base.
4. Evaluation: The system's performance is evaluated using generated questions and answers.

## 5. Configuration
The system's configuration is managed through the `config.py` file and environment variables. Key configurations include:
- OpenAI API key (set in .env file)
- Model selections for various tasks
- Chunk sizes and overlaps for document parsing

## 6. Best Practices
- Regularly update the knowledge base as new documents are added
- Experiment with different chunk sizes and overlaps for optimal performance
- Use a variety of document types to create a comprehensive knowledge base
- Regularly evaluate and fine-tune the system based on performance metrics

## 7. Troubleshooting
Common issues and their solutions:
- API key errors: Ensure your OpenAI API key is correctly set in the .env file
- Memory issues: For large documents, consider increasing Streamlit's memory limit
- Parsing errors: Check document formats and encoding; consider pre-processing problematic documents

## 8. API Reference
For detailed API references, please refer to the docstrings in individual module files.