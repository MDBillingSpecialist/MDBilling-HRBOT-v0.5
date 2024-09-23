# LLama RAG Toolkit

## Overview
The LLama RAG Toolkit is a powerful Streamlit-based application that implements a Retrieval-Augmented Generation (RAG) system using LlamaIndex. This toolkit allows users to process documents, create a knowledge base, generate synthetic questions, and evaluate the RAG system's performance.

## Features
- Document Processing: Upload and parse various document types (PDF, TXT, DOCX)
- Knowledge Base Creation: Generate a sophisticated knowledge base from parsed documents
- Synthetic Data Generation: Create question-answer pairs based on the knowledge base
- RAG System Evaluation: Assess the performance of the generated questions and answers

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LLama-Rag-Toolkit.git
   cd LLama-Rag-Toolkit
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key in a `.env` file:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage
Run the Streamlit app:
```
streamlit run main.py