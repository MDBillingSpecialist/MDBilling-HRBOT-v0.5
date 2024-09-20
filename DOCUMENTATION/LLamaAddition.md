Evaluating RAG Pipelines with LabelledRagDataset
In this guide, we'll explore how to use the LabelledRagDataset from the llama_index library to evaluate Retrieval-Augmented Generation (RAG) pipelines. We'll cover how to create a dataset, generate synthetic data, and use it to benchmark your RAG system.

Table of Contents
Introduction
Creating a LabelledRagDataset
Constructing Examples Manually
Generating Synthetic Data
Using the LabelledRagDataset
Evaluating Your RAG System
Saving and Loading the Dataset
Contributing to the Community
Resources
Introduction
Evaluating a RAG system requires a dataset that contains queries, reference answers, and reference contexts. The LabelledRagDataset abstraction in the llama_index library facilitates this by providing a structured way to create and use such datasets.

Creating a LabelledRagDataset
Constructing Examples Manually
You can manually create a LabelledRagDataset by constructing LabelledRagDataExample instances:

python
Copy code
from llama_index.core.llama_dataset import (
    LabelledRagDataExample,
    LabelledRagDataset,
    CreatedBy,
    CreatedByType,
)

# Create individual examples
example1 = LabelledRagDataExample(
    query="What is the capital of France?",
    query_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_answer="Paris is the capital of France.",
    reference_contexts=[
        "France is a country in Western Europe. Its capital is Paris."
    ],
    reference_by=CreatedBy(type=CreatedByType.HUMAN),
)

example2 = LabelledRagDataExample(
    query="Who wrote 'Pride and Prejudice'?",
    query_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_answer="Jane Austen wrote 'Pride and Prejudice'.",
    reference_contexts=[
        "'Pride and Prejudice' is a novel by Jane Austen, published in 1813."
    ],
    reference_by=CreatedBy(type=CreatedByType.HUMAN),
)

# Create the dataset
rag_dataset = LabelledRagDataset(examples=[example1, example2])
Generating Synthetic Data
Alternatively, you can generate synthetic datasets using strong LLMs:

python
Copy code
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai import OpenAI

# Initialize your LLM (e.g., OpenAI's GPT-3.5-turbo)
llm = OpenAI(model="gpt-3.5-turbo")

# Assume you have a list of documents
documents = [...]  # Replace with your documents

# Create the dataset generator
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=llm,
    num_questions_per_chunk=5,  # Number of questions per document chunk
)

# Generate the dataset
rag_dataset = dataset_generator.generate_dataset_from_nodes()
Using the LabelledRagDataset
Once you have the dataset, you can inspect it or convert it to a pandas DataFrame for easier viewing:

python
Copy code
# Convert to pandas DataFrame
df = rag_dataset.to_pandas()
print(df)
Evaluating Your RAG System
To evaluate your RAG system, you'll need to generate predictions and compare them against the reference answers:

python
Copy code
from llama_index.core.llama_pack import download_llama_pack

# Download the evaluator pack
RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")

# Initialize the evaluator with your query engine and dataset
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine,  # Your RAG system
    rag_dataset=rag_dataset,
)

# Run the evaluation
benchmark_df = await rag_evaluator.run()

# View the evaluation results
print(benchmark_df)
The benchmark_df will contain metrics such as Correctness, Relevancy, and Faithfulness, allowing you to assess your system's performance.

Saving and Loading the Dataset
You can save your dataset to a JSON file and load it later:

python
Copy code
# Save the dataset
rag_dataset.save_json("rag_dataset.json")

# Load the dataset
from llama_index.core.llama_dataset import LabelledRagDataset
loaded_dataset = LabelledRagDataset.from_json("rag_dataset.json")
Contributing to the Community
You can contribute your LabelledRagDataset to the community by submitting it to llamahub. This involves:

Creating the dataset and saving it as a JSON file.
Submitting both the JSON file and the source text files to the llama_datasets GitHub repository.
Making a pull request to update the metadata in the llama_hub GitHub repository.
Refer to the "LlamaDataset Submission Template Notebook" for detailed instructions.

Resources
Labelled RAG Datasets on LlamaHub
Downloading Llama Datasets
Contributing a LabelledRagDataset