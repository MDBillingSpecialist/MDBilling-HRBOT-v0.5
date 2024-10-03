Core Imports and Modules for LabelledRagDataset Evaluation:
python
Copy code
# LlamaIndex Core Modules
from llama_index.core import Document, VectorStoreIndex, PromptTemplate, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode

# LabelledRagDataset and Evaluation Modules
from llama_index.core.llama_dataset import (
    LabelledRagDataset,
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
    download_llama_dataset
)

# RagDatasetGenerator for generating LabelledRagDatasets from documents
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

# Ingestion Pipeline for processing document nodes
from llama_index.core.ingestion import IngestionPipeline

# RagEvaluatorPack for streamlined evaluation of RAG systems
from llama_index.core.llama_pack import download_llama_pack

# Dataset Generation for generating QA pairs from documents
from llama_index.core.evaluation.dataset_generation import DatasetGenerator

# Evaluation Tools
from llama_index.core.evaluation import QueryResponseDataset, CorrectnessEvaluator, BatchEvalRunner
from llama_index.core.evaluation.eval_utils import get_responses

# Async utilities
import nest_asyncio
nest_asyncio.apply()
Explanation of Modules:
Core Components:

Document: Represents individual documents in your system.
VectorStoreIndex: For building vector-based retrieval indices.
PromptTemplate: For defining and formatting prompts.
Settings: Configures global settings for LLMs.
SentenceSplitter: Splits documents into smaller chunks (nodes).
IndexNode: Represents individual nodes in a document.
LabelledRagDataset:

LabelledRagDataset: Holds the dataset of queries, reference answers, and contexts.
LabelledRagDataExample: Defines individual query/response examples.
CreatedBy and CreatedByType: Metadata for tracking who created the query/response (human or machine).
download_llama_dataset: For downloading pre-built datasets from LlamaHub.
RagDatasetGenerator:

RagDatasetGenerator: Automatically generates LabelledRagDatasets from documents using a language model.
IngestionPipeline:

IngestionPipeline: Processes and transforms documents into nodes for indexing or RAG.
RagEvaluatorPack:

RagEvaluatorPack: Simplifies evaluating the RAG system by assessing correctness, relevancy, faithfulness, and context similarity.
DatasetGenerator:

DatasetGenerator: Generates QA pairs or question-answer datasets from documents.
Evaluation Tools:

QueryResponseDataset: Holds the dataset for query-response evaluation.
CorrectnessEvaluator: Evaluates the correctness of generated responses.
BatchEvalRunner: Runs evaluation in batches for efficiency.
