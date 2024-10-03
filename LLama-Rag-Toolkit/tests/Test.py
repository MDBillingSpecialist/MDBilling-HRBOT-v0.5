import os
import openai
import glob
from pathlib import Path
import nest_asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='../config/.env')

# Setup
openai.api_key = os.getenv("OPENAI_API_KEY")
nest_asyncio.apply()

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptHelper,
    Document,
)
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from langchain_community.chat_models import ChatOpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import CorrectnessEvaluator, BatchEvalRunner


# Load documents from folder
def load_pdf(file_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return Document(text, doc_id=Path(file_path).name)

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return Document(text, doc_id=Path(file_path).name)

def load_html(file_path):
    UnstructuredReader = download_loader("UnstructuredReader")
    loader = UnstructuredReader()
    docs = loader.load_data(file=Path(file_path), split_documents=False)
    for d in docs:
        d.doc_id = Path(file_path).name
    return docs

def load_documents_from_folder(folder_path):
    document_list = []
    for file_path in glob.glob(f"{folder_path}/*"):
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            document_list.append(load_pdf(file_path))
        elif ext in ['.txt', '.md']:
            document_list.append(load_text(file_path))
        elif ext in ['.html', '.htm']:
            document_list.extend(load_html(file_path))
        else:
            print(f"Unsupported file format: {file_path}")
    return document_list

folder_path = './data/your_documents'  # Replace with your folder path
documents = load_documents_from_folder(folder_path)

# Create nodes
node_parser = SimpleNodeParser()
nodes = node_parser.get_nodes_from_documents(documents)

# Configure settings
max_context_window = 128000
num_output = 4096
chunk_size_limit = 600

prompt_helper = PromptHelper(
    context_window=max_context_window,
    num_output=num_output,
    chunk_size_limit=chunk_size_limit,
)

# Initialize the LLM directly
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    max_tokens=num_output,
    temperature=0
)

# Set up global settings with LLM and embedding model
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002")
Settings.prompt_helper = prompt_helper

# Create index
storage_context = StorageContext.from_defaults()
index = VectorStoreIndex.from_documents(
    documents=nodes,
    storage_context=storage_context,
    embed_model=Settings.embed_model,
)
storage_context.persist(persist_dir="./storage/my_documents")

# Load index (if needed)
storage_context = StorageContext.from_defaults(persist_dir="./storage/my_documents")
index = load_index_from_storage(
    storage_context=storage_context, embed_model=Settings.embed_model
)
query_engine = index.as_query_engine(llm=Settings.llm)

# Create agent
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="document_query_engine",
        description="Useful for answering questions based on your custom documents.",
    ),
)
tools = [query_engine_tool]
agent = OpenAIAgent.from_tools(tools, verbose=True)

# Chatbot loop
print("You can now interact with your custom document chatbot. Type 'exit' to quit.\n")
while True:
    text_input = input("User: ")
    if text_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}\n")