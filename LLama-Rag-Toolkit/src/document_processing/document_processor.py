import asyncio
import aiofiles
import fitz  # PyMuPDF
from typing import List, Optional, Tuple, Any, Union
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.node_parser.topic import TopicNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from src.utils.helpers import generate_doc_id, get_current_time
from src.utils.logger import setup_logger
import tempfile
import os
import hashlib
from collections import defaultdict
import time
from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = setup_logger(__name__)

MAX_CHARS_PER_CHUNK = 2000
MIN_CHARS_FOR_TOPIC_PARSER = 500

# Add these near the top of the file, after the imports
api_call_count = defaultdict(int)
processing_times = defaultdict(float)
document_cache = {}
nodes_per_document = {}

class CaptureOpenAI(OpenAI):
    def __init__(self, model: str, **kwargs: Any):
        super().__init__(model=model, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        logger.info(f"API Call to OpenAI: model={self.model}, prompt={prompt[:100]}...")
        response = super().complete(prompt, **kwargs)
        logger.info(f"API Response: {response.text[:100]}...")
        # Update API call count
        api_call_count["OpenAI.complete"] += 1
        return response

async def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text content from a PDF file asynchronously."""
    text = ""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        if doc.is_encrypted:
            logger.warning(f"PDF is encrypted. Attempting to decrypt...")
            success = doc.authenticate("")  # Try empty password
            if not success:
                raise ValueError(f"PDF is encrypted and could not be decrypted.")
        for page in doc:
            text += page.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
    return text

def create_document(content: str, file_name: str) -> Document:
    """Create a LlamaIndex Document object from content."""
    return Document(
        text=content,
        metadata={
            "file_name": file_name,
            "doc_id": generate_doc_id(content),
            "created_at": get_current_time()
        }
    )

async def process_single_file(file_content: bytes, file_name: str) -> Optional[Document]:
    """Process a single file and return a Document object."""
    logger.info(f"Processing file: {file_name}")
    try:
        content = await extract_text_from_pdf(file_content)
        if not content.strip():
            logger.warning(f"No text content extracted from {file_name}")
            return None
        doc = create_document(content, file_name)
        return doc
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {str(e)}")
        return None

async def process_files(files: List[bytes], file_names: List[str]) -> List[Document]:
    """Process all files and return a list of Document objects."""
    tasks = [process_single_file(file, name) for file, name in zip(files, file_names)]
    documents = await asyncio.gather(*tasks)
    return [doc for doc in documents if doc is not None]

def create_topic_node_parser() -> TopicNodeParser:
    """Create a TopicNodeParser with the required configurations."""
    llm = CaptureOpenAI(model=Settings.llm.model)
    return TopicNodeParser.from_defaults(
        llm=llm,
        max_chunk_size=500,
        similarity_method="llm",
        window_size=2
    )

def parse_document(doc: Document) -> List[Document]:
    """Parse a single Document into smaller chunks."""
    simple_parser = SimpleNodeParser(chunk_size=MAX_CHARS_PER_CHUNK, chunk_overlap=100)
    topic_parser = create_topic_node_parser()
    
    if len(doc.text) > MAX_CHARS_PER_CHUNK:
        sentence_splitter = SentenceSplitter(chunk_size=MAX_CHARS_PER_CHUNK, chunk_overlap=100)
        chunks = sentence_splitter.split_text(doc.text)
        parsed_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(text=chunk, metadata={**doc.metadata, "chunk": i})
            if len(chunk) > MIN_CHARS_FOR_TOPIC_PARSER:
                chunk_nodes = topic_parser.get_nodes_from_documents([chunk_doc])
                api_call_count["TopicNodeParser"] += 1
            else:
                chunk_nodes = simple_parser.get_nodes_from_documents([chunk_doc])
                api_call_count["SimpleNodeParser"] += 1
            parsed_docs.extend(chunk_nodes)
        api_call_count["LargeDocumentSplitter"] += 1
    elif len(doc.text) > MIN_CHARS_FOR_TOPIC_PARSER:
        api_call_count["TopicNodeParser"] += 1
        parsed_docs = topic_parser.get_nodes_from_documents([doc])
    else:
        api_call_count["SimpleNodeParser"] += 1
        parsed_docs = simple_parser.get_nodes_from_documents([doc])
    
    nodes_per_document[doc.metadata.get('file_name', 'Unknown')] = len(parsed_docs)
    return parsed_docs

def parse_documents(documents: List[Document]) -> List[Document]:
    """Parse all documents into smaller chunks."""
    parsed_documents = []
    for doc in documents:
        parsed_documents.extend(parse_document(doc))
    return parsed_documents

# Main processing function
async def process_and_parse_documents(file_paths: List[str], file_names: List[str]) -> List[Document]:
    documents = await process_files(file_paths, file_names)
    return parse_documents(documents)

async def process_and_index_documents(files: List[Union[str, UploadedFile]], file_names: List[str]) -> Tuple[List[Document], int]:
    """Process uploaded files, parse documents, and prepare for indexing."""
    start_time = time.time()
    logger.info("Starting document processing and indexing")
    
    documents = await process_files(files, file_names)
    if not documents:
        logger.error("No documents processed or processing failed.")
        return [], 0

    logger.info(f"Processed {len(documents)} documents")

    parsed_documents = parse_documents(documents)
    total_nodes = sum(nodes_per_document.values())
    
    logger.info(f"Parsed {len(parsed_documents)} documents, created {total_nodes} nodes")

    end_time = time.time()
    processing_times["Total Processing Time"] = end_time - start_time

    return parsed_documents, total_nodes

# Add this at the end of the file
__all__ = ['process_and_parse_documents', 'process_and_index_documents', 'api_call_count', 'processing_times', 'document_cache', 'nodes_per_document', 'MAX_CHARS_PER_CHUNK', 'process_single_file']