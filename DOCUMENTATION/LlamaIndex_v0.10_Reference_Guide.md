
# LlamaIndex v0.10 Reference Guide

This document provides a comprehensive guide on the changes introduced in **LlamaIndex v0.10** as well as how to use key components like document loaders, chunking, indexing, and querying in your LlamaIndex-based applications. You’ll find examples, best practices, and detailed migration steps from earlier versions.

---

## Key Changes in LlamaIndex v0.10

### 1. Module Restructuring and New Imports
LlamaIndex v0.10 introduces significant changes to the structure of the library, with a clear separation between various functionalities like LLMs, vector stores, embeddings, and readers.

- **Old Usage**:
    ```python
    from llama_index.llms import OpenAI
    ```
- **New Usage**:
    ```python
    from llama_index.llms.openai import OpenAI
    ```

- **New Installations Required**:
    Install LLM and vector store dependencies:
    ```bash
    pip install llama-index-llms-openai llama-index-vector-stores-pinecone
    ```

### 2. Deprecation of `ServiceContext`
The `ServiceContext` object has been deprecated. Instead, you now directly pass LLMs, embedding models, and callbacks to each module.

- **Old Usage**:
    ```python
    service_context = ServiceContext(llm=OpenAI(), embed_model=OpenAIEmbedding())
    ```
- **New Usage**:
    ```python
    from llama_index.core.settings import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.llm = OpenAI()
    Settings.embed_model = OpenAIEmbedding()
    ```

### 3. New Node Parsers and Text Splitters
New node parsers and text splitters allow for more specialized document processing before indexing. For example, the `JSONNodeParser` and `MarkdownNodeParser` provide more flexibility when working with structured or semi-structured data formats.

### 4. Global Settings Object
You can now define LLMs, embeddings, and callbacks globally using the `Settings` object, eliminating the need to pass them through every function call.

- **Usage Example**:
    ```python
    from llama_index.core.settings import Settings
    from llama_index.llms.openai import OpenAI

    # Define the global LLM once
    Settings.llm = OpenAI()
    ```

---

## Core Functions Table

| **Function Category**       | **Function/Method Name**     | **Description**                                                                 | **Example Use Cases**                                                                                                      | **Import Statement**                                                                                       | **Example Code**                                                                                          |
|-----------------------------|------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Document Loading**         | `SimpleDirectoryReader`       | Reads documents from a specified directory and loads them for processing.        | Loading multiple documents from a folder for indexing and retrieval.                                                      | `from llama_index.core import SimpleDirectoryReader`                                                       | ``` # Load all documents from the 'data' folder documents = SimpleDirectoryReader("data").load_data()  # documents will contain text and metadata ``` |
| **LLM Interaction**          | `OpenAI`                     | Creates an OpenAI instance for LLM-based interactions like summarization or querying. | Using LLMs like GPT-4 to generate summaries or reorganize documents.                                                       | `from llama_index.llms.openai import OpenAI`                                                               | ``` # Initialize the OpenAI LLM model for summarization llm = OpenAI(model="gpt-4")  # Generate summaries for each document summaries = [llm.summarize(doc.text) for doc in documents] ``` |
| **Document Chunking**        | `SimpleNodeParser`            | Splits large documents into smaller chunks (nodes) for efficient retrieval.      | Chunking long documents for better search results.                                                                         | `from llama_index.core.node_parser import SimpleNodeParser`                                                | ``` # Initialize node parser with custom chunk size and overlap node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=20)  # Create nodes (chunks) from the loaded documents nodes = node_parser.get_nodes_from_documents(documents) ``` |
| **Metadata Handling**        | Manual metadata handling      | Attaches metadata like page numbers, section titles, etc. to nodes.              | Adding custom metadata to chunks for richer search results.                                                                | (custom logic)                                                                                             | ``` # Attach custom metadata like page numbers and sections to nodes for node in nodes:     node.metadata["page_number"] = get_page_number(node)     node.metadata["section"] = extract_section_title(node) ``` |
| **Indexing**                 | `VectorStoreIndex`            | Creates a vector-based index for efficient document retrieval.                   | Storing document embeddings for fast similarity searches.                                                                  | `from llama_index.core import VectorStoreIndex`                                                            | ``` # Create a vector index from the nodes index = VectorStoreIndex.from_nodes(nodes, embed_model=embed_model)  # You can use this index to perform searches later ``` |
|                             | `from_documents`              | Builds an index directly from documents.                                         | Automatically processing documents into an index for later querying.                                                       | `from llama_index.core import VectorStoreIndex`                                                            | ``` # Build a vector index directly from documents index = VectorStoreIndex.from_documents(documents) ``` |
|                             | `from_nodes`                  | Builds an index from preprocessed nodes.                                         | Indexing custom-chunked nodes.                                                                                             | `from llama_index.core import VectorStoreIndex`                                                            | ``` # Create index from previously processed nodes index = VectorStoreIndex.from_nodes(nodes) ``` |
| **Embeddings**               | `OpenAIEmbedding`             | Embeds document chunks using OpenAI’s API to convert text into vector embeddings. | Embedding document chunks for vector-based search.                                                                         | `from llama_index.embeddings.openai import OpenAIEmbedding`                                                | ``` # Initialize the OpenAI embedding model embed_model = OpenAIEmbedding()  # Generate embeddings for all document chunks vectors = embed_model.embed_text([node.text for node in nodes]) ``` |
| **Querying**                 | `as_query_engine`             | Creates a query engine from the vector index to handle search queries.           | Searching and retrieving specific information from indexed documents.                                                     | (built into `VectorStoreIndex`)                                                                            | ``` # Create a query engine from the index query_engine = index.as_query_engine()  # Query the engine for specific information response = query_engine.query("What is the main point of this document?") ``` |
|                             | `query`                       | Executes a query on the vector index and returns relevant results.               | Running a query to retrieve relevant text from an indexed dataset.                                                         | (method of `VectorStoreIndex.query_engine`)                                                               | ``` # Run a search query to retrieve relevant text response = query_engine.query("Find all financial data") ``` |
| **Custom Preprocessing**     | Custom LLM Preprocessing      | Uses LLM models to preprocess documents (e.g., summarizing, extracting).         | Summarizing or extracting key data from documents using LLMs before chunking.                                              | (custom logic using OpenAI)                                                                                | ``` # Use LLM to preprocess document text before chunking for doc in documents:     processed_text = llm.summarize(doc.text) ``` |
| **Post-Processing Queries**  | Rerank query results          | Refines or reranks query results using LLMs or custom logic.                     | After querying, rerank results based on importance or relevance.                                                           | (custom logic using OpenAI)                                                                                | ``` # Use LLM to rerank query results based on relevance ranked_response = llm.rerank(response) ``` |
| **Advanced Document Handling**| Custom Parsing Instructions  | Allows you to give custom natural language parsing instructions via LLM.         | Custom extraction of sections, summaries, or key data based on user instructions (e.g., "extract only financial sections"). | (custom logic using OpenAI)                                                                                | ``` # Provide natural language instructions for parsing structured_data = llm.extract(parsing_instruction="Extract financial data only", doc.text) ``` |

---

## Migration Guide from v0.9 to v0.10

1. **Update Your Imports**: Use the new modular imports for LLMs, vector stores, and embeddings.
   
   ```bash
   pip install llama-index
   ```

2. **Remove `ServiceContext`**: Refactor your code to pass LLMs, embeddings, and callbacks directly into the modules that require them or define them globally using `Settings`.

3. **Install Any Required Third-Party Packages**: If you’re using integrations such as Pinecone, HuggingFace, or Notion, you will need to install the corresponding `llama-index` package separately.

4. **Check for Updated Node Parsers**: If you're working with non-standard document formats (e.g., Markdown or JSON), explore the new `NodeParser` options to handle more complex document structures.

Additional notes:  Please not that open ai has some new models.  As of today's date Septemper 19,2024  my favorite is "gpt-4o-mini", so please use this model.  I never want to use "gpt 40" again.  So wehn you have a choice use the "gpt-4o-moini" model.