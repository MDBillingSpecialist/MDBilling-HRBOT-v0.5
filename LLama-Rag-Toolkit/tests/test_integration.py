import pytest
from unittest.mock import patch, Mock, MagicMock
import streamlit as st
from document_processing import process_documents
from knowledge_base import create_knowledge_base
from question_generation import generate_synthetic_data
from evaluation import evaluate_rag_system
from llama_index.core import Document
from collections import deque
import pandas as pd

@pytest.fixture
def mock_session_state():
    return {
        'documents': {},
        'metrics': {'uploaded_docs': 0, 'parsed_docs': 0, 'total_chunks': 0},
        'parsed_documents': [],
        'knowledge_base': None,
        'knowledge_base_nodes': [],
        'synthetic_qa_pairs': []
    }

@patch('document_processing.SimpleDirectoryReader')
@patch('document_processing.SimpleNodeParser')
@patch('knowledge_base.VectorStoreIndex')
@patch('question_generation.RagDatasetGenerator')
@patch('evaluation.FaithfulnessEvaluator')
@patch('evaluation.RelevancyEvaluator')
@patch('evaluation.download_llama_pack')
@patch('question_generation.generate_dataset')
def test_full_pipeline_success(mock_generate_dataset, mock_download_llama_pack, mock_relevancy_evaluator, mock_faithfulness_evaluator,
                       mock_rag_dataset_generator, mock_vector_store_index,
                       mock_simple_node_parser, mock_simple_directory_reader,
                       mock_session_state):
    with patch.object(st, 'session_state', mock_session_state):
        # Mock document processing
        mock_simple_directory_reader.return_value.load_data.return_value = [Document(text="Test document")]
        mock_node = Mock(node_id="test_node")
        mock_simple_node_parser.from_defaults.return_value.get_nodes_from_documents.return_value = [mock_node]

        # Process documents
        mock_file = Mock(spec=["name", "read", "size"])
        mock_file.name = "test.txt"
        mock_file.read.return_value = b"Test content"
        mock_file.size = 1024
        mock_files = [mock_file]
        processed_docs = process_documents(mock_files)
        assert processed_docs is not None
        assert len(processed_docs) == 1
        assert st.session_state['metrics']['uploaded_docs'] == 1

        # Ensure knowledge_base_nodes are set in session_state
        st.session_state['knowledge_base_nodes'] = [mock_node]

        # Create knowledge base
        mock_index = Mock()
        mock_vector_store_index.from_documents.return_value = mock_index
        kb_created = create_knowledge_base()
        assert kb_created is True

        # Add print statements for debugging
        print(f"Session state after create_knowledge_base: {st.session_state}")
        print(f"kb_created: {kb_created}")

        # Generate synthetic data
        mock_dataset = Mock()
        mock_dataset.examples = [Mock() for _ in range(5)]
        mock_generate_dataset.return_value = mock_dataset
        
        with patch('question_generation.process_examples') as mock_process_examples:
            mock_process_examples.return_value = [{'question': f'Q{i}', 'answer': f'A{i}'} for i in range(1, 6)]
            qa_pairs_generated = generate_synthetic_data(5)
        
        print(f"qa_pairs_generated: {qa_pairs_generated}")
        assert qa_pairs_generated == 5
        assert len(st.session_state['synthetic_qa_pairs']) == 5

        # Evaluate RAG system
        mock_faithfulness_evaluator.return_value.evaluate.return_value = Mock(score=0.8)
        mock_relevancy_evaluator.return_value.evaluate.return_value = Mock(score=0.9)

        mock_rag_dataset = MagicMock()
        mock_rag_dataset.__len__.return_value = 5
        mock_rag_dataset.eval_queue = deque([0, 1, 2, 3, 4])

        mock_query_engine = Mock()

        mock_rag_evaluator_pack = Mock()
        mock_download_llama_pack.return_value = mock_rag_evaluator_pack
        mock_rag_evaluator_pack.return_value.run.return_value = pd.DataFrame({'metric': ['faithfulness', 'relevancy'], 'score': [0.8, 0.9]})

        evaluation_results = evaluate_rag_system(mock_rag_dataset, mock_query_engine)
        assert isinstance(evaluation_results, pd.DataFrame)
        assert len(evaluation_results) == 2  # faithfulness and relevancy

def test_full_pipeline_empty_documents(mock_session_state):
    with patch.object(st, 'session_state', mock_session_state):
        mock_files = []
        processed_docs = process_documents(mock_files)
        assert processed_docs is None
        assert st.session_state['metrics']['uploaded_docs'] == 0

@patch('knowledge_base.VectorStoreIndex')
def test_full_pipeline_knowledge_base_creation_failure(mock_vector_store_index, mock_session_state):
    with patch.object(st, 'session_state', mock_session_state):
        mock_vector_store_index.from_documents.side_effect = Exception("KB creation failed")

        print("Before create_knowledge_base() call")
        
        # Add this line to check the mock setup
        print(f"Mock side_effect: {mock_vector_store_index.from_documents.side_effect}")

        result = create_knowledge_base()
        
        print(f"create_knowledge_base() returned: {result}")
        
        # Check if the mock was called
        print(f"Mock called: {mock_vector_store_index.from_documents.called}")
        
        # If the mock was called, print the arguments
        if mock_vector_store_index.from_documents.called:
            print(f"Mock call args: {mock_vector_store_index.from_documents.call_args}")

        assert result is False, "create_knowledge_base() should return False when an exception is raised"

    # Remove the pytest.raises assertion since we're now checking the return value