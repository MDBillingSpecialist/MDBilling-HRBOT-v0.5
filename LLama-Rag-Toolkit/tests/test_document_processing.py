import pytest
from unittest.mock import patch, Mock, mock_open
import streamlit as st
from io import BytesIO
from llama_index.core import Document
from document_processing import process_documents, determine_optimal_chunk_params, display_parsed_documents

@pytest.fixture
def mock_uploaded_files():
    file1 = Mock()
    file1.name = "test1.txt"
    file1.read.return_value = b"This is a test document."
    
    file2 = Mock()
    file2.name = "test2.txt"
    file2.read.return_value = b"This is another test document, slightly longer."
    
    return [file1, file2]

@pytest.fixture
def mock_session_state():
    return {
        'documents': {},
        'metrics': {'uploaded_docs': 0, 'parsed_docs': 0, 'total_chunks': 0},
        'parsed_documents': []
    }

def test_determine_optimal_chunk_params(mock_uploaded_files):
    chunk_size, chunk_overlap = determine_optimal_chunk_params(mock_uploaded_files)
    assert isinstance(chunk_size, int)
    assert isinstance(chunk_overlap, int)
    assert chunk_size > 0
    assert chunk_overlap >= 0
    assert chunk_overlap < chunk_size

@patch('document_processing.SimpleDirectoryReader')
@patch('document_processing.SimpleNodeParser')
@patch('document_processing.generate_doc_id')
@patch('document_processing.get_current_time')
def test_process_documents(mock_get_current_time, mock_generate_doc_id, MockSimpleNodeParser, MockSimpleDirectoryReader, mock_uploaded_files, mock_session_state):
    with patch.object(st, 'session_state', mock_session_state):
        MockSimpleDirectoryReader.return_value.load_data.return_value = [Document(text="Test document")]
        MockSimpleNodeParser.from_defaults.return_value.get_nodes_from_documents.return_value = [Mock(node_id="test_node")]
        mock_generate_doc_id.side_effect = ["doc_id_1", "doc_id_2"]  # Return different IDs for each file
        mock_get_current_time.return_value = "2023-01-01T00:00:00"
        
        result = process_documents(mock_uploaded_files)
        
        assert result is not None
        assert len(result) > 0
        assert st.session_state['metrics']['uploaded_docs'] == len(mock_uploaded_files)
        assert st.session_state['metrics']['parsed_docs'] == len(mock_uploaded_files)
        assert st.session_state['metrics']['total_chunks'] == len(mock_uploaded_files)
        assert len(st.session_state['documents']) == len(mock_uploaded_files)

def test_process_documents_no_files():
    result = process_documents([])
    assert result is None

@patch('streamlit.subheader')
@patch('streamlit.write')
@patch('streamlit.json')
def test_display_parsed_documents(mock_json, mock_write, mock_subheader, mock_session_state):
    mock_node = Mock(text="Test node text", extra_info={'doc_id': 'test_doc'}, node_id='test_node', metadata={})
    mock_session_state['parsed_documents'] = [mock_node]
    
    with patch.object(st, 'session_state', mock_session_state):
        display_parsed_documents()
    
    mock_subheader.assert_called_once_with("Parsed Documents")
    mock_write.assert_called_once_with("Node 1:")
    mock_json.assert_called_once()

def test_display_parsed_documents_empty():
    with patch.object(st, 'session_state', {}):
        display_parsed_documents()
    # Assert that nothing is displayed when there are no parsed documents

@patch('document_processing.tempfile.NamedTemporaryFile')
@patch('document_processing.os.unlink')
def test_process_documents_file_handling(mock_unlink, mock_named_temp_file, mock_uploaded_files, mock_session_state):
    mock_temp_file = Mock()
    mock_temp_file.name = '/tmp/test_file.txt'
    mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file
    
    with patch.object(st, 'session_state', mock_session_state):
        with patch('document_processing.SimpleDirectoryReader') as MockSimpleDirectoryReader:
            with patch('document_processing.SimpleNodeParser') as MockSimpleNodeParser:
                MockSimpleDirectoryReader.return_value.load_data.return_value = [Document(text="Test document")]
                MockSimpleNodeParser.from_defaults.return_value.get_nodes_from_documents.return_value = [Mock(node_id="test_node")]
                
                process_documents(mock_uploaded_files)
    
    mock_named_temp_file.assert_called()
    mock_unlink.assert_called_with('/tmp/test_file.txt')

import unittest
from src.document_processing.processor import process_documents

class TestDocumentProcessing(unittest.TestCase):
    def test_process_documents(self):
        # ... (test cases)