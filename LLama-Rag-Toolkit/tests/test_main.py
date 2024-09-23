import pytest
from unittest.mock import patch, Mock
import streamlit as st
from main import main

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

@patch('streamlit.title')
@patch('streamlit.sidebar.title')
@patch('streamlit.sidebar.header')
@patch('streamlit.sidebar.slider')
@patch('streamlit.sidebar.file_uploader')
@patch('streamlit.sidebar.button')
@patch('main.process_documents')
@patch('main.create_knowledge_base')
@patch('main.export_knowledge_base')
@patch('streamlit.sidebar.text_input')
def test_main(mock_text_input, mock_export_kb, mock_create_kb, mock_process_docs,
              mock_button, mock_file_uploader, mock_slider,
              mock_sidebar_header, mock_sidebar_title, mock_title,
              mock_session_state):
    with patch.object(st, 'session_state', mock_session_state):
        mock_file_uploader.return_value = [Mock(name="test.txt")]
        mock_button.return_value = True
        
        # Create a more realistic mock for parsed documents
        mock_node = Mock()
        mock_node.text = "This is a test document"
        mock_node.extra_info = {'doc_id': 'test_doc_id'}
        mock_node.node_id = 'test_node_id'
        mock_node.metadata = {'key': 'value'}
        
        mock_process_docs.return_value = [mock_node]
        mock_session_state['parsed_documents'] = [mock_node]
        
        mock_create_kb.return_value = True
        mock_export_kb.return_value = "path/to/exported/kb"
        
        mock_text_input.return_value = "path/to/knowledge_base"
        
        main()

        # Add your assertions here

# Add more tests for different scenarios in the main app