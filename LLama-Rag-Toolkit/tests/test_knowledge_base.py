import pytest
from unittest.mock import patch, Mock
import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from knowledge_base import create_knowledge_base, create_transformations, export_knowledge_base, load_knowledge_base

@pytest.fixture
def mock_session_state():
    return {
        'knowledge_base_nodes': [Document(text="Test document")],
        'knowledge_base': None
    }

@patch('knowledge_base.VectorStoreIndex')
@patch('knowledge_base.create_transformations')
def test_create_knowledge_base(mock_create_transformations, mock_vector_store_index, mock_session_state):
    with patch.object(st, 'session_state', mock_session_state):
        mock_create_transformations.return_value = []
        mock_vector_store_index.from_documents.return_value = Mock()
        
        result = create_knowledge_base()
        
        assert result is True
        assert st.session_state['knowledge_base'] is not None

def test_create_transformations():
    transformations = create_transformations()
    assert len(transformations) > 0

@patch('knowledge_base.os.path.exists')
@patch('knowledge_base.os.makedirs')
@patch('builtins.open')
def test_export_knowledge_base(mock_open, mock_makedirs, mock_exists, mock_session_state):
    with patch.object(st, 'session_state', mock_session_state):
        mock_exists.return_value = False  # Simulate that the directory doesn't exist
        mock_session_state['knowledge_base'] = Mock()
        mock_session_state['knowledge_base'].storage_context.persist.return_value = None
        
        result = export_knowledge_base()
        
        assert result is True  # or False, depending on what you expect

@patch('knowledge_base.VectorStoreIndex')
@patch('knowledge_base.StorageContext')
def test_load_knowledge_base(mock_storage_context, mock_vector_store_index):
    mock_vector_store_index.from_vector_store.return_value = Mock()
    
    result = load_knowledge_base('test_path')
    
    assert result is not None