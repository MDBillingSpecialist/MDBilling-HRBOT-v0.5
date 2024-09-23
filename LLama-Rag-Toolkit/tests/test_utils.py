import pytest
from unittest.mock import patch, Mock
from utils import query_llm, generate_doc_id, get_current_time

@patch('utils.client.chat.completions.create')
def test_query_llm(mock_create):
    mock_create.return_value = Mock(choices=[Mock(message=Mock(content="Test response"))])
    
    result = query_llm("Test query")
    
    assert result == "Test response"
    mock_create.assert_called_once()

def test_generate_doc_id():
    doc_id = generate_doc_id(b"Test content")
    assert isinstance(doc_id, str)
    assert len(doc_id) == 32  # MD5 hash length

def test_get_current_time():
    current_time = get_current_time()
    assert isinstance(current_time, str)
    assert 'T' in current_time  # ISO format contains 'T' between date and time