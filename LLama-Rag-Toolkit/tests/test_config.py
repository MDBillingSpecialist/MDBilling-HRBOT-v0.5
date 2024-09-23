import pytest
from unittest.mock import patch, Mock
from config import load_config

@patch('config.load_dotenv')
@patch('config.os.environ.get')
@patch('config.OpenAI')
@patch('config.OpenAIEmbedding')
@patch('config.Settings')
def test_load_config(mock_settings, mock_openai_embedding, mock_openai, mock_environ_get, mock_load_dotenv):
    mock_environ_get.return_value = "test_api_key"
    
    client, logger = load_config()
    
    assert client is not None
    assert logger is not None
    mock_load_dotenv.assert_called_once()
    assert mock_environ_get.call_count == 5
    mock_environ_get.assert_any_call('OPENAI_API_KEY')
    mock_openai.assert_called_once()
    mock_openai_embedding.assert_called_once()
    assert mock_settings.llm is not None
    assert mock_settings.embed_model is not None

@patch('config.os.environ.get')
def test_load_config_missing_api_key(mock_environ_get):
    mock_environ_get.return_value = None
    
    with pytest.raises(ValueError):
        load_config()