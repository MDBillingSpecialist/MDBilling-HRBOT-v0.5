import pytest
from unittest.mock import patch, Mock, MagicMock
import streamlit as st
from evaluation import evaluate_rag_system, evaluate_responses, save_evaluation_results
import pandas as pd

@pytest.fixture
def mock_session_state():
    return {
        'knowledge_base': Mock(),
        'synthetic_qa_pairs': [{'question': 'Test', 'answer': 'Test', 'reference': 'Test'}]
    }

@pytest.fixture
def mock_rag_dataset():
    dataset = MagicMock()
    dataset.__len__.return_value = 5
    dataset.examples = [
        MagicMock(query="Q1", reference_answer="A1"),
        MagicMock(query="Q2", reference_answer="A2"),
    ]
    return dataset

@pytest.fixture
def mock_query_engine():
    engine = Mock()
    engine.query.return_value = Mock(
        response="Test response",
        source_nodes=[Mock(node=Mock(get_content=lambda: "Test context"))]
    )
    return engine

def test_evaluate_rag_system(mock_rag_dataset, mock_query_engine):
    with patch('evaluation.download_llama_pack') as mock_download:
        mock_evaluator = Mock()
        mock_evaluator.return_value.run.return_value = pd.DataFrame({
            'metric': ['faithfulness', 'relevancy'],
            'score': [0.8, 0.9]
        })
        mock_download.return_value = mock_evaluator

        result = evaluate_rag_system(mock_rag_dataset, mock_query_engine)
        
        assert isinstance(result, pd.DataFrame)
        assert 'metric' in result.columns
        assert 'score' in result.columns

def test_evaluate_responses(mock_query_engine, mock_rag_dataset):
    with patch('evaluation.OpenAI'), \
         patch('evaluation.RelevancyEvaluator') as MockRelevancy, \
         patch('evaluation.FaithfulnessEvaluator') as MockFaithfulness:
        
        MockRelevancy.return_value.evaluate.return_value = Mock(score=0.9)
        MockFaithfulness.return_value.evaluate.return_value = Mock(score=0.8)

        all_evaluations, correct_relevancy, faithful_responses, total = evaluate_responses(
            mock_query_engine, mock_rag_dataset, 2, 0.7
        )

        assert len(all_evaluations) == 2
        assert correct_relevancy == 2
        assert faithful_responses == 2
        assert total == 2

def test_save_evaluation_results():
    with patch('evaluation.os.makedirs'), patch('evaluation.pd.DataFrame.to_html'):
        summary_df, all_evaluations_df, reports_dir = save_evaluation_results(
            [{'Query': 'Q1', 'Generated Response': 'A1', 'Expected Response': 'E1', 
              'Relevancy Score': 0.9, 'Faithfulness Score': 0.8, 'Is Hallucination': False}],
            1, 1, 1, 0.7, 'test_experiment'
        )

        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(all_evaluations_df, pd.DataFrame)
        assert 'test_experiment' in reports_dir

def test_evaluate_rag_system_no_dataset():
    mock_rag_dataset = None
    mock_query_engine = Mock()
    result = evaluate_rag_system(mock_rag_dataset, mock_query_engine)
    assert result == {"error": "Empty dataset"}

# Remove or comment out the test for generate_evaluation_report
# @patch('evaluation.pd.DataFrame')
# @patch('evaluation.os.path.exists')
# @patch('evaluation.os.makedirs')
# def test_generate_evaluation_report(mock_makedirs, mock_exists, mock_dataframe):
#     mock_exists.return_value = False
#     mock_dataframe.return_value.to_html.return_value = "<html></html>"
    
#     evaluation_results = {'faithfulness': [0.8], 'relevancy': [0.9]}
#     result = generate_evaluation_report(evaluation_results)
    
#     assert result is not None
#     mock_makedirs.assert_called_once()
#     mock_dataframe.assert_called_once()

# Remove the test_evaluate_rag_system_no_knowledge_base function as it's no longer relevant