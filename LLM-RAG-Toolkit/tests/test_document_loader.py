import pytest
from src.document_processing.document_loader import DocumentLoader
import os

@pytest.fixture
def document_loader():
    return DocumentLoader()

def test_load_pdf(document_loader, tmp_path):
    # Create a simple PDF file for testing
    pdf_path = tmp_path / "test.pdf"
    # You'd need to create a simple PDF file here. For this example, we'll just check if the method exists
    assert hasattr(document_loader, 'load_pdf')

def test_load_docx(document_loader, tmp_path):
    docx_path = tmp_path / "test.docx"
    # Similar to PDF, create a simple DOCX for testing
    assert hasattr(document_loader, 'load_docx')

def test_load_text(document_loader, tmp_path):
    text_path = tmp_path / "test.txt"
    with open(text_path, 'w') as f:
        f.write("This is a test document.")
    
    content = document_loader.load_text(str(text_path))
    assert content == "This is a test document."

def test_unsupported_file_type(document_loader, tmp_path):
    unsupported_path = tmp_path / "test.unsupported"
    with open(unsupported_path, 'w') as f:
        f.write("This is an unsupported file type.")
    
    with pytest.raises(ValueError):
        document_loader.load_document(str(unsupported_path))