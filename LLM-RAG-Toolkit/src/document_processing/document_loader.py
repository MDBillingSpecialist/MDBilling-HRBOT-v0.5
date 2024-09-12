import os
import logging
from typing import Dict, Any
import fitz  # PyMuPDF for PDF handling
from utils.config_manager import config

logger = logging.getLogger(__name__)

def quality_check(document: Dict[str, Any]) -> bool:
    """
    Perform quality checks on the document.
    """
    # Check if the document has content
    if not document.get('content'):
        logger.warning(f"Empty content for document: {document.get('source', 'Unknown')}")
        return False

    # Check if the content meets a minimum length requirement, if specified in config
    min_content_length = config.document_processing.get('min_content_length', 0)
    if len(document.get('content', '')) < min_content_length:
        logger.warning(f"Document content length ({len(document.get('content', ''))}) is less than the minimum required length ({min_content_length}): {document.get('source', 'Unknown')}")
        return False

    return True

class DocumentLoader:
    def __init__(self):
        self.supported_extensions = {
            '.pdf': self.load_pdf,
            '.txt': self.load_text,
            # Add more file types as needed
        }

    def load_document(self, file_path: str) -> Dict[str, Any]:
        file_path = os.path.abspath(file_path)
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")

        try:
            content = self.supported_extensions[file_extension](file_path)
            document = {
                "file_path": file_path,
                "file_type": file_extension,
                "content": content
            }
            if quality_check(document):
                return document
            else:
                logger.warning(f"Document failed quality check: {file_path}")
                return {
                    "file_path": file_path,
                    "file_type": file_extension,
                    "content": "",
                    "error": "Failed quality check"
                }
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
            return {
                "file_path": file_path,
                "file_type": file_extension,
                "content": "",
                "error": str(e)
            }

    def load_pdf(self, file_path: str) -> str:
        try:
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}", exc_info=True)
            return ""

    def load_text(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}", exc_info=True)
            return ""
