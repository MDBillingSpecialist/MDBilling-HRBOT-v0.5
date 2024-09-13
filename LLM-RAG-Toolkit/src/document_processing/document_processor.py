import os
import logging
import json
from typing import List, Dict, Any
from PIL import Image
import pytesseract
import spacy
from nltk.tokenize import sent_tokenize
import fitz  # Add this import
from .content_segmenter import (
    segment_pdf,
    post_process_segments,
    get_token_count,
    extract_toc_from_pdf,
    extract_text_with_headings,
    extract_images
)
from utils.config_manager import config

logger = logging.getLogger(__name__)

# Initialize spaCy model for semantic analysis
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Download the model if not present
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


class DocumentProcessor:
    """A class for processing documents (PDFs and images) and extracting relevant information."""

    def __init__(self):
        self.supported_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff'}

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract information.

        Args:
            file_path (str): The path to the document.

        Returns:
            Dict[str, Any]: The processed document data.
        """
        document_type = self.detect_document_type(file_path)
        if document_type == 'pdf':
            return self.process_pdf(file_path)
        elif document_type == 'image':
            return self.process_image(file_path)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")

    def detect_document_type(self, file_path: str) -> str:
        """Detect the type of the document based on the file extension.

        Args:
            file_path (str): The path to the document.

        Returns:
            str: The document type ('pdf' or 'image').

        Raises:
            ValueError: If the file type is unsupported.
        """
        _, extension = os.path.splitext(file_path)
        if extension.lower() not in self.supported_types:
            raise ValueError(f"Unsupported file type: {extension}")
        return 'pdf' if extension.lower() == '.pdf' else 'image'

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process a PDF document.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            Dict[str, Any]: The processed PDF data.
        """
        try:
            metadata = self.extract_metadata(file_path)
            toc_data = extract_toc_from_pdf(file_path)
            raw_segments = segment_pdf(file_path, toc_data)
            processed_segments = post_process_segments(raw_segments)
            semantic_data = self.perform_semantic_analysis(processed_segments)

            result = {
                "file_path": file_path,
                "metadata": metadata,
                "toc": toc_data,
                "segments": processed_segments,
                "semantic_data": semantic_data
            }

            self.log_processing_results(result, "PDF")

            return result
        except Exception as e:
            logger.error(f"Error processing PDF {os.path.basename(file_path)}: {e}", exc_info=True)
            return {
                "file_path": file_path,
                "error": str(e)
            }

    def process_image(self, file_path: str) -> Dict[str, Any]:
        """Process an image document.

        Args:
            file_path (str): The path to the image file.

        Returns:
            Dict[str, Any]: The processed image data.
        """
        try:
            image = Image.open(file_path)
            metadata = {
                "type": "image",
                "format": image.format,
                "mode": image.mode,
                "size": image.size
            }

            # Attempt OCR using Tesseract
            try:
                text = pytesseract.image_to_string(image)
                segments = self.segment_image_text(text)
            except pytesseract.pytesseract.TesseractNotFoundError:
                logger.warning("Tesseract OCR not found. Unable to extract text from image.")
                segments = [{
                    "title": "Image Content",
                    "content": "Image text extraction not available (Tesseract OCR not installed)",
                    "tokens": 0,
                    "level": 1,
                    "path": ["Image Content"]
                }]

            semantic_data = self.perform_semantic_analysis(segments)

            result = {
                "file_path": file_path,
                "metadata": metadata,
                "segments": segments,
                "semantic_data": semantic_data
            }

            self.log_processing_results(result, "Image")

            return result
        except Exception as e:
            logger.error(f"Error processing image {os.path.basename(file_path)}: {e}", exc_info=True)
            return {
                "file_path": file_path,
                "error": str(e)
            }

    def segment_image_text(self, text: str) -> List[Dict[str, Any]]:
        """Segment the extracted text from an image.

        Args:
            text (str): The OCR-extracted text.

        Returns:
            List[Dict[str, Any]]: A list of text segments.
        """
        sentences = sent_tokenize(text)
        segments = []
        max_tokens = config.document_processing['max_segment_tokens']
        overlap = config.document_processing.get('overlap_sentences', 2)
        current_segment = []
        current_tokens = 0
        segment_index = 1
        i = 0

        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = get_token_count(sentence)
            if current_tokens + sentence_tokens > max_tokens and current_segment:
                segment_content = " ".join(current_segment)
                tokens = get_token_count(segment_content)
                segments.append({
                    "title": f"Segment {segment_index}",
                    "content": segment_content,
                    "tokens": tokens,
                    "level": 1,
                    "path": [f"Segment {segment_index}"]
                })
                segment_index += 1
                current_segment = current_segment[-overlap:]  # Keep last 'overlap' sentences
                current_tokens = get_token_count(" ".join(current_segment))
            else:
                current_segment.append(sentence)
                current_tokens += sentence_tokens
                i += 1

        if current_segment:
            segment_content = " ".join(current_segment)
            tokens = get_token_count(segment_content)
            segments.append({
                "title": f"Segment {segment_index}",
                "content": segment_content,
                "tokens": tokens,
                "level": 1,
                "path": [f"Segment {segment_index}"]
            })

        return segments

    def perform_semantic_analysis(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform semantic analysis on the segments.

        Args:
            segments (List[Dict[str, Any]]): A list of content segments.

        Returns:
            Dict[str, Any]: Semantic analysis results.
        """
        return {
            "main_topics": self.extract_main_topics(segments),
            "key_entities": self.extract_key_entities(segments),
            "summary": self.generate_summary(segments)
        }

    def extract_main_topics(self, segments: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from the segments.

        Args:
            segments (List[Dict[str, Any]]): A list of content segments.

        Returns:
            List[str]: A list of main topic titles.
        """
        titles = [segment['title'] for segment in segments if 'title' in segment]
        # Remove duplicates while preserving order
        seen = set()
        main_topics = []
        for title in titles:
            if title not in seen:
                seen.add(title)
                main_topics.append(title)
            if len(main_topics) >= 5:
                break
        return main_topics

    def extract_key_entities(self, segments: List[Dict[str, Any]]) -> List[str]:
        """Extract key entities from the segments using spaCy.

        Args:
            segments (List[Dict[str, Any]]): A list of content segments.

        Returns:
            List[str]: A list of key entities.
        """
        text = " ".join([segment['content'] for segment in segments])
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        # Remove duplicates while preserving order
        seen = set()
        key_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                key_entities.append(entity)
            if len(key_entities) >= 10:
                break
        return key_entities

    def generate_summary(self, segments: List[Dict[str, Any]]) -> str:
        """Generate a summary of the document.

        Args:
            segments (List[Dict[str, Any]]): A list of content segments.

        Returns:
            str: A summary string.
        """
        text = " ".join([segment['content'] for segment in segments])
        doc = nlp(text)
        # Simple summarization by selecting key sentences
        sentences = [sent.text for sent in doc.sents]
        # Use spaCy's sentence ranking (if available) or select the first few sentences
        summary_sentences = sentences[:5]
        return " ".join(summary_sentences)

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF document.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            Dict[str, Any]: The metadata dictionary.
        """
        metadata = {}
        try:
            with fitz.open(file_path) as doc:
                metadata = doc.metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF {os.path.basename(file_path)}: {e}", exc_info=True)
        return metadata

    def log_processing_results(self, result: Dict[str, Any], doc_type: str):
        """Log the processing results.

        Args:
            result (Dict[str, Any]): The processed document data.
            doc_type (str): The type of document ('PDF' or 'Image').
        """
        logger.info(f"\n--- {doc_type} Processing Results ---")
        logger.info(f"File: {os.path.basename(result['file_path'])}")

        if 'metadata' in result:
            logger.info("\nMetadata:")
            logger.info(json.dumps(result['metadata'], indent=2, ensure_ascii=False))

        if 'toc' in result and result['toc']:
            logger.info("\nTable of Contents:")
            logger.info(json.dumps(result['toc'], indent=2, ensure_ascii=False))

        if 'segments' in result:
            logger.info("\nSegments (first 3):")
            for segment in result['segments'][:3]:
                logger.info(f"- Title: {segment['title']}")
                logger.info(f"  Content: {segment['content'][:100]}...")
                logger.info(f"  Tokens: {segment['tokens']}")
            logger.info(f"\nTotal segments: {len(result['segments'])}")

        if 'semantic_data' in result:
            logger.info("\nSemantic Data:")
            logger.info(json.dumps(result['semantic_data'], indent=2, ensure_ascii=False))
