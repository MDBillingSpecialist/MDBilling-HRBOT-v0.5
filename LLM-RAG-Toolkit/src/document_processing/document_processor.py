import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from typing import List, Dict, Any
from .content_segmenter import segment_pdf, post_process_segments
from .structure_analyzer import extract_toc
from .metadata_extractor import extract_metadata
from utils.config_manager import config
import os
import logging
import json

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.supported_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff'}

    def process_document(self, file_path: str) -> Dict[str, Any]:
        document_type = self.detect_document_type(file_path)
        if document_type == 'pdf':
            return self.process_pdf(file_path)
        elif document_type == 'image':
            return self.process_image(file_path)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")

    def detect_document_type(self, file_path: str) -> str:
        _, extension = os.path.splitext(file_path)
        if extension.lower() not in self.supported_types:
            raise ValueError(f"Unsupported file type: {extension}")
        return 'pdf' if extension.lower() == '.pdf' else 'image'

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        try:
            metadata = extract_metadata(file_path)
            toc_data = extract_toc(file_path)
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
            logger.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
            return {
                "file_path": file_path,
                "error": str(e)
            }

    def process_image(self, file_path: str) -> Dict[str, Any]:
        try:
            image = Image.open(file_path)
            metadata = {
                "type": "image",
                "format": image.format,
                "mode": image.mode,
                "size": image.size
            }
            
            # Only attempt OCR if Tesseract is installed
            try:
                text = pytesseract.image_to_string(image)
                segments = self.segment_image_text(text)
            except pytesseract.pytesseract.TesseractNotFoundError:
                logger.warning("Tesseract OCR not found. Unable to extract text from image.")
                segments = [{
                    "title": "Image Content",
                    "content": "Image text extraction not available (Tesseract OCR not installed)",
                    "tokens": 0
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
            logger.error(f"Error processing image {file_path}: {e}", exc_info=True)
            raise

    def segment_image_text(self, text: str) -> List[Dict[str, Any]]:
        lines = text.split('\n')
        segments = []
        current_segment = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(line.split())
            if current_tokens + line_tokens > config.document_processing['max_segment_tokens']:
                segments.append({
                    "title": f"Segment {len(segments) + 1}",
                    "content": " ".join(current_segment),
                    "tokens": current_tokens
                })
                current_segment = []
                current_tokens = 0
            
            current_segment.append(line)
            current_tokens += line_tokens
        
        if current_segment:
            segments.append({
                "title": f"Segment {len(segments) + 1}",
                "content": " ".join(current_segment),
                "tokens": current_tokens
            })
        
        return segments

    def perform_semantic_analysis(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "main_topics": self.extract_main_topics(segments),
            "key_entities": self.extract_key_entities(segments),
            "summary": self.generate_summary(segments)
        }

    def extract_main_topics(self, segments: List[Dict[str, Any]]) -> List[str]:
        return [segment['title'] for segment in segments if 'title' in segment][:5]

    def extract_key_entities(self, segments: List[Dict[str, Any]]) -> List[str]:
        all_words = " ".join([segment['content'] for segment in segments]).split()
        return list(set(word.capitalize() for word in all_words if len(word) > 5))[:10]

    def generate_summary(self, segments: List[Dict[str, Any]]) -> str:
        return " ".join([segment['content'][:100] for segment in segments[:3]])

    def log_processing_results(self, result: Dict[str, Any], doc_type: str):
        logger.info(f"\n--- {doc_type} Processing Results ---")
        logger.info(f"File: {os.path.basename(result['file_path'])}")
        
        if 'metadata' in result:
            logger.info("\nMetadata:")
            logger.info(json.dumps(result['metadata'], indent=2))
        
        if 'toc' in result:
            logger.info("\nTable of Contents:")
            logger.info(json.dumps(result['toc'], indent=2))
        
        if 'segments' in result:
            logger.info("\nSegments (first 3):")
            for segment in result['segments'][:3]:
                logger.info(f"- Title: {segment['title']}")
                logger.info(f"  Content: {segment['content'][:100]}...")
                logger.info(f"  Tokens: {segment['tokens']}")
            logger.info(f"\nTotal segments: {len(result['segments'])}")
        
        if 'semantic_data' in result:
            logger.info("\nSemantic Data:")
            logger.info(json.dumps(result['semantic_data'], indent=2))