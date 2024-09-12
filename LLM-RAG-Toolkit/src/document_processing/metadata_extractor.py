import fitz
import logging
from typing import Dict, Any, List
from datetime import datetime
from utils.config_manager import config
from openai import OpenAI
import json

logger = logging.getLogger(__name__)

def load_openai_client() -> OpenAI:
    """Initialize and return an OpenAI client."""
    try:
        return OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        raise

def extract_basic_metadata(pdf_path: str) -> Dict[str, Any]:
    """Extract basic metadata from the PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            metadata = {
                "title": doc.metadata.get("title", "Unknown"),
                "author": doc.metadata.get("author", "Unknown"),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": len(doc),
                "file_size": doc.filesize,
            }
        return metadata
    except Exception as e:
        logger.error(f"Error extracting basic metadata from {pdf_path}: {e}", exc_info=True)
        return {}

def extract_text_statistics(pdf_path: str) -> Dict[str, Any]:
    """Extract text statistics from the PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            
            word_count = len(text.split())
            char_count = len(text)
            
            return {
                "word_count": word_count,
                "character_count": char_count,
                "average_word_length": char_count / word_count if word_count > 0 else 0
            }
    except Exception as e:
        logger.error(f"Error extracting text statistics from {pdf_path}: {e}", exc_info=True)
        return {}

def extract_image_info(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract information about images in the PDF."""
    try:
        image_info = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_info.append({
                        "page_number": page_num + 1,
                        "image_index": img_index + 1,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "color_space": base_image["colorspace"],
                        "bits_per_component": base_image["bpc"],
                        "image_format": base_image["ext"],
                    })
        return image_info
    except Exception as e:
        logger.error(f"Error extracting image info from {pdf_path}: {e}", exc_info=True)
        return []

def extract_semantic_metadata(client: OpenAI, text: str) -> Dict[str, Any]:
    """Extract semantic metadata using LLM."""
    prompt = (
        "Analyze the following text and extract key semantic metadata. "
        "Provide a brief summary, main topics, and any notable entities mentioned. "
        "Format the result as a JSON object with the following structure:\n"
        "{\n"
        "  'summary': 'Brief summary of the content',\n"
        "  'main_topics': ['list', 'of', 'main', 'topics'],\n"
        "  'entities': ['list', 'of', 'notable', 'entities'],\n"
        "  'document_type': 'Inferred document type (e.g., research paper, manual, report)'\n"
        "}\n\n"
        f"Text to analyze:\n\n{text[:4000]}"  # Limit text to avoid token limits
    )

    try:
        response = client.chat.completions.create(
            model=config.models['generation_model'],
            messages=[
                {"role": "system", "content": "You are an expert in document analysis and metadata extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        semantic_metadata = response.choices[0].message.content
        return json.loads(semantic_metadata)
    except Exception as e:
        logger.error(f"Error extracting semantic metadata: {e}", exc_info=True)
        return {}

def extract_metadata(pdf_path: str) -> Dict[str, Any]:
    """Extract all metadata from the PDF file."""
    try:
        basic_metadata = extract_basic_metadata(pdf_path)
        text_stats = extract_text_statistics(pdf_path)
        image_info = extract_image_info(pdf_path)
        
        # Extract text for semantic analysis
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        
        client = load_openai_client()
        semantic_metadata = extract_semantic_metadata(client, text)
        
        metadata = {
            **basic_metadata,
            **text_stats,
            "image_info": image_info,
            **semantic_metadata,
            "extraction_date": datetime.now().isoformat()
        }
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {pdf_path}: {e}", exc_info=True)
        return {}

def save_metadata_to_json(metadata: Dict[str, Any], output_file: str) -> None:
    """Save the extracted metadata to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving metadata to JSON: {e}", exc_info=True)

def main(pdf_path: str, output_file: str) -> None:
    """Main function to extract and save metadata."""
    try:
        metadata = extract_metadata(pdf_path)
        save_metadata_to_json(metadata, output_file)
    except Exception as e:
        logger.error(f"Error in main metadata extraction process: {e}", exc_info=True)

if __name__ == "__main__":
    pdf_path = config.file_paths['pdf_path']
    output_file = config.file_paths['metadata_output_path']
    main(pdf_path, output_file)
