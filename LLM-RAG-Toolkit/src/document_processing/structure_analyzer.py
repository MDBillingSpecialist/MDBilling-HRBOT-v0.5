import re
import fitz
import logging
import json
import os
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from utils.config_manager import config

logger = logging.getLogger(__name__)

def load_openai_client() -> OpenAI:
    try:
        return OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        raise

def extract_text_and_images(pdf_path: str) -> Tuple[str, List[Dict]]:
    try:
        with fitz.open(pdf_path) as pdf_document:
            text = ""
            images = []
            for page_num in range(0, min(10, pdf_document.page_count)):
                page = pdf_document[page_num]
                text += page.get_text() + "\n"
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_info = {
                        "page": page_num,
                        "type": base_image["ext"],
                        "size": len(base_image["image"])
                    }
                    images.append(image_info)
        return text, images
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        return "", []
    except Exception as e:
        logger.error(f"Error extracting content from PDF {pdf_path}: {str(e)}")
        return "", []

def detect_document_structure(text: str, images: List[Dict]) -> Dict[str, bool]:
    has_toc = bool(re.search(r'^\s*(?:Table of Contents|Contents|TOC)\s*$', text, re.MULTILINE | re.IGNORECASE))
    has_headers = bool(re.findall(r'^\s*(?:\d+\.)*\d+\s+[\w\s]+', text, re.MULTILINE))
    has_images = len(images) > 0
    
    return {
        "has_toc": has_toc,
        "has_headers": has_headers,
        "has_images": has_images
    }

def clean_json_string(json_string):
    json_string = json_string.replace("'", '"')
    json_string = json_string.strip()
    if not json_string.startswith('{'): json_string = '{' + json_string
    if not json_string.endswith('}'): json_string = json_string + '}'
    return json_string

import hashlib
import pickle
import os

def cache_result(func):
    def wrapper(*args, **kwargs):
        key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()
        cache_file = f"cache/{key}.pkl"
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        result = func(*args, **kwargs)
        
        os.makedirs("cache", exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    return wrapper

@cache_result
def extract_structure_llm(client: OpenAI, content: Dict[str, Any]) -> Dict[str, Any]:
    text_sample = content.get("text", "")[:4000]
    image_info = json.dumps(content.get("images", [])[:10])
    
    if not text_sample and not image_info:
        return {
            "error": "No content provided for analysis",
            "document_type": "unknown",
            "structure": [],
            "toc": {},
            "notable_features": []
        }
    
    prompt = (
        "Analyze this document excerpt and describe its structure. "
        "Identify main sections, subsections, and any notable features like images or graphs. "
        "If a Table of Contents is present, extract it. "
        "If no clear structure is found, suggest logical divisions based on content. "
        "Format the result as a JSON object with the following structure:\n"
        "{\n"
        '  "document_type": "report/article/manual/etc",\n'
        '  "structure": ["list", "of", "main", "sections"],\n'
        '  "toc": {"section1": {"subsection1": {}, "subsection2": {}}, "section2": {}},\n'
        '  "notable_features": ["list", "of", "features"]\n'
        "}\n\n"
        f"Document text sample:\n\n{text_sample}\n\n"
        f"Image information:\n{image_info}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in document analysis and structuring."},
                {"role": "user", "content": prompt}
            ],
            temperature=config.generation_parameters['temperature'],
            max_tokens=config.generation_parameters['max_tokens']
        )
        response_content = response.choices[0].message.content
        logger.info(f"Raw LLM response: {response_content}")
        print(f"Raw LLM response: {response_content}")
        
        try:
            structure_json = json.loads(response_content)
            logger.info("LLM structure extraction completed successfully.")
            return structure_json
        except json.JSONDecodeError as json_error:
            logger.error(f"Error parsing JSON from LLM response: {json_error}")
            return {
                "error": "Failed to parse LLM response as JSON",
                "raw_response": response_content,
                "document_type": "unknown",
                "structure": [],
                "toc": {},
                "notable_features": []
            }
    except Exception as e:
        logger.error(f"Error during LLM structure extraction: {e}", exc_info=True)
        return {
            "error": str(e),
            "document_type": "unknown",
            "structure": [],
            "toc": {},
            "notable_features": []
        }

def extract_info_from_text(text):
    doc_type_match = re.search(r'"document_type":\s*"([^"]+)"', text)
    structure_match = re.search(r'"structure":\s*(\[[^\]]+\])', text)
    toc_match = re.search(r'"toc":\s*(\{[^}]+\})', text)
    features_match = re.search(r'"notable_features":\s*(\[[^\]]+\])', text)

    return {
        "document_type": doc_type_match.group(1) if doc_type_match else None,
        "structure": json.loads(structure_match.group(1)) if structure_match else [],
        "toc": json.loads(toc_match.group(1)) if toc_match else {},
        "notable_features": json.loads(features_match.group(1)) if features_match else []
    }

def extract_toc(pdf_path: str) -> Dict[str, Any]:
    client = load_openai_client()
    if not client:
        return {"error": "Failed to initialize OpenAI client"}

    try:
        text, images = extract_text_and_images(pdf_path)
    except FileNotFoundError:
        return {"error": f"PDF file not found: {pdf_path}"}
    except Exception as e:
        return {"error": f"Error extracting content from PDF: {str(e)}"}

    if not text and not images:
        return {"error": "No content extracted from the PDF"}

    document_structure = detect_document_structure(text, images)
    
    content = {
        "text": text[:4000],  # Limit text to 4000 characters
        "images": images[:10]  # Limit to first 10 images
    }
    
    structure_data = extract_structure_llm(client, content)

    if isinstance(structure_data, dict) and "error" not in structure_data:
        structure_data["detected_structure"] = document_structure
        return structure_data
    else:
        logger.error("Structure extraction failed or incomplete.")
        return {
            "error": "Structure extraction failed or incomplete",
            "detected_structure": document_structure,
            "llm_output": structure_data
        }

def save_toc_to_json(toc_data: Dict[str, Any], output_file: str) -> None:
    try:
        with open(output_file, 'w') as f:
            json.dump(toc_data, f, indent=4)
        logger.info(f"Document structure saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving TOC to JSON: {e}", exc_info=True)

def main(pdf_path: str, output_file: str) -> None:
    try:
        toc_data = extract_toc(pdf_path)
        if toc_data:
            save_toc_to_json(toc_data, output_file)
        else:
            logger.error("Failed to extract TOC and analyze document structure.")
    except Exception as e:
        logger.error(f"Error in main TOC extraction process: {e}", exc_info=True)

if __name__ == "__main__":
    pdf_path = config.file_paths['pdf_path']
    output_file = config.file_paths['toc_json_path']
    main(pdf_path, output_file)
