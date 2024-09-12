import re
import json
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import fitz  # PyMuPDF
import pdfplumber
from utils.config_manager import config
import os

logger = logging.getLogger(__name__)

def load_toc(toc_json_path: str) -> Dict:
    """Load the table of contents structure from a JSON file."""
    try:
        with open(toc_json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading TOC from {toc_json_path}: {e}", exc_info=True)
        return {}

def create_toc_patterns(toc_structure: Dict) -> List[Tuple[str, re.Pattern]]:
    """Create regex patterns from the TOC structure."""
    patterns = []
    for section, subsections in toc_structure.items():
        patterns.append((section, re.compile(rf"^\s*{re.escape(section)}[\s:]*$", re.IGNORECASE | re.MULTILINE)))
        if isinstance(subsections, dict):
            for subsection in subsections:
                patterns.append((subsection, re.compile(rf"^\s*{re.escape(subsection)}[\s:]*$", re.IGNORECASE | re.MULTILINE)))
    return patterns

def extract_images(pdf_path: str) -> List[Dict]:
    """Extract images from the PDF."""
    images = []
    try:
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    images.append({
                        "page": page_num,
                        "type": base_image["ext"],
                        "size": len(base_image["image"]),
                        "content": base_image["image"]
                    })
    except Exception as e:
        logger.error(f"Error extracting images from PDF {pdf_path}: {e}", exc_info=True)
    return images

def segment_pdf(pdf_path: str, toc_data: Dict[str, Any] = None) -> List[Dict]:
    """Segment the PDF using TOC if available, otherwise use heuristic segmentation."""
    if toc_data:
        toc_patterns = create_toc_patterns(toc_data)
        return segment_pdf_using_toc(pdf_path, toc_patterns)
    else:
        return segment_pdf_heuristically(pdf_path)

def segment_pdf_using_toc(pdf_path: str, toc_patterns: List[Tuple[str, re.Pattern]]) -> List[Dict]:
    """Segment the PDF using the table of contents structure."""
    segments = []
    current_title = "Introduction"
    current_segment = []
    current_tokens = 0
    images = extract_images(pdf_path)
    image_index = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Processing pages")):
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split('\n')
                for line in lines:
                    matched = False
                    for title, pattern in toc_patterns:
                        if pattern.match(line):
                            if current_segment:
                                segments.append({
                                    "title": current_title,
                                    "content": " ".join(current_segment),
                                    "tokens": current_tokens
                                })
                            current_title = title
                            current_segment = []
                            current_tokens = 0
                            matched = True
                            break

                    if not matched:
                        current_segment.append(line)
                        current_tokens += len(line.split())

                    if current_tokens >= config.document_processing['max_segment_tokens']:
                        segments.append({
                            "title": current_title,
                            "content": " ".join(current_segment),
                            "tokens": current_tokens
                        })
                        current_segment = []
                        current_tokens = 0

                segments.extend(process_images(images, image_index, page_num, current_title))
                image_index = update_image_index(images, image_index, page_num)

    except Exception as e:
        logger.error(f"Error during PDF processing: {e}", exc_info=True)

    if current_segment:
        segments.append({
            "title": current_title,
            "content": " ".join(current_segment),
            "tokens": current_tokens
        })

    return segments

def segment_pdf_heuristically(pdf_path: str) -> List[Dict]:
    """Segment the PDF using heuristic methods, ensuring some information is extracted for any document."""
    segments = []
    all_content = []
    total_tokens = 0
    images = []
    metadata = {}

    try:
        images = extract_images(pdf_path)
    except Exception as e:
        logger.warning(f"Failed to extract images from {pdf_path}: {e}")

    try:
        with fitz.open(pdf_path) as pdf_document:
            metadata = pdf_document.metadata

            for page_num, page in enumerate(tqdm(pdf_document, desc="Processing pages")):
                try:
                    text = page.get_text()
                    if text:
                        all_content.append(text)
                        total_tokens += len(text.split())
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} in {pdf_path}: {e}")

                segments.append({
                    "title": f"Page {page_num + 1} Info",
                    "content": f"Page number: {page_num + 1}, Size: {page.rect.width}x{page.rect.height}",
                    "tokens": 10
                })

        segments.extend(process_content(all_content, total_tokens))
        segments.extend(process_images(images))
        segments.extend(process_metadata(metadata))
        segments.append(create_document_summary(pdf_path, len(pdf_document), total_tokens, len(images)))

    except Exception as e:
        logger.error(f"Error during PDF processing for {pdf_path}: {e}", exc_info=True)
        segments.append({
            "title": "Processing Error",
            "content": f"An error occurred while processing this PDF: {str(e)}",
            "tokens": 20
        })

    return segments

def process_content(all_content: List[str], total_tokens: int) -> List[Dict]:
    if all_content:
        return [{
            "title": "Document Content",
            "content": " ".join(all_content),
            "tokens": total_tokens
        }]
    else:
        return [{
            "title": "Content Notice",
            "content": "This document contains no extractable text content.",
            "tokens": 10
        }]

def process_images(images: List[Dict], start_index: int = 0, page_num: int = None, current_title: str = None) -> List[Dict]:
    processed_images = []
    for i, img in enumerate(images[start_index:]):
        if page_num is not None and img["page"] != page_num:
            break
        title = f"Image in {current_title}" if current_title else f"Image {i + 1}"
        processed_images.append({
            "title": title,
            "content": f"[Image: {img['type']} format, size: {img['width']}x{img['height']} pixels, on page: {img['page'] + 1}]",
            "tokens": 20,
            "image_data": {
                "page": img["page"],
                "type": img["type"],
                "width": img["width"],
                "height": img["height"]
            }
        })
    return processed_images

def update_image_index(images: List[Dict], current_index: int, current_page: int) -> int:
    while current_index < len(images) and images[current_index]["page"] <= current_page:
        current_index += 1
    return current_index

def process_metadata(metadata: Dict) -> List[Dict]:
    metadata_content = "\n".join([f"{k}: {v}" for k, v in metadata.items() if v])
    return [{
        "title": "Document Metadata",
        "content": metadata_content if metadata_content else "No metadata available",
        "tokens": len(metadata_content.split()) if metadata_content else 5
    }]

def create_document_summary(pdf_path: str, total_pages: int, total_tokens: int, total_images: int) -> Dict:
    return {
        "title": "Document Summary",
        "content": (f"File: {pdf_path}\n"
                    f"Total pages: {total_pages}\n"
                    f"Total extracted tokens: {total_tokens}\n"
                    f"Images: {total_images}\n"
                    f"File size: {os.path.getsize(pdf_path)} bytes"),
        "tokens": 30
    }

def post_process_segments(segments: List[Dict]) -> List[Dict]:
    """Post-process segments to ensure they meet the desired format and token limits."""
    processed_segments = []
    for segment in segments:
        if "image" in segment:
            processed_segments.append(segment)
            continue

        content = re.sub(r'\s+', ' ', segment['content']).strip()
        
        if segment['tokens'] > config.document_processing['max_segment_tokens']:
            processed_segments.extend(split_long_segment(segment, content))
        else:
            processed_segments.append({
                "title": segment['title'],
                "content": content,
                "tokens": segment['tokens']
            })
    
    return processed_segments if processed_segments else [{
        "title": "Empty Document",
        "content": "This document contains no extractable content.",
        "tokens": 8
    }]

def split_long_segment(segment: Dict, content: str) -> List[Dict]:
    words = content.split()
    current_chunk = []
    current_tokens = 0
    chunks = []
    for word in words:
        current_chunk.append(word)
        current_tokens += 1
        if current_tokens >= config.document_processing['max_segment_tokens']:
            chunks.append({
                "title": f"{segment['title']} (Part {len(chunks) + 1})",
                "content": " ".join(current_chunk),
                "tokens": current_tokens
            })
            current_chunk = []
            current_tokens = 0
    if current_chunk:
        chunks.append({
            "title": f"{segment['title']} (Part {len(chunks) + 1})",
            "content": " ".join(current_chunk),
            "tokens": current_tokens
        })
    return chunks

def save_segments_to_json(segments: List[Dict], output_file: str):
    """Save the processed segments to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(segments, f, indent=2)
        logger.info(f"Segments saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving segments to {output_file}: {e}", exc_info=True)

def main(pdf_path: str, toc_json_path: str, output_file: str):
    """Main function to segment a PDF and save the results."""
    try:
        raw_segments = segment_pdf(pdf_path, toc_json_path)
        processed_segments = post_process_segments(raw_segments)
        save_segments_to_json(processed_segments, output_file)
        logger.info(f"PDF segmentation completed for {pdf_path}. Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error in main PDF segmentation process for {pdf_path}: {e}", exc_info=True)
        error_segment = [{
            "title": "Processing Error",
            "content": f"An error occurred while processing {pdf_path}: {str(e)}",
            "tokens": 20
        }]
        save_segments_to_json(error_segment, output_file)

if __name__ == "__main__":
    pdf_path = config.file_paths['pdf_path']
    toc_json_path = config.file_paths['toc_json_path']
    output_file = config.file_paths['output_file']
    main(pdf_path, toc_json_path, output_file)
