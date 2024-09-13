import os
import re
import json
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import spacy
import tiktoken
from langdetect import detect, LangDetectException
from collections import defaultdict
from utils.config_manager import config

logger = logging.getLogger(__name__)

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Initialize tokenizer for accurate token counts
tokenizer = tiktoken.get_encoding('cl100k_base')

# Load spaCy model for semantic analysis
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Download the model if not present
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def get_token_count(text: str) -> int:
    """Calculate the number of tokens in the text using the tokenizer."""
    tokens = tokenizer.encode(text)
    return len(tokens)


def load_toc(toc_json_path: str) -> Dict:
    """Load the table of contents from a JSON file.

    Args:
        toc_json_path (str): The path to the TOC JSON file.

    Returns:
        Dict: The TOC data.
    """
    try:
        with open(toc_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"TOC file not found: {toc_json_path}", exc_info=True)
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in TOC file: {toc_json_path}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Error loading TOC from {toc_json_path}: {e}", exc_info=True)
        return {}


def extract_toc_from_pdf(pdf_path: str) -> Dict:
    """Extract the table of contents from the PDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        Dict: The TOC data structured as a nested dictionary.
    """
    toc_dict = {}
    try:
        with fitz.open(pdf_path) as doc:
            toc = doc.get_toc(simple=False)
            for entry in toc:
                if len(entry) >= 3:  # Ensure we have at least level, title, and page number
                    level, title, page_num = entry[:3]
                    insert_toc_entry(toc_dict, level, title.strip())
        return toc_dict
    except Exception as e:
        logger.error(f"Error extracting TOC from PDF {os.path.basename(pdf_path)}: {e}", exc_info=True)
        return {}


def insert_toc_entry(toc_dict: Dict, level: int, title: str):
    """Insert a TOC entry into the nested dictionary based on its level.

    Args:
        toc_dict (Dict): The current TOC dictionary.
        level (int): The level of the TOC entry.
        title (str): The title of the TOC entry.
    """
    current_level = toc_dict
    for _ in range(level - 1):
        if not current_level:
            current_level[title] = {}
            return
        last_key = next(reversed(current_level))
        current_level = current_level[last_key]
    current_level[title] = {}


def create_toc_patterns(toc_structure: Dict, level: int = 1) -> List[Tuple[int, str, re.Pattern]]:
    """Create regex patterns from the TOC structure recursively.

    Args:
        toc_structure (Dict): The TOC data.
        level (int): The current level in the TOC hierarchy.

    Returns:
        List[Tuple[int, str, re.Pattern]]: A list of tuples containing level, section titles, and regex patterns.
    """
    patterns = []
    for section, subsections in toc_structure.items():
        pattern = re.compile(rf"^\s*{re.escape(section)}[\s]*$", re.IGNORECASE)
        patterns.append((level, section, pattern))
        if isinstance(subsections, dict) and level < config.document_processing.get('max_toc_depth', 5):
            patterns.extend(create_toc_patterns(subsections, level + 1))
    return patterns


def extract_images(pdf_path: str) -> List[Dict]:
    """Extract images from the PDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict]: A list of dictionaries containing image data.
    """
    images = []
    try:
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    images.append({
                        "page": page_num,
                        "type": base_image.get("ext", ""),
                        "size": len(base_image["image"]),
                        "content": base_image["image"],
                        "width": base_image.get("width", 0),
                        "height": base_image.get("height", 0)
                    })
    except Exception as e:
        logger.error(f"Error extracting images from PDF {os.path.basename(pdf_path)}: {e}", exc_info=True)
    return images


def extract_text_with_headings(pdf_path: str) -> List[Dict]:
    """Extract text from the PDF along with heading information.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict]: A list of dictionaries containing text and heading information.
    """
    content = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    content.append({
                                        "text": text,
                                        "font_size": span["size"],
                                        "font_flags": span["flags"],
                                        "page_num": page_num
                                    })
    except Exception as e:
        logger.error(f"Error extracting text from PDF {os.path.basename(pdf_path)}: {e}", exc_info=True)
    return content


def is_heading(span: Dict[str, Any], avg_font_size: float) -> bool:
    """Determine if a text span is a heading based on font size and style.

    Args:
        span (Dict[str, Any]): The text span dictionary.
        avg_font_size (float): The average font size on the page.

    Returns:
        bool: True if the span is considered a heading, False otherwise.
    """
    # Consider text as heading if font size is larger than average or if it's bold
    is_bold = span["font_flags"] & 2  # Bold flag
    return span["font_size"] > avg_font_size * 1.2 or is_bold


def segment_pdf(pdf_path: str, toc_data: Dict[str, Any] = None) -> List[Dict]:
    """Segment the PDF using TOC if available, otherwise use semantic segmentation.

    Args:
        pdf_path (str): The path to the PDF file.
        toc_data (Dict[str, Any], optional): The TOC data. Defaults to None.

    Returns:
        List[Dict]: A list of segmented content from the PDF.
    """
    if not toc_data:
        toc_data = extract_toc_from_pdf(pdf_path)
    if toc_data:
        toc_patterns = create_toc_patterns(toc_data)
        return segment_pdf_using_toc(pdf_path, toc_patterns)
    else:
        return segment_pdf_with_semantics(pdf_path)


def segment_pdf_using_toc(pdf_path: str, toc_patterns: List[Tuple[int, str, re.Pattern]]) -> List[Dict]:
    """Segment the PDF using the table of contents structure.

    Args:
        pdf_path (str): The path to the PDF file.
        toc_patterns (List[Tuple[int, str, re.Pattern]]): A list of TOC patterns with levels.

    Returns:
        List[Dict]: A list of segmented content from the PDF.
    """
    segments = []
    images = extract_images(pdf_path)
    image_index = 0
    current_segment = {
        "title": "Introduction",
        "content": [],
        "tokens": 0,
        "level": 1,
        "path": ["Introduction"]
    }

    content = extract_text_with_headings(pdf_path)
    avg_font_size = sum(span['font_size'] for span in content) / len(content)

    for span in tqdm(content, desc="Processing content"):
        text = span['text']
        matched = False
        for level, title, pattern in toc_patterns:
            if pattern.match(text):
                # Save current segment
                if current_segment['content']:
                    segment_content = " ".join(current_segment['content'])
                    tokens = get_token_count(segment_content)
                    segments.append({
                        "title": current_segment['title'],
                        "content": segment_content,
                        "tokens": tokens,
                        "level": current_segment['level'],
                        "path": current_segment['path']
                    })
                # Start new segment
                current_segment = {
                    "title": title,
                    "content": [],
                    "tokens": 0,
                    "level": level,
                    "path": current_segment['path'][:level - 1] + [title]
                }
                matched = True
                break

        if not matched:
            current_segment['content'].append(text)
            current_segment['tokens'] += get_token_count(text)

        if current_segment['tokens'] >= config.document_processing['max_segment_tokens']:
            segment_content = " ".join(current_segment['content'])
            tokens = get_token_count(segment_content)
            segments.append({
                "title": current_segment['title'],
                "content": segment_content,
                "tokens": tokens,
                "level": current_segment['level'],
                "path": current_segment['path']
            })
            # Reset current segment content but keep the same title and path
            current_segment['content'] = []
            current_segment['tokens'] = 0

    # Add any remaining content
    if current_segment['content']:
        segment_content = " ".join(current_segment['content'])
        tokens = get_token_count(segment_content)
        segments.append({
            "title": current_segment['title'],
            "content": segment_content,
            "tokens": tokens,
            "level": current_segment['level'],
            "path": current_segment['path']
        })

    # Process images
    image_segments = process_images(images)
    segments.extend(image_segments)

    return segments


def segment_pdf_with_semantics(pdf_path: str) -> List[Dict]:
    """Segment the PDF using semantic analysis for better RAG compatibility.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict]: A list of segmented content from the PDF.
    """
    segments = []
    images = extract_images(pdf_path)

    try:
        content = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                if text:
                    content.append(text)
        full_text = "\n".join(content)
        # Perform semantic segmentation
        segments = semantic_segmentation(full_text)
        # Process images
        image_segments = process_images(images)
        segments.extend(image_segments)
    except Exception as e:
        logger.error(f"Error processing PDF {os.path.basename(pdf_path)}: {e}", exc_info=True)

    return segments


def semantic_segmentation(text: str) -> List[Dict]:
    """Segment text based on semantic coherence.

    Args:
        text (str): The full text to segment.

    Returns:
        List[Dict]: A list of text segments.
    """
    sentences = sent_tokenize(text)
    max_tokens = config.document_processing['max_segment_tokens']
    overlap = config.document_processing.get('overlap_sentences', 2)
    segments = []
    current_segment = []
    current_tokens = 0
    part = 1
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = get_token_count(sentence)
        if current_tokens + sentence_tokens > max_tokens and current_segment:
            # Create segment
            segment_content = " ".join(current_segment)
            tokens = get_token_count(segment_content)
            segments.append({
                "title": f"Segment {part}",
                "content": segment_content,
                "tokens": tokens,
                "level": 1,
                "path": [f"Segment {part}"]
            })
            # Prepare for next segment with overlap
            part += 1
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
            "title": f"Segment {part}",
            "content": segment_content,
            "tokens": tokens,
            "level": 1,
            "path": [f"Segment {part}"]
        })

    return segments


def process_images(images: List[Dict]) -> List[Dict]:
    """Process images and create segments for them.

    Args:
        images (List[Dict]): A list of image data dictionaries.

    Returns:
        List[Dict]: A list of image segments.
    """
    processed_images = []
    for i, img in enumerate(images):
        title = f"Image {i + 1}"
        content = f"[Image: {img['type']} format, size: {img['width']}x{img['height']} pixels, on page: {img['page'] + 1}]"
        tokens = get_token_count(content)
        processed_images.append({
            "title": title,
            "content": content,
            "tokens": tokens,
            "level": 1,
            "path": [title],
            "image_data": {
                "page": img["page"],
                "type": img["type"],
                "width": img["width"],
                "height": img["height"]
            }
        })
    return processed_images


def post_process_segments(segments: List[Dict]) -> List[Dict]:
    """Post-process segments to ensure they meet the desired format and token limits.

    Args:
        segments (List[Dict]): A list of segments.

    Returns:
        List[Dict]: A list of processed segments.
    """
    processed_segments = []
    max_tokens = config.document_processing['max_segment_tokens']
    overlap = config.document_processing.get('overlap_sentences', 2)

    for segment in segments:
        if "image_data" in segment:
            processed_segments.append(segment)
            continue

        content = re.sub(r'\s+', ' ', segment['content']).strip()
        tokens = get_token_count(content)

        if tokens > max_tokens:
            split_segments = split_long_segment(segment, content, max_tokens, overlap)
            processed_segments.extend(split_segments)
        else:
            segment['content'] = content
            segment['tokens'] = tokens
            processed_segments.append(segment)

    if not processed_segments:
        processed_segments.append({
            "title": "Empty Document",
            "content": "This document contains no extractable content.",
            "tokens": get_token_count("This document contains no extractable content."),
            "level": 1,
            "path": ["Empty Document"]
        })

    return processed_segments


def split_long_segment(segment: Dict, content: str, max_tokens: int, overlap: int) -> List[Dict]:
    """Split a long segment into smaller chunks based on sentence boundaries.

    Args:
        segment (Dict): The original segment dictionary.
        content (str): The content to be split.
        max_tokens (int): The maximum number of tokens per segment.
        overlap (int): The number of sentences to overlap between segments.

    Returns:
        List[Dict]: A list of split segments.
    """
    sentences = sent_tokenize(content)
    chunks = []
    current_chunk = []
    current_tokens = 0
    part = 1
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = get_token_count(sentence)
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunk_content = " ".join(current_chunk)
            tokens = get_token_count(chunk_content)
            new_segment = segment.copy()
            new_segment['title'] = f"{segment['title']} (Part {part})"
            new_segment['content'] = chunk_content
            new_segment['tokens'] = tokens
            new_segment['path'] = segment['path'] + [f"Part {part}"]
            chunks.append(new_segment)
            part += 1
            current_chunk = current_chunk[-overlap:]  # Keep last 'overlap' sentences
            current_tokens = get_token_count(" ".join(current_chunk))
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1

    if current_chunk:
        chunk_content = " ".join(current_chunk)
        tokens = get_token_count(chunk_content)
        new_segment = segment.copy()
        new_segment['title'] = f"{segment['title']} (Part {part})"
        new_segment['content'] = chunk_content
        new_segment['tokens'] = tokens
        new_segment['path'] = segment['path'] + [f"Part {part}"]
        chunks.append(new_segment)

    return chunks


def save_segments_to_json(segments: List[Dict], output_file: str):
    """Save the processed segments to a JSON file.

    Args:
        segments (List[Dict]): A list of processed segments.
        output_file (str): The output file path.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        logger.info(f"Segments saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving segments to {output_file}: {e}", exc_info=True)


def main(pdf_path: str, toc_json_path: str, output_file: str):
    """Main function to segment a PDF and save the results.

    Args:
        pdf_path (str): The path to the PDF file.
        toc_json_path (str): The path to the TOC JSON file.
        output_file (str): The output file path.
    """
    try:
        toc_data = load_toc(toc_json_path)
        if not toc_data:
            toc_data = extract_toc_from_pdf(pdf_path)
        raw_segments = segment_pdf(pdf_path, toc_data)
        processed_segments = post_process_segments(raw_segments)
        save_segments_to_json(processed_segments, output_file)
        logger.info(f"PDF segmentation completed for {os.path.basename(pdf_path)}. Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error in main PDF segmentation process for {os.path.basename(pdf_path)}: {e}", exc_info=True)
        error_segment = [{
            "title": "Processing Error",
            "content": f"An error occurred while processing {os.path.basename(pdf_path)}: {str(e)}",
            "tokens": get_token_count(f"An error occurred while processing {os.path.basename(pdf_path)}: {str(e)}"),
            "level": 1,
            "path": ["Processing Error"]
        }]
        save_segments_to_json(error_segment, output_file)


if __name__ == "__main__":
    pdf_path = config.file_paths['pdf_path']
    toc_json_path = config.file_paths['toc_json_path']
    output_file = config.file_paths['output_file']
    main(pdf_path, toc_json_path, output_file)
