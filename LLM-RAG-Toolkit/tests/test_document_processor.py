import unittest
import os
import json
from src.document_processing.document_processor import DocumentProcessor
from utils.config_manager import config

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()
        self.raw_directory = config.file_paths['input_directory']
        
        # Find a PDF file in the raw directory
        pdf_files = [f for f in os.listdir(self.raw_directory) if f.lower().endswith('.pdf')]
        if not pdf_files:
            raise ValueError("No PDF files found in the raw directory. Please add a PDF file for testing.")
        self.sample_pdf_path = os.path.join(self.raw_directory, pdf_files[0])
        
        # Find an image file in the raw directory (if available)
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff')
        image_files = [f for f in os.listdir(self.raw_directory) if f.lower().endswith(image_extensions)]
        self.sample_image_path = os.path.join(self.raw_directory, image_files[0]) if image_files else None

    def test_detect_document_type(self):
        self.assertEqual(self.processor.detect_document_type(self.sample_pdf_path), 'pdf')
        if self.sample_image_path:
            self.assertEqual(self.processor.detect_document_type(self.sample_image_path), 'image')
        
        with self.assertRaises(ValueError):
            self.processor.detect_document_type("invalid_file.txt")

    def test_process_pdf(self):
        result = self.processor.process_pdf(self.sample_pdf_path)
        self.assertIsInstance(result, dict)
        self.assertIn('file_path', result)
        self.assertIn('metadata', result)
        self.assertIn('toc', result)
        self.assertIn('segments', result)
        self.assertIn('semantic_data', result)

        print("\n--- PDF Processing Results ---")
        print(f"File: {os.path.basename(result['file_path'])}")
        print("\nMetadata:")
        print(json.dumps(result['metadata'], indent=2))
        print("\nTable of Contents:")
        print(json.dumps(result['toc'], indent=2))
        print("\nSegments (first 3):")
        for segment in result['segments'][:3]:
            print(f"- Title: {segment['title']}")
            print(f"  Content: {segment['content'][:100]}...")
            print(f"  Tokens: {segment['tokens']}")
        print(f"\nTotal segments: {len(result['segments'])}")
        print("\nSemantic Data:")
        print(json.dumps(result['semantic_data'], indent=2))

    def test_process_image(self):
        if self.sample_image_path:
            result = self.processor.process_image(self.sample_image_path)
            self.assertIsInstance(result, dict)
            self.assertIn('file_path', result)
            self.assertIn('metadata', result)
            self.assertIn('segments', result)
            self.assertIn('semantic_data', result)

            print("\n--- Image Processing Results ---")
            print(f"File: {os.path.basename(result['file_path'])}")
            print("\nMetadata:")
            print(json.dumps(result['metadata'], indent=2))
            print("\nSegments:")
            for segment in result['segments']:
                print(f"- Title: {segment['title']}")
                print(f"  Content: {segment['content'][:100]}...")
                print(f"  Tokens: {segment['tokens']}")
            print("\nSemantic Data:")
            print(json.dumps(result['semantic_data'], indent=2))

            if "Image text extraction not available" in result['segments'][0]['content']:
                print("Warning: Tesseract OCR not installed. Text extraction from images is limited.")
        else:
            print("Warning: No image file found for testing. Skipping image processing test.")

    def test_segment_image_text(self):
        sample_text = "This is a sample text.\nIt has multiple lines.\nWe will test segmentation."
        result = self.processor.segment_image_text(sample_text)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(segment, dict) for segment in result))
        self.assertTrue(all('title' in segment and 'content' in segment and 'tokens' in segment for segment in result))

        print("\n--- Image Text Segmentation Results ---")
        for segment in result:
            print(f"- Title: {segment['title']}")
            print(f"  Content: {segment['content']}")
            print(f"  Tokens: {segment['tokens']}")

    def test_perform_semantic_analysis(self):
        sample_segments = [
            {"title": "Segment 1", "content": "This is the content of segment 1."},
            {"title": "Segment 2", "content": "This is the content of segment 2."}
        ]
        result = self.processor.perform_semantic_analysis(sample_segments)
        self.assertIsInstance(result, dict)
        self.assertIn('main_topics', result)
        self.assertIn('key_entities', result)
        self.assertIn('summary', result)

        print("\n--- Semantic Analysis Results ---")
        print(json.dumps(result, indent=2))

    def test_process_document(self):
        result_pdf = self.processor.process_document(self.sample_pdf_path)
        self.assertIsInstance(result_pdf, dict)
        self.assertEqual(result_pdf['file_path'], self.sample_pdf_path)

        print("\n--- PDF Document Processing Results ---")
        print(f"File: {os.path.basename(result_pdf['file_path'])}")
        print(f"Total segments: {len(result_pdf['segments'])}")
        print("Metadata keys:", list(result_pdf['metadata'].keys()))
        print("Semantic data keys:", list(result_pdf['semantic_data'].keys()))

        if self.sample_image_path:
            result_image = self.processor.process_document(self.sample_image_path)
            self.assertIsInstance(result_image, dict)
            self.assertEqual(result_image['file_path'], self.sample_image_path)

            print("\n--- Image Document Processing Results ---")
            print(f"File: {os.path.basename(result_image['file_path'])}")
            print(f"Total segments: {len(result_image['segments'])}")
            print("Metadata keys:", list(result_image['metadata'].keys()))
            print("Semantic data keys:", list(result_image['semantic_data'].keys()))
        else:
            print("Warning: No image file found for testing. Skipping image document processing test.")

if __name__ == '__main__':
    unittest.main()