from pathlib import Path
from typing import List, Dict, Any
from gemini_client import GeminiClient
from vector_store import VectorStore
from config import TOP_K_RETRIEVAL
from utils import extract_images_from_pdf, extract_tables_from_pdf, extract_text_from_pdf, save_metadata

class PDFProcessor:
    def __init__(self, gemini_client: GeminiClient, vector_store: VectorStore):
        self.gemini_client = gemini_client
        self.vector_store = vector_store
        self.all_chunks = []
    
    def process_pdf(self, pdf_path: str) -> Dict[str, int]:
        print(f"Processing PDF: {pdf_path}")
        text_chunks = extract_text_from_pdf(pdf_path)
        image_chunks = extract_images_from_pdf(pdf_path)
        table_chunks = extract_tables_from_pdf(pdf_path)

        for chunk in image_chunks:
            if chunk['image_path']:
                caption = self.gemini_client.generate_image_caption(chunk['image_path'])
                chunk['content'] = caption

        all_chunks = text_chunks + image_chunks + table_chunks
        self.all_chunks.extend(all_chunks)
        doc_name = Path(pdf_path).stem
        save_metadata(all_chunks, doc_name)

        return {
            'text_chunks': len(text_chunks),
            'image_chunks': len(image_chunks),
            'table_chunks': len(table_chunks),
            'total_chunks': len(all_chunks)
        }
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        results = {}
        for pdf_path in pdf_paths:
            try:
                result = self.process_pdf(pdf_path)
                results[Path(pdf_path).stem] = result
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                results[Path(pdf_path).stem] = {'error': str(e)}
        return results
    
    def build_vector_index(self):
        if self.all_chunks:
            self.vector_store.build_index(self.all_chunks)
            self.vector_store.save_index("multimodal_rag")
    
    def query(self, question: str) -> Dict[str, Any]:
        relevant_chunks = self.vector_store.search(question, TOP_K_RETRIEVAL)
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the documents.",
                "images": [],
                "tables": []
            }
        result = self.gemini_client.generate_answer(question, relevant_chunks)
        return result
