from pathlib import Path
from typing import List, Dict, Any
from gemini_client import GeminiClient
from vector_store import VectorStore
from query_decomposer import QueryDecomposer
from config import TOP_K_RETRIEVAL
from utils import extract_images_from_pdf, extract_tables_from_pdf, extract_text_from_pdf, save_metadata

class PDFProcessor:
    def __init__(self, gemini_client: GeminiClient, vector_store: VectorStore):
        self.gemini_client = gemini_client
        self.vector_store = vector_store
        self.query_decomposer = QueryDecomposer()
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
    
    def query_complex(self, question: str) -> Dict[str, Any]:
        print(f"\n🔴 Processing COMPLEX query: '{question}'")
        
        sub_queries = self.query_decomposer.decompose_query(question)
        
        all_chunks_collected = []
        sub_query_answers = []
        
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"\n🔍 Processing sub-query {i}: '{sub_query}'")
            
            relevant_chunks = self.vector_store.search(sub_query, TOP_K_RETRIEVAL)
            
            if relevant_chunks:
                print(f"📊 Found {len(relevant_chunks)} chunks for sub-query {i}")
                for j, chunk in enumerate(relevant_chunks, 1):
                    score = chunk.get('similarity_score', 0)
                    chunk_type = chunk.get('type', 'unknown')
                    page = chunk.get('page_number', 'N/A')
                    print(f"      {j}. Type: {chunk_type}, Page: {page}, Score: {score:.4f}")
                
                sub_answer = self.gemini_client.generate_answer(sub_query, relevant_chunks)
                
                sub_query_answers.append({
                    'question': sub_query,
                    'answer': sub_answer,
                    'chunks_count': len(relevant_chunks)
                })
                
                all_chunks_collected.extend(relevant_chunks)
                
                print(f"✅ Sub-query {i} answered (length: {len(sub_answer)} chars)")
            else:
                print(f"❌ No relevant chunks found for sub-query {i}")
                sub_query_answers.append({
                    'question': sub_query,
                    'answer': f"No relevant information found for: {sub_query}",
                    'chunks_count': 0
                })
        
        if all_chunks_collected:
            reranked_chunks = self.query_decomposer.rerank_chunks(all_chunks_collected, question)
        else:
            reranked_chunks = []
        
        if sub_query_answers and any(qa['chunks_count'] > 0 for qa in sub_query_answers):
            final_answer = self.query_decomposer.combine_answers(question, sub_query_answers)
        else:
            final_answer = "No relevant information found in the documents to answer your complex query."
        
        self.query_decomposer.log_query_flow(question, sub_queries, sub_query_answers, final_answer)
        
        return {
        "answer": final_answer,
        "method": "complex",
        "sub_queries": sub_queries,
        "sub_answers": sub_query_answers,
        "total_chunks_collected": len(all_chunks_collected),
        "reranked_chunks_used": len(reranked_chunks),
        "relevant_image_chunks": [c for c in reranked_chunks if c["type"] == "image"]
        }


    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n" + "="*60)
        print(f"🚀 PROCESSING QUERY: '{question}'")
        print("="*60)

        # Always treat every query as complex
        result = self.query_complex(question)

        print(f"\n✅ Query processing complete using {result['method']} method")
        return result
