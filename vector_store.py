import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
import pickle
from config import GOOGLE_API_KEY, EMBEDDING_MODEL,VECTORS_DIR

class VectorStore:
    def __init__(self, api_key: str = None):
        # Configure Gemini for embeddings
        genai.configure(api_key=api_key or GOOGLE_API_KEY)
        self.index = None
        self.chunks = []
        self.dimension = 768  # text-embedding-004 dimension
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using Gemini's embedding API"""
        embeddings = []
        
        for text in texts:
            try:
                # Use Gemini's embedding API
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document",
                    title="PDF Content"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Error creating embedding for text: {e}")
                # Fallback: create zero vector
                embeddings.append([0.0] * self.dimension)
        
        return np.array(embeddings, dtype=np.float32)
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            return np.array([result['embedding']], dtype=np.float32)
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return np.array([[0.0] * self.dimension], dtype=np.float32)
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from chunks"""
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.create_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.chunks = chunks
        print(f"Built index with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if self.index is None:
            return []
        
        query_embedding = self.create_query_embedding(query)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                results.append(chunk)
        
        return results
    
    def save_index(self, filename: str):
        """Save FAISS index and chunks"""
        if self.index is not None:
            index_path = VECTORS_DIR / f"{filename}_index.faiss"
            chunks_path = VECTORS_DIR / f"{filename}_chunks.pkl"
            
            faiss.write_index(self.index, str(index_path))
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
    
    def load_index(self, filename: str) -> bool:
        """Load FAISS index and chunks"""
        index_path = VECTORS_DIR / f"{filename}_index.faiss"
        chunks_path = VECTORS_DIR / f"{filename}_chunks.pkl"
        
        if index_path.exists() and chunks_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            return True
        return False
