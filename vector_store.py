import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
import pickle
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, VECTORS_DIR

class VectorStore:
    def __init__(self, api_key: str = None):
        genai.configure(api_key=api_key or GOOGLE_API_KEY)
        self.index = None
        self.chunks = []
        self.dimension = 768  # Gemini embedding dimension

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []

        for text in texts:
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document",
                    title="PDF Content"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Error creating embedding for text: {e}")
                embeddings.append([0.0] * self.dimension)

        return np.array(embeddings, dtype=np.float32)

    def create_query_embedding(self, query: str) -> np.ndarray:
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
        """Build FAISS index from non-empty chunks, log skipped ones."""
        filtered_chunks = []
        valid_texts = []

        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                doc = chunk.get("doc_name", "UnknownDoc")
                page = chunk.get("page_number", "N/A")
                ctype = chunk.get("type", "unknown")
                print(f"⚠️ Skipping empty chunk | Type: {ctype} | Page: {page} | Doc: {doc}")
                continue
            filtered_chunks.append(chunk)
            valid_texts.append(content)

        if not valid_texts:
            print("⚠️ No valid content found for indexing.")
            return

        embeddings = self.create_embeddings(valid_texts)

        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks = filtered_chunks

        print(f"✅ Built index with {len(filtered_chunks)} valid chunks")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
        if self.index is not None:
            index_path = VECTORS_DIR / f"{filename}_index.faiss"
            chunks_path = VECTORS_DIR / f"{filename}_chunks.pkl"

            faiss.write_index(self.index, str(index_path))
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)

    def load_index(self, filename: str) -> bool:
        index_path = VECTORS_DIR / f"{filename}_index.faiss"
        chunks_path = VECTORS_DIR / f"{filename}_chunks.pkl"

        if index_path.exists() and chunks_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            return True
        return False
