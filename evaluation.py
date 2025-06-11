# evaluation.py

import json
from pathlib import Path
import numpy as np
from config import GOOGLE_API_KEY
from gemini_client import GeminiClient
from vector_store import VectorStore
from pdf_processor import PDFProcessor
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
embedding_model = genai.GenerativeModel("models/embedding-001")

def load_ground_truth(document_name):
    gt_path = Path("data/ground_truth") / f"{document_name}.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_embedding(text):
    """Use Gemini embedding API to get embeddings for text"""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return response["embedding"]



def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def evaluate_document(document_name, pdf_processor):
    gt_data = load_ground_truth(document_name)
    results = []

    for qa in gt_data:
        question = qa.get("question", "")
        gt_answer = qa.get("ground_truth_answer", "")

        if not question or not gt_answer:
            continue

        # Generate answer using your system
        generated_result = pdf_processor.query(question)
        if isinstance(generated_result, dict):
            generated_answer = generated_result.get("answer", next(iter(generated_result.values()), ""))
        else:
            generated_answer = generated_result or ""

        # Get embeddings
        gt_embedding = get_embedding(gt_answer)
        gen_embedding = get_embedding(generated_answer)

        # Compute similarity
        similarity = cosine_similarity(gt_embedding, gen_embedding)

        results.append({
            "question": question,
            "ground_truth_answer": gt_answer,
            "generated_answer": generated_answer,
            "similarity": similarity
        })

    return results

if __name__ == "__main__":
    document_name = "transformer_comparison"
    gemini_client = GeminiClient(api_key=GOOGLE_API_KEY)
    vector_store = VectorStore()
    pdf_processor = PDFProcessor(gemini_client, vector_store)

    results = evaluate_document(document_name, pdf_processor)

    for i, res in enumerate(results):
        print(f"Q{i+1}: {res['question']}")
        print(f"Ground Truth: {res['ground_truth_answer']}")
        print(f"Generated Answer: {res['generated_answer']}")
        print(f"Similarity Score: {res['similarity']:.4f}")
        print("=" * 60)
