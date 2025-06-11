import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Any
from config import GEMINI_MODEL,Path
import base64
import io

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def generate_image_caption(self, image_path: str) -> str:
        """Generate caption for image using Gemini Vision"""
        try:
            image = Image.open(image_path)
            
            prompt = """
            Analyze this image and provide a detailed description. 
            Focus on:
            1. Main objects, people, or content visible
            2. Text or numbers if present (OCR)
            3. Charts, graphs, or diagrams if present
            4. Overall context and purpose
            
            Provide a comprehensive description that would help someone understand the image content without seeing it.
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return f"Image from {Path(image_path).name}"
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer based on retrieved context"""
        try:
            # Prepare context
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                if chunk['type'] == 'text':
                    context_parts.append(f"Text Context {i+1}:\n{chunk['content']}")
                elif chunk['type'] == 'table':
                    context_parts.append(f"Table Context {i+1} (Page {chunk['page_number']}):\n{chunk['content']}")
                elif chunk['type'] == 'image':
                    context_parts.append(f"Image Context {i+1} (Page {chunk['page_number']}):\n{chunk['content']}")
            
            context_text = "\n\n".join(context_parts)
            
            prompt = f"""
            Based on the following context from PDF documents, please answer the user's question.
            
            Context:
            {context_text}
            
            Question: {query}
            
            Instructions:
            1. Provide a comprehensive answer based on the retrieved context
            2. If the context includes tables, reference specific data points
            3. If the context includes images, reference visual information
            4. Cite which page numbers the information comes from
            5. If the context doesn't contain enough information, say so clearly
            
            Answer:
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating answer: {e}"