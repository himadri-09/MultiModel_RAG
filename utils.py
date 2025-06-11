import fitz  # PyMuPDF
import pdfplumber
import pickle
import json
from typing import List, Dict, Any, Tuple
from PIL import Image
import pandas as pd
from pathlib import Path
from config import CHUNK_SIZE, METADATA_DIR,IMAGES_DIR

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text_chunks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        if text.strip():
            # Split text into chunks
            words = text.split()
            for i in range(0, len(words), CHUNK_SIZE):
                chunk_words = words[i:i + CHUNK_SIZE]
                chunk_text = " ".join(chunk_words)
                
                text_chunks.append({
                    'content': chunk_text,
                    'type': 'text',
                    'page_number': page_num + 1,
                    'doc_name': Path(pdf_path).stem,
                    'metadata': {
                        'word_count': len(chunk_words),
                        'char_count': len(chunk_text)
                    }
                })
    
    doc.close()
    return text_chunks

# def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
#     """Extract images from PDF using PyMuPDF"""
#     doc = fitz.open(pdf_path)
#     image_chunks = []
#     doc_name = Path(pdf_path).stem
    
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         image_list = page.get_images()
        
#         for img_index, img in enumerate(image_list):
#             xref = img[0]
#             pix = fitz.Pixmap(doc, xref)
            
#             if pix.n - pix.alpha < 4:  # GRAY or RGB
#                 img_path = IMAGES_DIR / f"{doc_name}_page_{page_num + 1}_img_{img_index}.png"
#                 pix.save(str(img_path))
                
#                 image_chunks.append({
#                     'content': '',  # Will be filled by image captioning
#                     'type': 'image',
#                     'page_number': page_num + 1,
#                     'doc_name': doc_name,
#                     'image_path': str(img_path),
#                     'metadata': {
#                         'width': pix.width,
#                         'height': pix.height,
#                         'image_index': img_index
#                     }
#                 })
            
#             pix = None
    
#     doc.close()
#     return image_chunks

def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract images from PDF using PyMuPDF, along with heading metadata."""
    doc = fitz.open(pdf_path)
    image_chunks = []
    doc_name = Path(pdf_path).stem
    
    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_path = IMAGES_DIR / f"{doc_name}_page_{page_num + 1}_img_{img_index}.png"
                pix.save(str(img_path))
                
                # --- Extract first heading or nearby text ---
                heading = ""
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:  # text block
                        heading = block["lines"][0]["spans"][0]["text"]
                        break
                
                image_chunks.append({
                    'content': '',  # Will be filled by image captioning
                    'type': 'image',
                    'page_number': page_num + 1,
                    'doc_name': doc_name,
                    'image_path': str(img_path),
                    'metadata': {
                        'width': pix.width,
                        'height': pix.height,
                        'image_index': img_index,
                        'heading': heading
                    }
                })
            
            pix = None
    
    doc.close()
    return image_chunks


def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using pdfplumber"""
    table_chunks = []
    doc_name = Path(pdf_path).stem
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            
            for table_index, table in enumerate(tables):
                if table and len(table) > 1:  # Ensure table has header and data
                    # Convert table to DataFrame
                    df = pd.DataFrame(table[1:], columns=table[0])

                    # Remove duplicate columns (fix!)
                    df = df.loc[:, ~df.columns.duplicated()]
                    
                    # Convert to markdown
                    markdown_table = df.to_markdown(index=False)
                    
                    # Convert to JSON
                    table_json = df.to_json(orient='records')
                    
                    table_chunks.append({
                        'content': markdown_table,
                        'type': 'table',
                        'page_number': page_num + 1,
                        'doc_name': doc_name,
                        'table_json': table_json,
                        'metadata': {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'table_index': table_index,
                            'column_names': list(df.columns)
                        }
                    })
    
    return table_chunks


def save_metadata(chunks: List[Dict[str, Any]], filename: str):
    """Save metadata to pickle file"""
    metadata_path = METADATA_DIR / f"{filename}_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)

def load_metadata(filename: str) -> List[Dict[str, Any]]:
    """Load metadata from pickle file"""
    metadata_path = METADATA_DIR / f"{filename}_metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    return []