import streamlit as st
import tempfile
import os
import json
from pathlib import Path
from evaluation import evaluate_document

# Import our modules
from config import *
from gemini_client import GeminiClient
from vector_store import VectorStore
from pdf_processor import PDFProcessor

# Page configuration
st.set_page_config(
    page_title="Multimodal PDF RAG System",
    page_icon="📚",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    initialize_session_state()
    
    st.title("📚 Multimodal PDF RAG System")
    st.markdown("Upload PDFs and ask questions about their content (text, images, and tables). You can also evaluate the system's answers.")

    # Sidebar for PDF upload and Evaluation
    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to process"
        )
        
        if uploaded_files and st.button("Process PDFs"):
            process_pdfs(uploaded_files)
        
        st.markdown("---")
        st.header("Evaluation Mode")
        evaluation_file = st.file_uploader(
            "Upload Evaluation JSON file",
            type="json",
            help="This JSON file should contain either:\n"
                 "1️⃣ a dict with 'questions' and 'answers' lists\n"
                 "2️⃣ a list of {'question', 'answer'} pairs"
        )
        if evaluation_file and st.button("Run Evaluation"):
            run_evaluation_mode(evaluation_file)

    # Main content area
    if st.session_state.pdf_processor:
        st.header("Ask Questions")
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("Processed Files")
            for file_info in st.session_state.processed_files:
                with st.expander(f"📄 {file_info['name']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Text Chunks", file_info['text_chunks'])
                    with col2:
                        st.metric("Image Chunks", file_info['image_chunks'])
                    with col3:
                        st.metric("Table Chunks", file_info['table_chunks'])
                    with col4:
                        st.metric("Total Chunks", file_info['total_chunks'])
        
        # Chat interface
        st.subheader("Chat with your Documents")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Q{i+1}:** {question}")
                st.markdown(f"**A{i+1}:** {answer}")
                st.divider()
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings in the research paper?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Ask", type="primary"):
                if question:
                    ask_question(question)
                else:
                    st.warning("Please enter a question.")
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    else:
        st.info("Upload and process PDF files to start asking questions.")
        
        # Display example questions
        st.subheader("Example Questions You Can Ask:")
        example_questions = [
            "What are the main topics covered in the documents?",
            "Summarize the key findings from the tables.",
            "What do the images show?",
            "What are the conclusions mentioned in the documents?",
            "Extract all numerical data from the tables.",
            "Describe any charts or graphs present in the documents."
        ]
        
        for question in example_questions:
            st.markdown(f"• {question}")

def process_pdfs(uploaded_files):
    """Process uploaded PDF files"""
    with st.spinner("Processing PDFs..."):
        try:
            # Initialize components
            gemini_client = GeminiClient(GOOGLE_API_KEY)
            vector_store = VectorStore()
            pdf_processor = PDFProcessor(gemini_client, vector_store)
            
            # Save uploaded files temporarily
            temp_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_paths.append(tmp_file.name)
            
            # Process PDFs
            results = pdf_processor.process_multiple_pdfs(temp_paths)
            
            # Build vector index
            pdf_processor.build_vector_index()
            
            # Store in session state
            st.session_state.pdf_processor = pdf_processor
            st.session_state.processed_files = []
            
            for i, (file_name, result) in enumerate(results.items()):
                if 'error' not in result:
                    st.session_state.processed_files.append({
                        'name': uploaded_files[i].name,
                        **result
                    })
            
            # Clean up temporary files
            for temp_path in temp_paths:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            st.success(f"Successfully processed {len(uploaded_files)} PDF files!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing PDFs: {e}")

def ask_question(question):
    """Process a user question"""
    with st.spinner("Searching for relevant information..."):
        try:
            result = st.session_state.pdf_processor.query(question)
            if isinstance(result, dict) and 'answer' in result:
                answer = result['answer']
            else:
                answer = result  # fallback in case it's not a dict
            st.session_state.chat_history.append((question, answer))
            st.rerun()
        except Exception as e:
            st.error(f"Error processing question: {e}")

def run_evaluation_mode(evaluation_file):
    if st.session_state.pdf_processor:
        document_name = "transformer_comparison"  # Or let user select
        results = evaluate_document(document_name, st.session_state.pdf_processor)

        for i, res in enumerate(results):
            st.markdown(f"**Q{i+1}:** {res['question']}")
            st.markdown(f"**System Answer:** {res['generated_answer']}")
            st.markdown(f"**Ground Truth:** {res['ground_truth_answer']}")
            st.markdown(f"**Similarity Score:** {res['similarity']:.4f}")
            st.divider()

        avg_score = sum(r["similarity"] for r in results) / len(results)
        st.success(f"✅ Evaluation complete! Average Similarity Score: {avg_score:.4f}")
    else:
        st.warning("Please process PDFs first!")



if __name__ == "__main__":
    main()
