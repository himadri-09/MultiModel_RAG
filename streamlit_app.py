import streamlit as st
import tempfile
import os
import json
from pathlib import Path
from evaluation import evaluate_document

from config import *
from gemini_client import GeminiClient
from vector_store import VectorStore
from pdf_processor import PDFProcessor

st.set_page_config(
    page_title="Multimodal PDF RAG System with Query Decomposition",
    page_icon="📚",
    layout="wide"
)

def initialize_session_state():
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    initialize_session_state()
    
    st.title("📚 Multimodal PDF RAG System with Query Decomposition")
    st.markdown("Upload PDFs and ask any question. The system **always decomposes** your queries into sub-questions to generate rich, context-aware answers from text, images, and tables.")

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
        st.header("Query Handling Info")
        st.info(
            "All queries are **automatically treated as complex** and broken down into sub-questions for improved retrieval and accuracy.\n\n"
            "Check the terminal/console for a detailed breakdown!"
        )
        
        st.markdown("---")
        st.header("Evaluation Mode")
        evaluation_file = st.file_uploader(
            "Upload Evaluation JSON file",
            type="json",
            help="Upload a JSON file containing 'question' and 'answer' pairs or lists."
        )
        if evaluation_file and st.button("Run Evaluation"):
            run_evaluation_mode(evaluation_file)

    if st.session_state.pdf_processor:
        st.header("Ask Questions")

        if st.session_state.processed_files:
            st.subheader("Processed Files")
            for file_info in st.session_state.processed_files:
                with st.expander(f"📄 {file_info['name']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Text Chunks", file_info['text_chunks'])
                    col2.metric("Image Chunks", file_info['image_chunks'])
                    col3.metric("Table Chunks", file_info['table_chunks'])
                    col4.metric("Total Chunks", file_info['total_chunks'])

        st.subheader("Chat with your Documents")

        for i, chat_item in enumerate(st.session_state.chat_history):
            question = chat_item['question']
            answer = chat_item['answer']
            sub_queries = chat_item.get('sub_queries', [])
            total_chunks = chat_item.get('total_chunks_collected', 0)
            reranked_chunks = chat_item.get('reranked_chunks_used', 0)

            with st.container():
                st.markdown(f"**🔴 Q{i+1}:** {question}")
                if sub_queries:
                    with st.expander("🧩 Sub-queries used"):
                        for j, sub_q in enumerate(sub_queries, 1):
                            st.write(f"{j}. {sub_q}")
                st.markdown(f"**A{i+1}:** {answer}")

                # 🔽 Render relevant images
                relevant_images = chat_item.get('relevant_images', [])
                if relevant_images:
                    with st.expander("🖼️ Relevant Images"):
                        for img in relevant_images:
                            image_path = img.get("image_path", "")
                            caption = img.get("content", "")
                            page = img.get("page_number", "N/A")
                            if os.path.exists(image_path):
                                st.image(image_path, caption=f"📄 Page {page}: {caption}", use_column_width=True)

                st.caption(f"📊 Chunks collected: {total_chunks} | Final reranked chunks: {reranked_chunks}")
                st.divider()

        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings and how do they compare to previous research?",
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

        st.subheader("Example Questions You Can Ask:")
        example_questions = [
            "What are the main findings and how do they compare to previous research?",
            "Summarize the methodology and discuss the results from all tables.",
            "What do the images show and how do they relate to the textual content?",
            "Compare the different approaches mentioned and evaluate their effectiveness."
        ]
        for question in example_questions:
            st.markdown(f"• {question}")

def process_pdfs(uploaded_files):
    with st.spinner("Processing PDFs..."):
        try:
            gemini_client = GeminiClient(GOOGLE_API_KEY)
            vector_store = VectorStore()
            pdf_processor = PDFProcessor(gemini_client, vector_store)

            temp_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_paths.append(tmp_file.name)

            results = pdf_processor.process_multiple_pdfs(temp_paths)
            pdf_processor.build_vector_index()

            st.session_state.pdf_processor = pdf_processor
            st.session_state.processed_files = []

            for i, (file_name, result) in enumerate(results.items()):
                if 'error' not in result:
                    st.session_state.processed_files.append({
                        'name': uploaded_files[i].name,
                        **result
                    })

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
    with st.spinner("Processing your question..."):
        try:
            result = st.session_state.pdf_processor.query(question)

            answer = result.get('answer', 'No answer generated')

            chat_entry = {
                'question': question,
                'answer': answer,
                'method': 'complex',
                'sub_queries': result.get('sub_queries', []),
                'total_chunks_collected': result.get('total_chunks_collected', 0),
                'reranked_chunks_used': result.get('reranked_chunks_used', 0),
                'relevant_images': result.get('relevant_image_chunks', [])
            }

            st.session_state.chat_history.append(chat_entry)
            st.rerun()

        except Exception as e:
            st.error(f"Error processing question: {e}")

def run_evaluation_mode(evaluation_file):
    if st.session_state.pdf_processor:
        document_name = "transformer_comparison"
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
