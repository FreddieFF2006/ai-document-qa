import streamlit as st
from pdf_ai_agent import EmbeddingManager, ClaudeAIAgent
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="AI Document Q&A", page_icon="🤖", layout="wide")

def smart_chunk_text(text, min_chunk_size=1500, max_chunk_size=2500):
    """Smart chunking that preserves paragraph structure"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        potential_size = len(current_chunk) + len(para) + 2
        
        if potential_size <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = para
            elif current_chunk:
                if len(para) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    sentences = re.split(r'([.!?]+\s+)', para)
                    temp_chunk = ""
                    for i in range(0, len(sentences), 2):
                        sentence = sentences[i]
                        if i + 1 < len(sentences):
                            sentence += sentences[i + 1]
                        if len(temp_chunk) + len(sentence) <= max_chunk_size:
                            temp_chunk += sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    current_chunk = temp_chunk
                else:
                    chunks.append(current_chunk)
                    current_chunk = para
            else:
                current_chunk = para
    
    if current_chunk and len(current_chunk.strip()) > 100:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_text_from_pdf(file):
    """Extract text from PDF"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    """Extract text from Word document"""
    doc = DocxDocument(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_pptx(file):
    """Extract text from PowerPoint"""
    try:
        from pptx import Presentation
        prs = Presentation(file)
        all_text = ""
        for i, slide in enumerate(prs.slides, 1):
            all_text += f"\n=== Slide {i} ===\n\n"
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    all_text += shape.text + "\n"
            all_text += "\n"
        return all_text
    except:
        return ""

def extract_text_from_excel(file):
    """Extract text from Excel"""
    try:
        import pandas as pd
        excel_file = pd.ExcelFile(file)
        all_text = ""
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet_name)
            all_text += f"\n=== Sheet: {sheet_name} ===\n\n"
            all_text += df.to_string(index=False) + "\n\n"
        return all_text
    except ImportError:
        return ""

def extract_text_from_txt(file):
    """Extract text from text file"""
    return file.read().decode('utf-8')

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return chunks"""
    if not st.session_state.embedding_manager:
        st.session_state.embedding_manager = EmbeddingManager()
    
    all_chunks = []
    processed_files = []
    total_chars = 0
    
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_ext == 'pdf':
                text = extract_text_from_pdf(uploaded_file)
            elif file_ext in ['docx', 'doc']:
                text = extract_text_from_docx(uploaded_file)
            elif file_ext in ['pptx', 'ppt']:
                text = extract_text_from_pptx(uploaded_file)
            elif file_ext == 'txt':
                text = extract_text_from_txt(uploaded_file)
            else:
                continue
            
            if text and len(text.strip()) > 0:
                total_chars += len(text)
                chunks = smart_chunk_text(text)
                chunks_with_source = [f"[Source: {uploaded_file.name}]\n\n{chunk}" for chunk in chunks]
                all_chunks.extend(chunks_with_source)
                processed_files.append(uploaded_file.name)
        
        except Exception as e:
            st.warning(f"⚠️ Could not process {uploaded_file.name}: {str(e)}")
    
    if all_chunks:
        # Add to existing chunks or create new index
        st.session_state.chunks.extend(all_chunks)
        st.session_state.embedding_manager.build_index(st.session_state.chunks)
        st.session_state.processed_files.extend(processed_files)
        
        return len(processed_files), len(all_chunks), total_chars
    
    return 0, 0, 0

# Check for API key
api_key = os.getenv('ANTHROPIC_API_KEY')

if not api_key:
    st.error("❌ API Key not found!")
    st.info("Please make sure your .env file contains: ANTHROPIC_API_KEY=your-key-here")
    st.stop()

# Initialize session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'claude_agent' not in st.session_state:
    try:
        st.session_state.claude_agent = ClaudeAIAgent(api_key)
    except Exception as e:
        st.error(f"Error initializing Claude: {e}")
        st.stop()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🤖 AI Document Q&A Assistant")
st.markdown("Upload your documents and ask questions!")

# Sidebar - Show loaded documents only
with st.sidebar:
    st.header("📚 Loaded Documents")
    
    if st.session_state.processed_files:
        for i, file in enumerate(st.session_state.processed_files, 1):
            st.text(f"{i}. {file}")
        st.info(f"Total chunks: {len(st.session_state.chunks)}")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents"):
            st.session_state.chunks = []
            st.session_state.embedding_manager = None
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No documents loaded yet")
    
    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("💬 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Main chat interface
st.header("💬 Chat")

# File uploader above chat
uploaded_files = st.file_uploader(
    "📎 Upload documents to add to knowledge base",
    type=['pdf', 'docx', 'doc', 'txt', 'pptx', 'ppt'],
    accept_multiple_files=True,
    help="Upload PDF, Word, PowerPoint, or text files - they will be processed automatically",
    key="main_uploader"
)

# Auto-process when files are uploaded
if uploaded_files:
    # Check if these are new files (not already processed)
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        with st.spinner(f"Processing {len(new_files)} file(s)..."):
            num_files, num_chunks, total_chars = process_uploaded_files(new_files)
            
            if num_files > 0:
                st.success(f"✅ Processed {num_files} file(s): {num_chunks} chunks, {total_chars:,} characters")
            else:
                st.error("Could not extract text from the uploaded files")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if not st.session_state.processed_files:
    st.info("👆 Upload documents above to get started!")

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.embedding_manager:
        st.error("Please upload documents first!")
    else:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Search for relevant chunks
                    results = st.session_state.embedding_manager.search(prompt, top_k=12)
                    chunks = [c for c, _ in results]
                    
                    # Get answer from Claude
                    answer = st.session_state.claude_agent.ask(prompt, chunks, max_tokens=3000)
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})