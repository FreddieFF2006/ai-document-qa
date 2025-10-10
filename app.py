import streamlit as st
from pdf_ai_agent import EmbeddingManager, ClaudeAIAgent
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import re
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="AI Document Q&A", page_icon="🤖", layout="wide")

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Center the icons in buttons */
    .stButton button {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Make chat buttons take full width */
    div[data-testid="column"] button {
        width: 100%;
    }
    
    /* Style for attach button to look integrated */
    .attach-btn button {
        background-color: transparent;
        border: none;
        padding: 0.5rem;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

def process_uploaded_files(uploaded_files, chat_id):
    """Process uploaded files and return chunks"""
    current_chat = st.session_state.chats[chat_id]
    
    if not current_chat['embedding_manager']:
        current_chat['embedding_manager'] = EmbeddingManager()
    
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
        current_chat['chunks'].extend(all_chunks)
        current_chat['embedding_manager'].build_index(current_chat['chunks'])
        current_chat['processed_files'].extend(processed_files)
        
        return len(processed_files), len(all_chunks), total_chars
    
    return 0, 0, 0

def generate_chat_title(first_message):
    """Generate a short title from the first message"""
    title = first_message[:40]
    if len(first_message) > 40:
        title += "..."
    return title

def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        'title': 'New Chat',
        'messages': [],
        'chunks': [],
        'embedding_manager': None,
        'processed_files': [],
        'created_at': datetime.now()
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

# Check for API key
api_key = os.getenv('ANTHROPIC_API_KEY')

if not api_key:
    st.error("❌ API Key not found!")
    st.info("Please make sure your .env file contains: ANTHROPIC_API_KEY=your-key-here")
    st.stop()

# Initialize Claude agent (shared across all chats)
if 'claude_agent' not in st.session_state:
    try:
        st.session_state.claude_agent = ClaudeAIAgent(api_key)
    except Exception as e:
        st.error(f"Error initializing Claude: {e}")
        st.stop()

# Initialize chats dictionary
if 'chats' not in st.session_state:
    st.session_state.chats = {}
    create_new_chat()

# Initialize current chat ID
if 'current_chat_id' not in st.session_state:
    if st.session_state.chats:
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
    else:
        create_new_chat()

# Header
st.title("🤖 AI Document Q&A Assistant")
st.markdown("Upload your documents and ask questions!")

# Sidebar - Chat management
with st.sidebar:
    st.header("💬 Chats")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # List all chats
    chat_items = sorted(st.session_state.chats.items(), 
                       key=lambda x: x[1]['created_at'], 
                       reverse=True)
    
    for chat_id, chat_data in chat_items:
        col1, col2 = st.columns([5, 1])
        
        with col1:
            # Chat button
            is_current = chat_id == st.session_state.current_chat_id
            button_type = "primary" if is_current else "secondary"
            icon = "📍" if is_current else "💬"
            
            if st.button(
                f"{icon} {chat_data['title']}", 
                key=f"chat_{chat_id}",
                use_container_width=True,
                type=button_type
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        with col2:
            # Delete button - centered
            if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
                if len(st.session_state.chats) > 1:
                    del st.session_state.chats[chat_id]
                    # Switch to another chat
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                    st.rerun()
                else:
                    st.warning("Cannot delete the last chat!")

# Get current chat
current_chat = st.session_state.chats[st.session_state.current_chat_id]

# Main chat interface
st.header(f"💬 {current_chat['title']}")

# Show document count for current chat
if current_chat['processed_files']:
    st.caption(f"📚 {len(current_chat['processed_files'])} document(s) loaded | {len(current_chat['chunks'])} chunks")

# Display chat history
for message in current_chat['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input area with attach button on the left (like Claude/ChatGPT)
col1, col2 = st.columns([0.7, 9.3])

with col1:
    # File uploader styled as a button
    uploaded_files = st.file_uploader(
        "📎",
        type=['pdf', 'docx', 'doc', 'txt', 'pptx', 'ppt'],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.current_chat_id}",
        label_visibility="collapsed"
    )

with col2:
    # Chat input
    prompt = st.chat_input("Ask me anything...")

# Auto-process when files are uploaded
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in current_chat['processed_files']]
    
    if new_files:
        with st.spinner(f"Processing {len(new_files)} file(s)..."):
            num_files, num_chunks, total_chars = process_uploaded_files(new_files, st.session_state.current_chat_id)
            
            if num_files > 0:
                st.success(f"✅ Processed {num_files} file(s): {num_chunks} chunks, {total_chars:,} characters")
            else:
                st.error("Could not extract text from the uploaded files")

# Handle chat input
if prompt:
    # Update chat title if this is the first message
    if len(current_chat['messages']) == 0:
        current_chat['title'] = generate_chat_title(prompt)
    
    # Add user message to chat
    current_chat['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if we have documents
                if current_chat['embedding_manager'] and current_chat['chunks']:
                    # Search for relevant chunks in current chat
                    results = current_chat['embedding_manager'].search(prompt, top_k=12)
                    chunks = [c for c, _ in results]
                    
                    # Get answer from Claude with document context
                    answer = st.session_state.claude_agent.ask(prompt, chunks, max_tokens=3000)
                else:
                    # No documents - just chat with Claude directly
                    message = st.session_state.claude_agent.client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=3000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = message.content[0].text
                
                st.markdown(answer)
                current_chat['messages'].append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                current_chat['messages'].append({"role": "assistant", "content": error_msg})