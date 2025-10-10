import os
import anthropic
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# File conversion imports
from PyPDF2 import PdfReader, PdfWriter
from docx import Document
from PIL import Image
import io

load_dotenv()


class FileStandardizer:
    """Converts various file formats to PDF"""
    
    @staticmethod
    def convert_to_pdf(file_path: str, output_path: str = None) -> str:
        """
        Convert various file formats to PDF
        
        Args:
            file_path: Path to input file
            output_path: Path for output PDF (optional)
            
        Returns:
            Path to the converted PDF file
        """
        file_ext = Path(file_path).suffix.lower()
        
        if output_path is None:
            output_path = str(Path(file_path).with_suffix('.pdf'))
        
        # If already PDF, just return the path
        if file_ext == '.pdf':
            return file_path
        
        # Handle DOCX files
        elif file_ext in ['.docx', '.doc']:
            return FileStandardizer._docx_to_pdf(file_path, output_path)
        
        # Handle text files
        elif file_ext == '.txt':
            return FileStandardizer._txt_to_pdf(file_path, output_path)
        
        # Handle image files
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return FileStandardizer._image_to_pdf(file_path, output_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def _docx_to_pdf(docx_path: str, pdf_path: str) -> str:
        """Convert DOCX to PDF (using text extraction method)"""
        doc = Document(docx_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        
        # For simplicity, we'll save as text-based PDF using reportlab
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            
            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            
            # Simple text wrapping
            y = height - inch
            for line in text.split('\n'):
                if y < inch:
                    c.showPage()
                    y = height - inch
                c.drawString(inch, y, line[:80])  # Simple truncation
                y -= 15
            
            c.save()
            return pdf_path
        except ImportError:
            # If reportlab not available, just extract text
            print("Warning: reportlab not installed. Creating text file instead.")
            txt_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return txt_path
    
    @staticmethod
    def _txt_to_pdf(txt_path: str, pdf_path: str) -> str:
        """Convert TXT to PDF"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            
            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            
            y = height - inch
            for line in text.split('\n'):
                if y < inch:
                    c.showPage()
                    y = height - inch
                c.drawString(inch, y, line[:80])
                y -= 15
            
            c.save()
            return pdf_path
        except ImportError:
            return txt_path
    
    @staticmethod
    def _image_to_pdf(image_path: str, pdf_path: str) -> str:
        """Convert image to PDF"""
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(pdf_path, 'PDF', resolution=100.0)
        return pdf_path


class PDFChunker:
    """Splits PDF into text chunks"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > 0:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.overlap
        
        return [c for c in chunks if c]  # Remove empty chunks


class EmbeddingManager:
    """Manages text embeddings and vector search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of sentence-transformer model
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks: List[str]):
        """Build FAISS index from chunks"""
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most relevant chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk_text, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.chunks[idx], float(dist)))
        
        return results


class ClaudeAIAgent:
    """Integrates with Claude AI for Q&A"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Claude AI client
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY or pass api_key parameter")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def ask(self, question: str, context: List[str], max_tokens: int = 1024) -> str:
        """
        Ask Claude a question with context
        
        Args:
            question: User's question
            context: List of relevant text chunks
            max_tokens: Maximum tokens in response
            
        Returns:
            Claude's response
        """
        # Build context string
        context_str = "\n\n".join([f"[Context {i+1}]:\n{chunk}" 
                                   for i, chunk in enumerate(context)])
        
        # Create prompt
        prompt = f"""Based on the following context from documents, please answer the question.

Context:
{context_str}

Question: {question}

Please provide a clear and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""
        
        # Call Claude API
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text


class PDFAIAgent:
    """Main AI agent that orchestrates the entire pipeline"""
    
    def __init__(self, api_key: str = None, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the complete AI agent
        
        Args:
            api_key: Anthropic API key
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        self.file_standardizer = FileStandardizer()
        self.chunker = PDFChunker(chunk_size, overlap)
        self.embedding_manager = EmbeddingManager()
        self.claude_agent = ClaudeAIAgent(api_key)
        self.processed_files = []
    
    def process_file(self, file_path: str) -> str:
        """
        Process a file: convert to PDF, extract text, and create embeddings
        
        Args:
            file_path: Path to input file
            
        Returns:
            Path to converted PDF
        """
        print(f"\n{'='*60}")
        print(f"Processing file: {file_path}")
        print(f"{'='*60}")
        
        # Step 1: Convert to PDF
        print("\n[Step 1] Converting to PDF...")
        pdf_path = self.file_standardizer.convert_to_pdf(file_path)
        print(f"✓ PDF created: {pdf_path}")
        
        # Step 2: Extract text and chunk
        print("\n[Step 2] Extracting and chunking text...")
        text = self.chunker.extract_text_from_pdf(pdf_path)
        chunks = self.chunker.chunk_text(text)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings and index
        print("\n[Step 3] Creating embeddings and building index...")
        self.embedding_manager.build_index(chunks)
        print("✓ Index built successfully")
        
        self.processed_files.append(pdf_path)
        
        return pdf_path
    
    def ask_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Ask a question about the processed documents
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Search for relevant chunks
        print("\n[Step 1] Searching for relevant context...")
        results = self.embedding_manager.search(question, top_k)
        
        relevant_chunks = [chunk for chunk, _ in results]
        print(f"✓ Found {len(relevant_chunks)} relevant chunks")
        
        # Ask Claude
        print("\n[Step 2] Asking Claude AI...")
        answer = self.claude_agent.ask(question, relevant_chunks)
        print("✓ Response received")
        
        return {
            'question': question,
            'answer': answer,
            'context_chunks': relevant_chunks,
            'relevance_scores': [dist for _, dist in results]
        }
    
    def interactive_mode(self):
        """Run in interactive Q&A mode"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE - Type 'quit' to exit")
        print("="*60)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                result = self.ask_question(question)
                print(f"\n💡 Answer:\n{result['answer']}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")


# Example usage function
def main():
    """Example usage of the PDF AI Agent"""
    
    # Initialize the agent
    print("Initializing PDF AI Agent...")
    agent = PDFAIAgent()
    
    # Example: Process a file
    # agent.process_file('path/to/your/document.pdf')
    # agent.process_file('path/to/your/document.docx')
    
    # Example: Ask questions
    # result = agent.ask_question("What is the main topic of this document?")
    # print(f"Answer: {result['answer']}")
    
    # Example: Interactive mode
    # agent.interactive_mode()
    
    print("\n" + "="*60)
    print("PDF AI Agent initialized successfully!")
    print("="*60)
    print("\nUsage:")
    print("1. agent.process_file('your_file.pdf')  # Process a document")
    print("2. agent.ask_question('your question')   # Ask questions")
    print("3. agent.interactive_mode()              # Start interactive Q&A")
    print("="*60)


if __name__ == "__main__":
    main()
