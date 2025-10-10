from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from anthropic import Anthropic

class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the embedding model"""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def build_index(self, text_chunks):
        """Build FAISS index from text chunks"""
        self.chunks = text_chunks
        embeddings = self.model.encode(text_chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query, top_k=5):
        """Search for most relevant chunks"""
        if self.index is None:
            return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
        
        return results

class ClaudeAIAgent:
    def __init__(self, api_key):
        """Initialize Claude AI client"""
        self.client = Anthropic(api_key=api_key)
    
    def ask(self, question, context_chunks, max_tokens=2000, conversation_history=None):
        """
        Ask a question with document context and conversation memory
        
        Args:
            question: The user's question
            context_chunks: List of relevant document chunks
            max_tokens: Maximum tokens for response
            conversation_history: List of previous messages for context
        """
        # Build context from chunks
        context = "\n\n---\n\n".join(context_chunks)
        
        # Build messages array with conversation history
        messages = []
        
        # Add conversation history if provided (last 10 messages to keep context manageable)
        if conversation_history and len(conversation_history) > 0:
            # Take last 10 messages (5 user + 5 assistant exchanges)
            recent_history = conversation_history[-10:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current question with document context
        current_prompt = f"""Based on the following document excerpts and our conversation history, please answer the question.

Document Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the document context. If referring to previous parts of our conversation, make those connections clear."""
        
        messages.append({
            "role": "user",
            "content": current_prompt
        })
        
        # Call Claude API
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=messages
        )
        
        return response.content[0].text