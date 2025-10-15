from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from anthropic import Anthropic
import uuid

class EmbeddingManager:
    def __init__(self, model_name='intfloat/e5-large-v2'):
        """Initialize the E5-Large-v2 embedding model with Qdrant"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize Qdrant client (in-memory mode)
        self.client = QdrantClient(":memory:")
        self.collection_name = f"documents_{uuid.uuid4().hex[:8]}"
        self.chunks = []
        self.dimension = 1024  # E5-Large-v2 produces 1024-dimensional embeddings
    
    def build_index(self, text_chunks):
        """Build Qdrant vector database from text chunks"""
        self.chunks = text_chunks
        
        print(f"Creating embeddings for {len(text_chunks)} chunks...")
        
        # E5 models require "query: " prefix for queries and "passage: " for documents
        passages = [f"passage: {chunk}" for chunk in text_chunks]
        embeddings = self.model.encode(passages, show_progress_bar=True)
        
        # Create collection with cosine distance
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE  # Using cosine similarity
            )
        )
        
        # Upload vectors to Qdrant
        points = [
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"text": chunk, "chunk_id": idx}
            )
            for idx, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Indexed {len(text_chunks)} chunks in Qdrant with cosine distance")
    
    def search(self, query, top_k=5):
        """Search for most relevant chunks using cosine similarity"""
        if not self.chunks:
            return []
        
        # E5 models require "query: " prefix for search queries
        query_with_prefix = f"query: {query}"
        query_embedding = self.model.encode([query_with_prefix])[0]
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        # Return chunks with similarity scores (higher is better with cosine)
        results = []
        for hit in search_results:
            chunk_text = hit.payload["text"]
            similarity_score = hit.score  # Cosine similarity (0-1, higher is better)
            results.append((chunk_text, similarity_score))
        
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