from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import anthropic

class EmbeddingManager:
    def __init__(self):
        """Initialize E5-Large-v2 embeddings and Qdrant"""
        print("Loading E5-Large-v2 model...")
        self.model = SentenceTransformer('intfloat/e5-large-v2')
        self.client = QdrantClient(":memory:")
        self.collection_name = "documents"
        self.chunks = []
        
    def build_index(self, chunks):
        """Build Qdrant index from text chunks"""
        self.chunks = chunks
        
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        # Create collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embeddings.shape[1],
                distance=Distance.COSINE
            )
        )
        
        # Upload vectors
        points = [
            PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={"text": chunk}
            )
            for idx, chunk in enumerate(chunks)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Index built with {len(chunks)} chunks")
    
    def search(self, query, top_k=25):
        """Search for most relevant chunks"""
        query_vector = self.model.encode([query])[0]
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        return [(hit.payload["text"], hit.score) for hit in results]

class ClaudeAIAgent:
    def __init__(self, api_key):
        """Initialize Claude Opus 4 client"""
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def ask(self, question, chunks, max_tokens=4000, conversation_history=None):
        """Ask a question with document context and conversation history"""
        
        # Build conversation with memory
        messages = []
        
        # Add previous conversation history (excluding current question)
        if conversation_history:
            # Get all messages except the last one (which is the current question)
            for msg in conversation_history[:-1]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Create the context from chunks
        context = "\n\n---\n\n".join(chunks)
        
        # Add current question with context
        current_prompt = f"""Based on the following document excerpts, please answer the question. If the answer cannot be found in the excerpts, say so.

Document excerpts:
{context}

Question: {question}

Please provide a detailed answer based on the document excerpts above."""
        
        messages.append({
            "role": "user",
            "content": current_prompt
        })
        
        # Call Claude Opus 4 API with conversation history
        message = self.client.messages.create(
            model="claude-opus-4-20250514",  # ✅ CLAUDE OPUS 4
            max_tokens=max_tokens,
            messages=messages
        )
        
        return message.content[0].text