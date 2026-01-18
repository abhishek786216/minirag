import openai
from typing import List, Dict
import logging
import asyncio
from functools import lru_cache
import numpy as np
import os
from .simple_embeddings import SimpleEmbeddingService

logger = logging.getLogger(__name__)

# We'll import sentence-transformers only when we actually need it
SENTENCE_TRANSFORMERS_AVAILABLE = False
SentenceTransformer = None

def _try_import_sentence_transformers():
    global SENTENCE_TRANSFORMERS_AVAILABLE, SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        logger.info("sentence-transformers successfully imported")
    except Exception as e:
        logger.warning(f"sentence-transformers not available: {e}")
        SENTENCE_TRANSFORMERS_AVAILABLE = False

class EmbeddingService:
    """Service for generating text embeddings using OpenAI, local models, or simple fallback"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        self.model = model
        self.use_local = False
        self.local_model = None
        self.simple_service = None
        
        # Try to use OpenAI first
        if api_key and api_key.strip():
            try:
                self.client = openai.AsyncOpenAI(api_key=api_key)
                self.dimension = 1536  # OpenAI ada-002 dimension
                logger.info("Using OpenAI embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self._setup_fallback()
        else:
            logger.info("No OpenAI API key provided, using fallback embeddings")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback embedding service"""
        # Try sentence-transformers first
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            _try_import_sentence_transformers()
            
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Attempting to use sentence-transformers for embeddings...")
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = 384
                self.use_local = True
                logger.info("Successfully initialized sentence-transformers")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize sentence-transformers: {e}")
        
        # Fallback to simple hash-based embeddings
        logger.info("Using simple hash-based embeddings as fallback")
        self.simple_service = SimpleEmbeddingService(dimension=1536)  # Match Pinecone index
        self.dimension = 1536  # Keep consistent with Pinecone index
        self.use_local = True
        
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            # Clean text
            text = text.strip().replace("\n", " ")
            if not text:
                raise ValueError("Empty text provided")
            
            if self.use_local or not hasattr(self, 'client'):
                # Use fallback embedding services
                if self.simple_service:
                    return await self.simple_service.generate_embedding(text)
                elif self.local_model:
                    embedding = self.local_model.encode(text, convert_to_numpy=True)
                    embedding_list = embedding.tolist()
                    logger.debug(f"Generated local embedding for text of length {len(text)}")
                    return embedding_list
                else:
                    raise ValueError("No embedding service available")
            else:
                # Try OpenAI API
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=text
                    )
                    
                    embedding = response.data[0].embedding
                    logger.debug(f"Generated OpenAI embedding for text of length {len(text)}")
                    return embedding
                    
                except Exception as api_error:
                    if "quota" in str(api_error).lower() or "insufficient_quota" in str(api_error).lower():
                        logger.warning("OpenAI quota exceeded, falling back to local embeddings")
                        # Set up and use fallback
                        self._setup_fallback()
                        return await self.generate_embedding(text)
                    else:
                        raise api_error
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        if self.use_local or not hasattr(self, 'client'):
            # Use fallback embedding services
            try:
                if self.simple_service:
                    return await self.simple_service.generate_embeddings_batch(texts)
                elif self.local_model:
                    batch_embeddings = self.local_model.encode(texts, convert_to_numpy=True)
                    return [embedding.tolist() for embedding in batch_embeddings]
                else:
                    raise ValueError("No embedding service available")
            except Exception as e:
                logger.error(f"Error in local batch embedding: {str(e)}")
                # Fallback to individual processing
                embeddings = []
                for text in texts:
                    embedding = await self.generate_embedding(text)
                    embeddings.append(embedding)
                return embeddings
        
        embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean texts
            cleaned_batch = [text.strip().replace("\n", " ") for text in batch]
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=cleaned_batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated {len(batch_embeddings)} embeddings in batch {i//batch_size + 1}")
                
                # Small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                raise
        
        logger.info(f"Generated total {len(embeddings)} embeddings")
        return embeddings
    
    async def generate_chunk_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = await self.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        enriched_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            enriched_chunk = chunk.copy()
            enriched_chunk['embedding'] = embedding
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    @lru_cache(maxsize=1000)
    def cached_embedding(self, text: str) -> List[float]:
        """
        Cached version of embedding generation for repeated queries
        Note: This is a synchronous cache, actual embedding generation is async
        """
        # This is just a placeholder for the cache structure
        # Actual implementation would need async caching
        pass