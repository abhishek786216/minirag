import hashlib
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class SimpleEmbeddingService:
    """Simple hash-based embedding service as fallback when no API is available"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension  # Match Pinecone index dimension
        
    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to a simple hash-based vector"""
        # Clean text
        text = text.strip().lower().replace("\n", " ")
        
        # Create multiple hash values for better distribution
        hashes = []
        hash_groups = self.dimension // 32  # Each MD5 hash gives us 32 hex chars = 32 values
        
        for i in range(hash_groups):
            hash_input = f"{text}_{i}".encode('utf-8')
            hash_obj = hashlib.md5(hash_input)
            hex_digest = hash_obj.hexdigest()
            
            # Convert each pair of hex chars to a float value
            for j in range(0, len(hex_digest), 2):
                if len(hashes) >= self.dimension:
                    break
                hex_pair = hex_digest[j:j+2]
                hash_int = int(hex_pair, 16)
                # Normalize to [-1, 1]
                normalized = (hash_int / 255.0) * 2 - 1
                hashes.append(normalized)
        
        # Fill remaining dimensions if needed
        while len(hashes) < self.dimension:
            # Use SHA256 for additional randomness if needed
            additional_hash = hashlib.sha256(f"{text}_{len(hashes)}".encode('utf-8'))
            hash_int = int(additional_hash.hexdigest()[:2], 16)
            normalized = (hash_int / 255.0) * 2 - 1
            hashes.append(normalized)
        
        # Normalize the vector
        vector = np.array(hashes[:self.dimension])
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self._text_to_vector(text)
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self._text_to_vector(text) for text in texts]