from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional, Tuple
import logging
import asyncio
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    """Pinecone vector database operations"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        
        # Initialize Pinecone with new API
        self.pc = Pinecone(api_key=api_key)
        
    async def initialize_index(self, dimension: int = 1536, metric: str = "cosine"):
        """Initialize Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                logger.info(f"Created new index: {self.index_name}")
                
                # Wait for index to be ready
                await asyncio.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}")
            raise
    
    async def upsert_chunks(self, chunks: List[Dict], batch_size: int = 100) -> bool:
        """
        Upsert chunks to vector database
        
        Args:
            chunks: List of chunks with text, embedding, and metadata
            batch_size: Number of chunks to upsert per batch
            
        Returns:
            Success status
        """
        if not self.index:
            await self.initialize_index()
        
        if not chunks:
            return True
        
        try:
            total_upserted = 0
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare vectors for upsert
                vectors = []
                for chunk in batch:
                    vector_id = str(uuid.uuid4())
                    
                    # Prepare metadata
                    metadata = chunk['metadata'].copy()
                    metadata.update({
                        'text': chunk['text'][:1000],  # Limit text in metadata
                        'created_at': datetime.now().isoformat(),
                        'vector_id': vector_id
                    })
                    
                    vector = {
                        'id': vector_id,
                        'values': chunk['embedding'],
                        'metadata': metadata
                    }
                    vectors.append(vector)
                
                # Upsert batch
                upsert_response = self.index.upsert(vectors=vectors)
                upserted_count = upsert_response.get('upserted_count', len(vectors))
                total_upserted += upserted_count
                
                logger.info(f"Upserted batch {i//batch_size + 1}: {upserted_count} vectors")
                
                # Small delay between batches
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.5)
            
            logger.info(f"Successfully upserted {total_upserted} vectors total")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting chunks: {str(e)}")
            raise
    
    async def query(self, query_embedding: List[float], top_k: int = 10, 
                   filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Query vector database for similar chunks
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching chunks with scores
        """
        if not self.index:
            await self.initialize_index()
        
        try:
            # Query index
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_values=False,
                include_metadata=True,
                filter=filter_dict
            )
            
            results = []
            for match in query_response.get('matches', []):
                result = {
                    'id': match['id'],
                    'score': float(match['score']),
                    'text': match['metadata'].get('text', ''),
                    'metadata': match['metadata']
                }
                results.append(result)
            
            logger.info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying index: {str(e)}")
            raise
    
    async def delete_by_filter(self, filter_dict: Dict) -> bool:
        """
        Delete vectors by metadata filter
        
        Args:
            filter_dict: Metadata filter for deletion
            
        Returns:
            Success status
        """
        if not self.index:
            await self.initialize_index()
        
        try:
            delete_response = self.index.delete(filter=filter_dict)
            logger.info(f"Deleted vectors with filter: {filter_dict}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise
    
    async def get_index_stats(self) -> Dict:
        """
        Get index statistics
        
        Returns:
            Index statistics dictionary
        """
        if not self.index:
            await self.initialize_index()
        
        try:
            stats = self.index.describe_index_stats()
            # Extract only serializable data to avoid JSON encoding issues
            return {
                'total_vector_count': int(stats.get('total_vector_count', 0)) if stats.get('total_vector_count') is not None else 0,
                'dimension': int(stats.get('dimension', 0)) if stats.get('dimension') is not None else 0,
                'index_fullness': float(stats.get('index_fullness', 0)) if stats.get('index_fullness') is not None else 0.0,
                'namespaces': dict(stats.get('namespaces', {})) if stats.get('namespaces') else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {
                'total_vector_count': 0,
                'dimension': 0,
                'index_fullness': 0.0,
                'namespaces': {}
            }
    
    async def add_text_direct(self, text: str, metadata: Dict, embedding: List[float]) -> str:
        """
        Add a single text directly to the vector store
        
        Args:
            text: Text content
            metadata: Associated metadata
            embedding: Pre-computed embedding
            
        Returns:
            Vector ID
        """
        if not self.index:
            await self.initialize_index()
        
        try:
            vector_id = str(uuid.uuid4())
            
            # Prepare metadata
            full_metadata = metadata.copy()
            full_metadata.update({
                'text': text[:1000],  # Limit text in metadata
                'created_at': datetime.now().isoformat(),
                'vector_id': vector_id
            })
            
            # Upsert single vector
            self.index.upsert(vectors=[{
                'id': vector_id,
                'values': embedding,
                'metadata': full_metadata
            }])
            
            logger.info(f"Added single text with vector ID: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Error adding single text: {str(e)}")
            raise