import cohere
from typing import List, Dict, Optional
import logging
from services.vector_store import VectorStore
from services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class RetrieverService:
    """Document retrieval and reranking service"""
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService, 
                 cohere_api_key: str, top_k: int = 10, rerank_top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.cohere_client = cohere.Client(cohere_api_key)
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
    
    async def retrieve_and_rerank(self, question: str, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant documents and rerank them
        
        Args:
            question: User question
            filter_dict: Optional metadata filters
            
        Returns:
            List of reranked documents with scores and citations
        """
        try:
            # Generate embedding for question
            question_embedding = await self.embedding_service.generate_embedding(question)
            
            # Retrieve initial candidates
            initial_results = await self.vector_store.query(
                query_embedding=question_embedding,
                top_k=self.top_k,
                filter_dict=filter_dict
            )
            
            if not initial_results:
                logger.warning("No initial results found for query")
                return []
            
            # Prepare documents for reranking
            documents = []
            result_map = {}
            
            for i, result in enumerate(initial_results):
                doc_id = f"doc_{i}"
                documents.append(result['text'])
                result_map[doc_id] = result
            
            # Rerank using Cohere
            rerank_results = self.cohere_client.rerank(
                model="rerank-english-v2.0",
                query=question,
                documents=documents,
                top_k=min(self.rerank_top_k, len(documents))
            )
            
            # Process reranked results
            final_results = []
            for rerank_result in rerank_results.results:
                original_result = result_map[f"doc_{rerank_result.index}"]
                
                reranked_doc = {
                    'id': original_result['id'],
                    'text': original_result['text'],
                    'metadata': original_result['metadata'],
                    'vector_score': original_result['score'],
                    'rerank_score': rerank_result.relevance_score,
                    'citation_id': len(final_results) + 1
                }
                
                final_results.append(reranked_doc)
            
            logger.info(f"Retrieved and reranked {len(final_results)} documents for question: {question[:100]}...")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in retrieve and rerank: {str(e)}")
            # Fallback to just vector search if reranking fails
            try:
                question_embedding = await self.embedding_service.generate_embedding(question)
                results = await self.vector_store.query(
                    query_embedding=question_embedding,
                    top_k=self.rerank_top_k,
                    filter_dict=filter_dict
                )
                
                # Add citation IDs
                for i, result in enumerate(results):
                    result['citation_id'] = i + 1
                    result['rerank_score'] = result['score']  # Use vector score as fallback
                
                return results
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {str(fallback_error)}")
                raise
    
    async def retrieve_only(self, question: str, top_k: Optional[int] = None, 
                           filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve documents without reranking
        
        Args:
            question: User question
            top_k: Number of results (uses default if None)
            filter_dict: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        try:
            k = top_k if top_k is not None else self.top_k
            
            # Generate embedding for question
            question_embedding = await self.embedding_service.generate_embedding(question)
            
            # Retrieve documents
            results = await self.vector_store.query(
                query_embedding=question_embedding,
                top_k=k,
                filter_dict=filter_dict
            )
            
            # Add citation IDs
            for i, result in enumerate(results):
                result['citation_id'] = i + 1
            
            logger.info(f"Retrieved {len(results)} documents for question: {question[:100]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            raise
    
    def format_context_for_llm(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents as context for LLM
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for doc in retrieved_docs:
            citation_id = doc.get('citation_id', 'Unknown')
            text = doc['text']
            metadata = doc.get('metadata', {})
            
            # Add source info if available
            source_info = []
            if metadata.get('filename'):
                source_info.append(f"File: {metadata['filename']}")
            if metadata.get('chunk_id') is not None:
                source_info.append(f"Section: {metadata['chunk_id']}")
            
            source_str = f" ({', '.join(source_info)})" if source_info else ""
            
            context_part = f"[{citation_id}]{source_str}: {text}"
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        logger.debug(f"Formatted context with {len(retrieved_docs)} documents")
        return context
    
    def extract_citations(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Extract citation information from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        for doc in retrieved_docs:
            citation = {
                'id': doc.get('citation_id', 'Unknown'),
                'text_preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'metadata': doc.get('metadata', {}),
                'vector_score': doc.get('vector_score', doc.get('score', 0)),
                'rerank_score': doc.get('rerank_score')
            }
            citations.append(citation)
        
        return citations