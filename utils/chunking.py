import tiktoken
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, model: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to a default encoding if model is not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with overlap and metadata
        
        Args:
            text: Text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text, metadata, and position info
        """
        if not text.strip():
            return []
        
        if metadata is None:
            metadata = {}
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            para_tokens = self.count_tokens(paragraph)
            
            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, i))
                    chunk_id += 1
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph
                sub_chunks = self._split_large_paragraph(paragraph, metadata, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Create chunk with overlap
                chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, i))
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                current_tokens = self.count_tokens(current_chunk)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, len(paragraphs)))
        
        logger.info(f"Created {len(chunks)} chunks from {self.count_tokens(text)} tokens")
        return chunks
    
    def _create_chunk(self, text: str, base_metadata: Dict, chunk_id: int, position: int) -> Dict:
        """Create a chunk dictionary with metadata"""
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            'chunk_id': chunk_id,
            'position': position,
            'token_count': self.count_tokens(text),
            'char_count': len(text)
        })
        
        return {
            'text': text.strip(),
            'metadata': chunk_metadata
        }
    
    def _split_large_paragraph(self, paragraph: str, metadata: Dict, start_chunk_id: int) -> List[Dict]:
        """Split a paragraph that's too large into smaller chunks"""
        sentences = paragraph.split('. ')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = start_chunk_id
        
        for i, sentence in enumerate(sentences):
            # Add period back if it's not the last sentence
            if i < len(sentences) - 1:
                sentence += ". "
            
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, i))
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap = self._get_overlap(current_chunk)
                current_chunk = overlap + " " + sentence if overlap else sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, len(sentences)))
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        
        overlap_tokens = tokens[-self.chunk_overlap:]
        overlap_text = self.encoding.decode(overlap_tokens)
        
        # Try to end at a sentence boundary
        sentences = overlap_text.split('. ')
        if len(sentences) > 1:
            return '. '.join(sentences[1:])
        
        return overlap_text