import google.generativeai as genai
from typing import List, Dict, Optional
import logging
import json
import os
import asyncio

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating answers using Google Gemini AI"""
    
    def __init__(self, model: str = "gemini-pro"):
        self.model = model
        
        # Configure Gemini AI
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
        genai.configure(api_key=gemini_api_key)
        self.client = genai.GenerativeModel(model_name=model)
        
        # System prompt for RAG
        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 

Your task:
1. Answer the user's question using ONLY the information provided in the context
2. Include inline citations in your answer using the format [1], [2], etc. that correspond to the source numbers in the context
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Be concise but comprehensive
5. Maintain a professional and helpful tone

Guidelines:
- Always cite your sources using the citation numbers from the context
- If multiple sources support the same point, cite all relevant sources like [1, 2]
- If the question cannot be answered from the context, state: "I don't have enough information in the provided context to answer this question."
- Don't make up information not present in the context
- Format your response clearly with proper structure when needed"""

    async def generate_answer(self, question: str, context: str, 
                            max_tokens: int = 1000, temperature: float = 0.1) -> Dict:
        """
        Generate an answer to the question based on provided context
        
        Args:
            question: User's question
            context: Retrieved context with citations
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Construct prompt for Gemini
            prompt = f"""{self.system_prompt}

Context:
{context}

Question: {question}

Please answer the question based on the provided context, including appropriate citations."""
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate response using asyncio to run sync method
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, 
                lambda: self.client.generate_content(prompt, generation_config=generation_config))
            
            answer = response.text.strip()
            
            # Calculate approximate token counts (Gemini doesn't provide exact counts)
            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
            completion_tokens = len(answer.split()) * 1.3
            total_tokens = prompt_tokens + completion_tokens
            
            result = {
                'answer': answer,
                'model': self.model,
                'usage': {
                    'prompt_tokens': int(prompt_tokens),
                    'completion_tokens': int(completion_tokens),
                    'total_tokens': int(total_tokens)
                },
                'finish_reason': 'completed'
            }
            
            logger.info(f"Generated answer using approximately {int(total_tokens)} tokens")
            return result
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error generating answer: {error_message}")
            
            # Handle specific API errors with fallback responses
            if "quota" in error_message.lower() or "limit" in error_message.lower():
                return self._generate_fallback_response(question, context, "API quota/limit exceeded")
            elif "invalid" in error_message.lower() or "key" in error_message.lower():
                return self._generate_fallback_response(question, context, "API key issue")
            else:
                return self._generate_fallback_response(question, context, "API temporarily unavailable")
    
    def _generate_fallback_response(self, question: str, context: str, reason: str) -> Dict:
        """Generate a simple fallback response when API is unavailable"""
        
        # Try to provide a simple context-based response
        context_lines = context.split('\n')
        relevant_lines = []
        question_words = question.lower().split()
        
        # Find lines in context that contain question words
        for line in context_lines:
            if any(word in line.lower() for word in question_words if len(word) > 3):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            fallback_answer = f"Based on the provided context, here are the most relevant excerpts related to your question:\n\n"
            for i, line in enumerate(relevant_lines[:3], 1):  # Limit to top 3 relevant lines
                if line:
                    fallback_answer += f"[{i}] {line}\n\n"
            fallback_answer += f"Note: This is a simplified response due to {reason}. For a more comprehensive answer, please try again later."
        else:
            fallback_answer = f"I found relevant information in the provided context, but I'm unable to generate a detailed response due to {reason}. Please review the context provided and try again later."
        
        return {
            'answer': fallback_answer,
            'model': f"{self.model} (fallback)",
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            },
            'finish_reason': 'fallback_response',
            'fallback': True,
            'fallback_reason': reason
        }
    
    async def generate_answer_streaming(self, question: str, context: str, 
                                      max_tokens: int = 1000, temperature: float = 0.1):
        """
        Generate streaming answer response
        
        Args:
            question: User's question
            context: Retrieved context with citations
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Yields:
            Chunks of the response as they're generated
        """
        try:
            # Construct prompt for Gemini
            prompt = f"""{self.system_prompt}

Context:
{context}

Question: {question}

Please answer the question based on the provided context, including appropriate citations."""
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Note: Gemini doesn't support streaming in the same way as OpenAI
            # We'll generate the full response and yield it in chunks
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, 
                lambda: self.client.generate_content(prompt, generation_config=generation_config))
            
            # Yield the response in chunks to simulate streaming
            answer = response.text.strip()
            words = answer.split()
            chunk_size = 10  # Words per chunk
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += ' '
                yield chunk
                await asyncio.sleep(0.05)  # Small delay to simulate streaming
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error generating streaming answer: {error_message}")
            
            # For streaming, yield a fallback response
            if "quota" in error_message.lower() or "limit" in error_message.lower():
                fallback = self._generate_fallback_response(question, context, "API quota/limit exceeded")
                yield f"⚠️ {fallback['answer']}"
            elif "invalid" in error_message.lower() or "key" in error_message.lower():
                fallback = self._generate_fallback_response(question, context, "API key issue")
                yield f"⚠️ {fallback['answer']}"
            else:
                fallback = self._generate_fallback_response(question, context, "API temporarily unavailable")
                yield f"⚠️ {fallback['answer']}"
    
    def extract_citations_from_answer(self, answer: str) -> List[int]:
        """
        Extract citation numbers from the answer text
        
        Args:
            answer: Generated answer text
            
        Returns:
            List of citation numbers found in the answer
        """
        import re
        
        # Find all citation patterns like [1], [2], [1,2], [1, 2, 3], etc.
        citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
        matches = re.findall(citation_pattern, answer)
        
        citations = set()
        for match in matches:
            # Split by comma and extract numbers
            numbers = [int(num.strip()) for num in match.split(',')]
            citations.update(numbers)
        
        return sorted(list(citations))
    
    async def check_answer_quality(self, question: str, answer: str, context: str) -> Dict:
        """
        Check the quality of generated answer
        
        Args:
            question: Original question
            answer: Generated answer
            context: Source context
            
        Returns:
            Quality assessment dictionary
        """
        try:
            # Check basic metrics
            has_citations = bool(self.extract_citations_from_answer(answer))
            has_no_answer = "don't have enough information" in answer.lower() or "cannot answer" in answer.lower()
            
            # Check if answer is too short (might indicate incomplete response)
            is_too_short = len(answer.split()) < 10
            
            # Check if answer seems complete
            ends_properly = answer.rstrip().endswith(('.', '!', '?', ')'))
            
            quality_score = 1.0
            issues = []
            
            if not has_citations and not has_no_answer:
                quality_score -= 0.3
                issues.append("Missing citations")
            
            if is_too_short and not has_no_answer:
                quality_score -= 0.2
                issues.append("Answer too short")
            
            if not ends_properly:
                quality_score -= 0.1
                issues.append("Incomplete ending")
            
            return {
                'quality_score': max(0, quality_score),
                'has_citations': has_citations,
                'has_no_answer': has_no_answer,
                'is_too_short': is_too_short,
                'ends_properly': ends_properly,
                'issues': issues,
                'citation_count': len(self.extract_citations_from_answer(answer))
            }
            
        except Exception as e:
            logger.error(f"Error checking answer quality: {str(e)}")
            return {'quality_score': 0.5, 'issues': ['Quality check failed']}
    
    async def handle_no_context_question(self, question: str) -> Dict:
        """
        Generate response when no relevant context is found
        
        Args:
            question: User's question
            
        Returns:
            Response dictionary
        """
        answer = "I don't have any relevant information in my knowledge base to answer this question. Please try uploading relevant documents or rephrasing your question."
        
        return {
            'answer': answer,
            'model': self.model,
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'finish_reason': 'no_context'
        }