import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime
from typing import List, Optional
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import services
from services.vector_store import VectorStore
from services.embeddings import EmbeddingService
from services.retriever import RetrieverService
from services.llm import LLMService
from utils.chunking import TextChunker
from utils.file_parser import FileParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mini RAG Application",
    description="A production-ready RAG application for document Q&A",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextInput(BaseModel):
    text: str
    title: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    metadata: dict

# Global services (initialized on startup)
vector_store: VectorStore = None
embedding_service: EmbeddingService = None
retriever_service: RetrieverService = None
llm_service: LLMService = None
text_chunker: TextChunker = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("FastAPI server starting...")
    # Server starts immediately, services initialize in background
    asyncio.create_task(initialize_services_background())

async def initialize_services_background():
    """Initialize services in background without blocking startup"""
    global vector_store, embedding_service, retriever_service, llm_service, text_chunker
    
    try:
        logger.info("Starting Mini RAG application services...")
        
        # Validate environment variables
        required_vars = [
            'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'COHERE_API_KEY'
        ]
        
        # Check LLM API key based on model
        llm_model = os.getenv('LLM_MODEL', 'gemini-pro')
        if 'gemini' in llm_model.lower():
            required_vars.append('GEMINI_API_KEY')
        elif 'deepseek' in llm_model.lower():
            required_vars.append('DEEPSEEK_API_KEY')
        else:
            required_vars.append('OPENAI_API_KEY')
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return
        
        # Initialize services
        vector_store = VectorStore(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT'),
            index_name=os.getenv('PINECONE_INDEX_NAME', 'minirag-index')
        )
        
        embedding_service = EmbeddingService(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        )
        
        retriever_service = RetrieverService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            cohere_api_key=os.getenv('COHERE_API_KEY'),
            top_k=int(os.getenv('TOP_K', 10)),
            rerank_top_k=int(os.getenv('RERANK_TOP_K', 5))
        )
        
        llm_service = LLMService(
            model=os.getenv('LLM_MODEL', 'gemini-pro')
        )
        
        text_chunker = TextChunker(
            chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200))
        )
        
        # Initialize vector store index
        try:
            await vector_store.initialize_index()
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {str(e)}")
        
        logger.info("Mini RAG application services started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application services: {str(e)}")
        # Don't raise exception to prevent blocking startup

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ping")
async def ping():
    """Simple ping endpoint for health checks"""
    return {"status": "ok", "message": "pong"}

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main application page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Mini RAG Application</h1><p>Frontend files not found. Please ensure static files are present.</p>",
            status_code=200
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Minimal health check without complex objects
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "vector_store": bool(vector_store),
                "embedding_service": bool(embedding_service),
                "retriever_service": bool(retriever_service),
                "llm_service": bool(llm_service)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file"""
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_content = await file.read()
        FileParser.validate_file(file.filename, len(file_content))
        
        # Parse file
        parsed_data = await FileParser.parse_file(file_content, file.filename)
        
        # Create metadata
        metadata = {
            'filename': file.filename,
            'file_type': parsed_data['file_type'],
            'upload_time': datetime.now().isoformat(),
            'char_count': parsed_data['char_count'],
            'word_count': parsed_data['word_count']
        }
        
        # Chunk text
        chunks = text_chunker.chunk_text(parsed_data['text'], metadata)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text content could be extracted")
        
        # Generate embeddings
        chunks_with_embeddings = await embedding_service.generate_chunk_embeddings(chunks)
        
        # Store in vector database
        success = await vector_store.upsert_chunks(chunks_with_embeddings)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document")
        
        logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks created")
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_tokens": sum(chunk['metadata']['token_count'] for chunk in chunks),
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/add_text")
async def add_text(text_input: TextInput):
    """Add text directly to the knowledge base"""
    try:
        logger.info("Adding text directly to knowledge base")
        
        if not text_input.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        # Create metadata
        metadata = {
            'title': text_input.title or "Direct Text Input",
            'source': 'direct_input',
            'upload_time': datetime.now().isoformat(),
            'char_count': len(text_input.text),
            'word_count': len(text_input.text.split())
        }
        
        # Chunk text
        chunks = text_chunker.chunk_text(text_input.text, metadata)
        if not chunks:
            raise HTTPException(status_code=400, detail="No content could be processed")
        
        # Generate embeddings
        chunks_with_embeddings = await embedding_service.generate_chunk_embeddings(chunks)
        
        # Store in vector database
        success = await vector_store.upsert_chunks(chunks_with_embeddings)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store text")
        
        logger.info(f"Successfully added direct text: {len(chunks)} chunks created")
        
        return {
            "message": "Text added successfully",
            "chunks_created": len(chunks),
            "total_tokens": sum(chunk['metadata']['token_count'] for chunk in chunks),
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add text: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(query_request: QueryRequest):
    """Query the knowledge base and get an AI-generated answer"""
    try:
        logger.info(f"Processing query: {query_request.question[:100]}...")
        
        if not query_request.question.strip():
            raise HTTPException(status_code=400, detail="Empty question provided")
        
        # Retrieve relevant documents
        retrieved_docs = await retriever_service.retrieve_and_rerank(
            question=query_request.question,
            filter_dict=None  # Add filters if needed
        )
        
        if not retrieved_docs:
            # Handle case with no relevant documents
            response = await llm_service.handle_no_context_question(query_request.question)
            return QueryResponse(
                answer=response['answer'],
                citations=[],
                metadata={
                    'model': response['model'],
                    'usage': response['usage'],
                    'retrieved_docs': 0,
                    'processing_time': 0
                }
            )
        
        # Format context for LLM
        context = retriever_service.format_context_for_llm(retrieved_docs)
        
        # Generate answer
        llm_response = await llm_service.generate_answer(
            question=query_request.question,
            context=context
        )
        
        # Extract citations
        citations = retriever_service.extract_citations(retrieved_docs)
        
        # Check answer quality
        quality_check = await llm_service.check_answer_quality(
            question=query_request.question,
            answer=llm_response['answer'],
            context=context
        )
        
        logger.info(f"Query processed successfully. Answer quality: {quality_check['quality_score']}")
        
        return QueryResponse(
            answer=llm_response['answer'],
            citations=citations,
            metadata={
                'model': llm_response['model'],
                'usage': llm_response['usage'],
                'retrieved_docs': len(retrieved_docs),
                'quality_check': quality_check,
                'finish_reason': llm_response['finish_reason']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    try:
        # Return minimal stats to avoid serialization issues
        return {
            "services_status": {
                "vector_store": bool(vector_store),
                "embedding_service": bool(embedding_service),
                "retriever_service": bool(retriever_service),
                "llm_service": bool(llm_service)
            },
            "vector_store": {
                "status": "connected" if vector_store else "disconnected"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        # Return a minimal response to avoid serialization issues
        return {
            "error": "Stats temporarily unavailable",
            "services_status": {
                "vector_store": False,
                "embedding_service": False,
                "retriever_service": False,
                "llm_service": False
            }
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Development server
    port = int(os.getenv("PORT", 8000))
    print("Starting Mini RAG application...")
    print(f"Access the web interface at: http://localhost:{port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )