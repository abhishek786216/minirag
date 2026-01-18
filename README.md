# Mini RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application that allows users to upload documents, ask questions, and receive AI-powered answers with citations.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │  Vector Store   │
│   (HTML/JS)     │◄──►│   (FastAPI)     │◄──►│   (Pinecone)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   LLM Service   │
                       │   (OpenAI)      │
                       └─────────────────┘
```

## ✨ Features

- **Document Upload**: Support for PDF and TXT files
- **Text Input**: Direct text input for immediate processing
- **Semantic Search**: Vector-based document retrieval
- **Reranking**: Cohere reranker for improved relevance
- **AI Answers**: OpenAI GPT-powered responses with citations
- **Real-time Processing**: Async document processing
- **Production Ready**: Error handling, logging, and monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API Keys:
  - OpenAI API Key
  - Pinecone API Key
  - Cohere API Key

### Installation

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd mini-rag
   pip install -r requirements.txt
   ```

2. **Environment Setup**:
   Create `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_env
   PINECONE_INDEX_NAME=minirag-index
   COHERE_API_KEY=your_cohere_key
   ```

3. **Run Application**:
   ```bash
   python main.py
   ```

4. **Access**: Open http://localhost:8000

## 📊 Configuration

### Vector Database Settings
- **Provider**: Pinecone
- **Index Name**: minirag-index
- **Dimensions**: 1536 (OpenAI text-embedding-ada-002)
- **Metric**: cosine
- **Upsert Strategy**: Batch processing with metadata

### Chunking Strategy
- **Size**: 1000 tokens
- **Overlap**: 200 tokens
- **Method**: Recursive text splitting
- **Metadata**: Source, title, chunk_id, position

### Retrieval Settings
- **Top-k**: 10 initial results
- **Reranker**: Cohere rerank-english-v2.0
- **Final Results**: Top 5 after reranking

### LLM Configuration
- **Provider**: OpenAI
- **Model**: gpt-3.5-turbo
- **Temperature**: 0.1
- **Max Tokens**: 1000

## 🔧 API Endpoints

- `POST /upload`: Upload document files
- `POST /add_text`: Add text directly
- `POST /query`: Query the knowledge base
- `GET /health`: Health check
- `GET /`: Frontend interface

## 📁 Project Structure

```
mini-rag/
├── main.py              # FastAPI application
├── services/
│   ├── vector_store.py  # Vector database operations
│   ├── embeddings.py    # Embedding generation
│   ├── retriever.py     # Document retrieval
│   └── llm.py          # LLM integration
├── utils/
│   ├── chunking.py     # Text chunking logic
│   └── file_parser.py  # Document parsing
├── static/
│   ├── index.html      # Frontend interface
│   ├── style.css       # Styling
│   └── script.js       # JavaScript logic
├── requirements.txt    # Dependencies
├── .env.example       # Environment template
└── README.md          # Documentation
```

## 🚀 Deployment

### Vercel (Recommended)
1. Install Vercel CLI: `npm i -g vercel`
2. Configure `vercel.json`
3. Set environment variables in Vercel dashboard
4. Deploy: `vercel --prod`

### Alternative Platforms
- **Render**: Direct GitHub integration
- **Railway**: Docker or buildpack deployment
- **Heroku**: Procfile-based deployment

## 🔍 Usage Examples

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

### Add Text
```bash
curl -X POST "http://localhost:8000/add_text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text content here", "title": "Sample Text"}'
```

### Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

## 📈 Monitoring & Evaluation

### Success Metrics
- Query response time: < 3 seconds
- Retrieval precision: > 0.8
- User satisfaction: Measured via feedback

### Evaluation Dataset
Includes 5 Q&A pairs for basic functionality testing:

1. **Q**: "What are the key features?"
   **A**: Lists main application capabilities

2. **Q**: "How do I set up the environment?"
   **A**: Environment configuration steps

3. **Q**: "What is the chunking strategy?"
   **A**: Token-based chunking with overlap

4. **Q**: "Which LLM provider is used?"
   **A**: OpenAI with specific model details

5. **Q**: "How is document retrieval performed?"
   **A**: Vector search with reranking process

## ⚡ Performance Considerations

- **Async Processing**: All I/O operations are asynchronous
- **Batch Operations**: Vector upserts in batches of 100
- **Caching**: Embedding cache for repeated queries
- **Rate Limiting**: Built-in API rate limiting
- **Error Handling**: Comprehensive error recovery

## 🔒 Security

- **API Key Management**: Server-side only
- **File Validation**: Type and size restrictions
- **Input Sanitization**: XSS protection
- **CORS**: Configured for production

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test document upload
curl -X POST "http://localhost:8000/upload" -F "file=@test.pdf"

# Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test question"}'
```

## 📝 Remarks

### Current Limitations
- **Rate Limits**: Subject to provider API limits
- **File Size**: Max 10MB per file
- **Concurrent Users**: Optimized for moderate load
- **Language**: Primarily English support

### Future Enhancements
- **Multi-language Support**: Add language detection
- **Advanced Chunking**: Semantic-aware chunking
- **User Authentication**: Add user management
- **Conversation Memory**: Maintain chat history
- **Advanced Analytics**: Usage metrics dashboard

### Trade-offs Made
- **Simplicity vs Features**: Focused on core RAG functionality
- **Speed vs Accuracy**: Balanced retrieval speed with quality
- **Cost vs Performance**: Used efficient but cost-effective models

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Open an issue with detailed error logs

## 📄 License

MIT License - see LICENSE file for details.

---

Built with ❤️ for efficient document Q&A experiences.