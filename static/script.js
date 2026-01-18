// Application State
let isLoading = false;
let queryCount = 0;

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileStatus = document.getElementById('file-status');
const textStatus = document.getElementById('text-status');
const queryInput = document.getElementById('query-input');
const queryBtn = document.getElementById('query-btn');
const answerSection = document.getElementById('answer-section');
const answerText = document.getElementById('answer-text');
const citationsSection = document.getElementById('citations-section');
const citationsList = document.getElementById('citations-list');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const addTextBtn = document.getElementById('add-text-btn');
const textTitle = document.getElementById('text-title');
const textContent = document.getElementById('text-content');

// Stats elements
const totalDocs = document.getElementById('total-docs');
const totalQueries = document.getElementById('total-queries');
const indexSize = document.getElementById('index-size');
const responseTime = document.getElementById('response-time');
const sourcesCount = document.getElementById('sources-count');
const tokensUsed = document.getElementById('tokens-used');

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkHealthStatus();
    loadStats();
});

// Initialize application
function initializeApp() {
    // Set up tab switching
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        });
    });
}

// Set up event listeners
function setupEventListeners() {
    // File upload events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Text input events
    addTextBtn.addEventListener('click', handleAddText);

    // Query events
    queryBtn.addEventListener('click', handleQuery);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isLoading) {
            handleQuery();
        }
    });

    // Auto-resize textarea
    textContent.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
}

// File selection handler
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
}

// File upload handler
async function handleFileUpload(file) {
    if (isLoading) return;

    // Validate file
    const allowedTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!allowedTypes.includes(file.type)) {
        showFileStatus('error', 'Unsupported file type. Please upload PDF, TXT, or DOCX files.');
        return;
    }

    if (file.size > maxSize) {
        showFileStatus('error', 'File too large. Maximum size is 10MB.');
        return;
    }

    setLoading(true, 'Uploading and processing file...');
    showFileStatus('processing', `Uploading ${file.name}...`);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            showFileStatus('success', `✓ Successfully processed ${result.filename}. Created ${result.chunks_created} chunks.`);
            showToast('success', `File uploaded successfully! Created ${result.chunks_created} chunks.`);
            loadStats(); // Refresh stats
        } else {
            throw new Error(result.detail || 'Upload failed');
        }

    } catch (error) {
        console.error('Upload error:', error);
        showFileStatus('error', `✗ Upload failed: ${error.message}`);
        showToast('error', `Upload failed: ${error.message}`);
    } finally {
        setLoading(false);
        // Reset file input
        fileInput.value = '';
    }
}

// Add text handler
async function handleAddText() {
    if (isLoading) return;

    const title = textTitle.value.trim();
    const content = textContent.value.trim();

    if (!content) {
        showTextStatus('error', 'Please enter some text content.');
        return;
    }

    setLoading(true, 'Processing text...');
    showTextStatus('processing', 'Adding text to knowledge base...');

    try {
        const response = await fetch('/add_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: content,
                title: title || null
            })
        });

        const result = await response.json();

        if (response.ok) {
            showTextStatus('success', `✓ Successfully added text. Created ${result.chunks_created} chunks.`);
            showToast('success', `Text added successfully! Created ${result.chunks_created} chunks.`);
            
            // Clear form
            textTitle.value = '';
            textContent.value = '';
            textContent.style.height = 'auto';
            
            loadStats(); // Refresh stats
        } else {
            throw new Error(result.detail || 'Failed to add text');
        }

    } catch (error) {
        console.error('Add text error:', error);
        showTextStatus('error', `✗ Failed to add text: ${error.message}`);
        showToast('error', `Failed to add text: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// Query handler
async function handleQuery() {
    if (isLoading) return;

    const question = queryInput.value.trim();
    if (!question) {
        showToast('error', 'Please enter a question.');
        return;
    }

    setLoading(true, 'Searching and generating answer...');
    const startTime = Date.now();

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question
            })
        });

        const result = await response.json();
        const endTime = Date.now();
        const duration = endTime - startTime;

        if (response.ok) {
            displayAnswer(result, duration);
            queryCount++;
            totalQueries.textContent = queryCount;
            showToast('success', 'Answer generated successfully!');
        } else {
            throw new Error(result.detail || 'Query failed');
        }

    } catch (error) {
        console.error('Query error:', error);
        showToast('error', `Query failed: ${error.message}`);
        answerSection.style.display = 'none';
    } finally {
        setLoading(false);
    }
}

// Display answer
function displayAnswer(result, duration) {
    // Display main answer
    answerText.textContent = result.answer;
    answerSection.style.display = 'block';

    // Update metadata
    responseTime.textContent = duration;
    sourcesCount.textContent = result.citations.length;
    tokensUsed.textContent = result.metadata.usage.total_tokens;

    // Display citations
    if (result.citations && result.citations.length > 0) {
        displayCitations(result.citations);
        citationsSection.style.display = 'block';
    } else {
        citationsSection.style.display = 'none';
    }

    // Scroll to answer
    answerSection.scrollIntoView({ behavior: 'smooth' });
}

// Display citations
function displayCitations(citations) {
    citationsList.innerHTML = '';
    
    citations.forEach(citation => {
        const citationDiv = document.createElement('div');
        citationDiv.className = 'citation-item';
        
        const header = document.createElement('div');
        header.className = 'citation-header';
        
        const citationId = document.createElement('span');
        citationId.className = 'citation-id';
        citationId.textContent = `Source ${citation.id}`;
        
        const scores = document.createElement('div');
        scores.className = 'citation-scores';
        
        if (citation.vector_score) {
            const vectorScore = document.createElement('span');
            vectorScore.textContent = `Relevance: ${(citation.vector_score * 100).toFixed(1)}%`;
            scores.appendChild(vectorScore);
        }
        
        if (citation.rerank_score) {
            const rerankScore = document.createElement('span');
            rerankScore.textContent = `Rerank: ${(citation.rerank_score * 100).toFixed(1)}%`;
            scores.appendChild(rerankScore);
        }
        
        header.appendChild(citationId);
        header.appendChild(scores);
        
        const text = document.createElement('div');
        text.className = 'citation-text';
        text.textContent = citation.text_preview;
        
        citationDiv.appendChild(header);
        citationDiv.appendChild(text);
        citationsList.appendChild(citationDiv);
    });
}

// Show file status
function showFileStatus(type, message) {
    fileStatus.className = `file-status ${type}`;
    fileStatus.textContent = message;
    fileStatus.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            fileStatus.style.display = 'none';
        }, 5000);
    }
}

// Show text status
function showTextStatus(type, message) {
    textStatus.className = `file-status ${type}`;
    textStatus.textContent = message;
    textStatus.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            textStatus.style.display = 'none';
        }, 5000);
    }
}

// Set loading state
function setLoading(loading, message = 'Processing...') {
    isLoading = loading;
    loadingText.textContent = message;
    loadingOverlay.style.display = loading ? 'flex' : 'none';
    
    // Disable/enable buttons
    queryBtn.disabled = loading;
    addTextBtn.disabled = loading;
    
    if (loading) {
        queryBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        addTextBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    } else {
        queryBtn.innerHTML = '<i class="fas fa-search"></i> Ask';
        addTextBtn.innerHTML = '<i class="fas fa-plus"></i> Add Text';
    }
}

// Show toast notification
function showToast(type, message) {
    const toastContainer = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = document.createElement('i');
    icon.className = type === 'success' ? 'fas fa-check-circle' : 
                    type === 'error' ? 'fas fa-exclamation-circle' : 
                    'fas fa-info-circle';
    
    const text = document.createElement('span');
    text.textContent = message;
    
    toast.appendChild(icon);
    toast.appendChild(text);
    toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Check health status
async function checkHealthStatus() {
    try {
        const response = await fetch('/health');
        const result = await response.json();
        
        if (response.ok && result.status === 'healthy') {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'Online';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Offline';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Offline';
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const result = await response.json();
        
        if (response.ok) {
            // Update stats display
            const vectorCount = result.vector_store.total_vector_count || 0;
            totalDocs.textContent = vectorCount;
            
            const indexFullness = result.vector_store.index_fullness || 0;
            indexSize.textContent = `${(indexFullness * 100).toFixed(1)}%`;
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
        totalDocs.textContent = 'Error';
        indexSize.textContent = 'Error';
    }
}

// Periodic health checks and stats updates
setInterval(() => {
    checkHealthStatus();
    loadStats();
}, 30000); // Every 30 seconds

// Auto-focus query input on page load
window.addEventListener('load', () => {
    queryInput.focus();
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to submit query
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !isLoading) {
        if (document.activeElement === queryInput) {
            handleQuery();
        } else if (document.activeElement === textContent) {
            handleAddText();
        }
    }
    
    // Escape to clear current operation
    if (e.key === 'Escape' && isLoading) {
        // Note: This would require cancellation support in the backend
        console.log('Escape pressed - operation cancellation not implemented');
    }
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        // Page became visible, refresh status
        checkHealthStatus();
        loadStats();
    }
});

// Error handling for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    showToast('error', 'An unexpected error occurred. Please try again.');
});

// Service worker registration (for future PWA capabilities)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Note: Service worker would be implemented in future versions
        console.log('Service Worker support detected');
    });
}