# Course Materials RAG System - Agent Guide

This document provides essential information for AI coding agents working on this project.

## Project Overview

A full-stack Retrieval-Augmented Generation (RAG) system that answers questions about course materials using semantic search and AI-powered responses. The application enables users to query educational content and receive intelligent, context-aware answers based on course documents.

## Technology Stack

### Backend
- **Python**: 3.13+
- **Web Framework**: FastAPI
- **Vector Database**: ChromaDB
- **AI Model**: Moonshot AI (Kimi)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Package Manager**: uv (Astral)

### Frontend
- **HTML/CSS/JavaScript**: Vanilla, no frameworks
- **Markdown Rendering**: marked.js (CDN)

## Project Structure

```
.
├── backend/              # FastAPI application
│   ├── app.py           # API entry point, endpoints, static files
│   ├── rag_system.py    # Main RAG orchestrator
│   ├── ai_generator.py  # Kimi API integration with tool support
│   ├── vector_store.py  # ChromaDB wrapper for semantic search
│   ├── document_processor.py  # Course document parsing and chunking
│   ├── search_tools.py  # Tool system for AI-driven search
│   ├── session_manager.py     # In-memory conversation sessions
│   ├── models.py        # Pydantic data models
│   └── config.py        # Configuration settings
├── frontend/            # Static web assets
│   ├── index.html       # Main page with chat UI
│   ├── script.js        # Client-side logic
│   └── style.css        # Styling
├── docs/                # Course document storage
│   └── *.txt            # Course material text files
├── pyproject.toml       # Python dependencies
├── run.sh              # Startup script
└── .env                # Environment variables (create from .env.example)
```

## Build and Run Commands

> ⚠️ **Important**: Always use `uv` to run the server. Do not use `pip` directly.

### Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your MOONSHOT_API_KEY
```

### Running the Application

**Quick Start:**
```bash
chmod +x run.sh
./run.sh
```

**Manual Start:**
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture Details

### RAG Pipeline Flow

1. **User Query** → Frontend → POST `/api/query`
2. **Session Management** → Creates/retrieves session for conversation history
3. **AI Generation** → Kimi receives query + system prompt + available tools
4. **Tool Execution** (if needed) → Kimi calls appropriate tool:
   - `search_course_content` for course content queries
   - `get_course_outline` for course structure/outline queries
5. **Semantic Search** → VectorStore queries ChromaDB collections
6. **Response Formation** → Kimi synthesizes search results into answer
7. **History Update** → Exchange saved to session, sources tracked

### Key Components

**VectorStore** (`vector_store.py`):
- Two ChromaDB collections:
  - `course_catalog`: Course metadata for name resolution (includes lesson links)
  - `course_content`: Chunked text content for semantic search
- Handles course name fuzzy matching via vector similarity
- Supports filtering by course and lesson number
- `get_lesson_link(course_title, lesson_number)`: Retrieves lesson URL from catalog

**DocumentProcessor** (`document_processor.py`):
- Parses course text files with expected format:
  - Line 1: `Course Title: [title]`
  - Line 2: `Course Link: [url]`
  - Line 3: `Course Instructor: [instructor]`
  - Then: `Lesson N: [title]` markers with content
- Sentence-aware text chunking with configurable overlap

**AIGenerator** (`ai_generator.py`):
- System prompt defines response style (brief, educational, example-supported)
- Tool-based search architecture using OpenAI-compatible API format
- Handles tool execution loop for multi-step reasoning
- Uses Moonshot AI (Kimi) via OpenAI client with custom base URL

**SearchTools** (`search_tools.py`):
- Abstract Tool base class for extensibility
- `CourseSearchTool`: Searches course content with metadata filtering
- `CourseOutlineTool`: Retrieves course outline (title, link, lesson list) for structure queries
- `ToolManager`: Registers and executes tools, tracks sources
- Tool definitions use OpenAI format (type: function)
- Sources include display text and lesson links for clickable citations

**Available Tools:**

1. **`search_course_content`**: Search for specific content within course materials
   - Parameters: `query` (required), `course_name` (optional), `lesson_number` (optional)
   - Use case: Finding specific information within lesson content

2. **`get_course_outline`**: Retrieve complete course outline
   - Parameters: `course_name` (required)
   - Returns: Course title, course link, total lesson count, and list of all lessons with numbers and titles
   - Format: Each lesson should be on a single line (e.g., "Lesson N: Title")
   - Use case: Answering questions about course structure, lesson lists, outlines

## Data Models

See `backend/models.py` for Pydantic definitions:
- `Course`: Title, instructor, lessons list
- `Lesson`: Number, title, link
- `CourseChunk`: Content with metadata (course, lesson, index)
- `Source`: Display text and optional link for citations

## Configuration

All settings in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MOONSHOT_MODEL` | moonshot-v1-8k | Kimi model version |
| `MOONSHOT_BASE_URL` | https://api.moonshot.cn/v1 | Moonshot API endpoint |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `CHUNK_SIZE` | 800 | Characters per text chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | Max search results |
| `MAX_HISTORY` | 2 | Conversation turns to remember |
| `CHROMA_PATH` | ./chroma_db | Vector DB storage location |

Environment variables loaded from `.env`:
- `MOONSHOT_API_KEY`: Required for Kimi API access
- `MOONSHOT_BASE_URL`: Optional, Moonshot API base URL
- `MOONSHOT_MODEL`: Optional, Kimi model to use

## API Endpoints

### POST `/api/query`
Query the RAG system.

**Request:**
```json
{
  "query": "What is RAG?",
  "session_id": null  // or existing session ID
}
```

**Response:**
```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": [
    {"text": "Course Title - Lesson 1", "link": "https://example.com/lesson1"},
    {"text": "Course Title - Lesson 3", "link": "https://example.com/lesson3"}
  ],
  "session_id": "session_1"
}
```

### GET `/api/courses`
Get course statistics.

**Response:**
```json
{
  "total_courses": 4,
  "course_titles": ["Course 1", "Course 2", ...]
}
```

## Code Style Guidelines

- **Type hints**: Use Python type hints throughout (`typing` module)
- **Docstrings**: Google-style docstrings for classes and functions
- **Pydantic models**: Use for request/response validation
- **Error handling**: Wrap external API calls and file operations in try-except blocks
- **Logging**: Use `print()` for operational messages (production may need proper logging)

## Development Notes

### Document Processing
- Course documents must follow the expected format (see DocumentProcessor)
- Files are processed at startup if placed in `docs/` folder
- Supported formats: `.txt`, `.pdf`, `.docx`
- Duplicate courses (by title) are automatically skipped

### Vector Store
- ChromaDB persists data to `backend/chroma_db/` (gitignored)
- To rebuild: pass `clear_existing=True` to `add_course_folder()`
- Collections are created automatically on first run

### Session Management
- Sessions are in-memory only (not persisted)
- Session IDs are formatted as `session_1`, `session_2`, etc.
- History is limited to `MAX_HISTORY` exchanges

### Frontend Cache Busting
- CSS and JS files include version query params (`?v=10`)
- Increment versions when making frontend changes

## Testing

Currently no automated test suite exists. Manual testing approach:

1. Start the server: `./run.sh`
2. Ensure documents are loaded (check console output)
3. Open `http://localhost:8000`
4. Test queries:
   - General knowledge (should not use search)
   - Course-specific (should trigger `search_course_content` tool)
   - Course outline queries (should trigger `get_course_outline` tool), e.g.:
     - "What lessons are in the MCP course?"
     - "Show me the course outline for Chroma"
     - "List all lessons in the Computer Use course"
   - With lesson filters
5. Verify sources appear in collapsible section as clickable links
6. Check conversation continuity (session persistence)

## Security Considerations

- **API Key**: Never commit `.env` file containing `MOONSHOT_API_KEY`
- **CORS**: Currently allows all origins (`["*"]`) - configure for production
- **Trusted Hosts**: Currently allows all hosts (`["*"]`) - restrict in production
- **Input Validation**: Pydantic models validate API inputs
- **File Access**: Document processor reads from filesystem - validate paths if accepting user uploads

## Common Issues

1. **ChromaDB resource tracker warnings**: Suppressed in `app.py` via warnings filter
2. **UTF-8 encoding issues**: DocumentProcessor falls back to `errors='ignore'`
3. **Port conflicts**: Change port in `run.sh` if 8000 is in use
4. **Missing API key**: Application will fail when trying to call Kimi API
5. **Embedding model download issues**: If you cannot access Hugging Face Hub, use the following steps to download models via mirror:
   ```bash
   # Set mirror endpoint and download model
   export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
     --local-dir ~/.cache/torch/sentence_transformers/all-MiniLM-L6-v2 \
     --local-dir-use-symlinks False
   ```
   The application will automatically detect and use locally cached models.

## Related Documentation

- `diagram.md`: Contains detailed architecture diagrams (Mermaid)
- `README.md`: User-facing setup and usage instructions
