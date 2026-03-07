# Course Materials RAG System - Agent Guide

This document provides essential information for AI coding agents working on this project.

## Project Overview

A full-stack Retrieval-Augmented Generation (RAG) system that answers questions about course materials using semantic search and AI-powered responses. The application enables users to query educational content and receive intelligent, context-aware answers based on course documents.

## Technology Stack

### Backend
- **Python**: 3.13+
- **Web Framework**: FastAPI
- **Vector Database**: ChromaDB
- **AI Model**: Moonshot AI (Kimi) via OpenAI-compatible API
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
│   ├── config.py        # Configuration settings
│   └── tests/           # Test suite
│       ├── test_ai_generator.py
│       ├── test_search_tools.py
│       ├── test_rag_system.py
│       ├── test_error_scenarios.py
│       ├── test_integration.py
│       └── test_max_results.py
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
# Edit .env and add your MOONSHOT_API_KEY from https://platform.moonshot.cn/
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
4. **Tool Execution** (if needed) → Kimi calls appropriate tool(s):
   - `search_course_content` for course content queries
   - `get_course_outline` for course structure/outline queries
   - Supports **up to 2 sequential rounds** of tool calling for complex queries
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
- System prompt defines response style (brief, educational, example-supported, well-formatted)
- Tool-based search architecture using OpenAI-compatible API format
- Supports **up to 2 sequential rounds** of tool calling (`MAX_TOOL_ROUNDS = 2`)
- `_execute_tool_round()`: Executes a single round of tool calls
- Tools remain available in each API round until max rounds reached or AI provides final answer
- Uses Moonshot AI (Kimi) via OpenAI client with custom base URL

**SearchTools** (`search_tools.py`):
- Abstract `Tool` base class for extensibility
- `CourseSearchTool`: Searches course content with metadata filtering
- `CourseOutlineTool`: Retrieves course outline (title, link, lesson list)
- `ToolManager`: Registers and executes tools, tracks sources
- Tool definitions use OpenAI format (type: function)
- Sources include display text and lesson links for clickable citations

**Available Tools:**

1. **`search_course_content`**: Search for specific content within course materials
   - Parameters: `query` (required), `course_name` (optional), `lesson_number` (optional)
   - Use case: Finding specific information within lesson content
   - **Important**: When users ask about a specific lesson (e.g., "lesson 5"), ALWAYS use the `lesson_number` parameter

2. **`get_course_outline`**: Retrieve complete course outline
   - Parameters: `course_name` (required)
   - Returns: Course title, course link, total lesson count, and list of all lessons
   - Format: Each lesson on a single line (e.g., "Lesson N: Title")
   - Use case: Answering questions about course structure, lesson lists

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
| `MAX_RESULTS` | 5 | Max search results (must be >= 1) |
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
  "session_id": null
}
```

**Response:**
```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": [
    {"text": "Course Title - Lesson 1", "link": "https://example.com/lesson1"}
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
  "course_titles": ["Course 1", "Course 2"]
}
```

### POST `/api/session/delete`
Delete a conversation session.

**Request:**
```json
{
  "session_id": "session_1"
}
```

## Code Style Guidelines

- **Type hints**: Use Python type hints throughout (`typing` module)
- **Docstrings**: Google-style docstrings for classes and functions
- **Pydantic models**: Use for request/response validation
- **Error handling**: Wrap external API calls and file operations in try-except blocks
- **Logging**: Use `print()` for operational messages

## Testing

### Running Tests

```bash
cd backend
uv run python -m unittest discover tests/ -v
```

### Test Files
- `test_search_tools.py` - Tests for CourseSearchTool, CourseOutlineTool, ToolManager
- `test_ai_generator.py` - Tests for AIGenerator and tool calling integration
- `test_rag_system.py` - Tests for RAGSystem content-query handling
- `test_error_scenarios.py` - Tests for error handling and edge cases
- `test_max_results.py` - Tests for MAX_RESULTS validation
- `test_integration.py` - End-to-end integration tests

### Test Coverage
- Tool execution with various inputs and filters
- Error handling in tools (database errors, missing courses, etc.)
- AI generator tool calling behavior (single call, sequential calls)
- RAG system query processing pipeline
- Edge cases (special characters, long queries, empty results)
- MAX_RESULTS validation (must be positive integer)

### Manual Testing

1. Start the server: `./run.sh`
2. Ensure documents are loaded (check console output)
3. Open `http://localhost:8000`
4. Test queries:
   - General knowledge (should not use search)
   - Course-specific (should trigger `search_course_content`)
   - Course outline queries (should trigger `get_course_outline`)
   - Sequential/multi-part queries (may trigger 2 rounds of tool calls)
5. Verify sources appear as clickable links
6. Check conversation continuity (session persistence)

## Security Considerations

- **API Key**: Never commit `.env` file containing `MOONSHOT_API_KEY`
- **CORS**: Currently allows all origins (`["*"]`) - configure for production
- **Trusted Hosts**: Currently allows all hosts (`["*"]`) - restrict in production
- **Input Validation**: Pydantic models validate API inputs
- **File Access**: Document processor reads from filesystem - validate paths if accepting user uploads

## Error Handling

### Tool-Level Error Handling
- `CourseSearchTool.execute()` catches all exceptions and returns error messages
- `CourseOutlineTool.execute()` catches all exceptions and returns error messages
- Error messages are passed back to AI for appropriate response generation

### AI Generator Error Handling
- `_execute_tool_round()` catches tool execution errors and returns success/failure status
- Sequential tool loop terminates early on tool failure
- API authentication errors provide helpful setup instructions in Chinese
- All other API errors are logged and re-raised with context

### API-Level Error Handling
- FastAPI endpoints catch exceptions and return appropriate HTTP status codes
- Authentication errors return 401 with setup instructions
- Other errors return 500 with error details

## Common Issues

1. **ChromaDB resource tracker warnings**: Suppressed in `app.py` via warnings filter
2. **UTF-8 encoding issues**: DocumentProcessor falls back to `errors='ignore'`
3. **Port conflicts**: Change port in `run.sh` if 8000 is in use
4. **Missing API key**: Application will fail when trying to call Kimi API
5. **Invalid MAX_RESULTS**: Raises `ValueError` on startup if <= 0
6. **Embedding model download issues**: Use HF mirror if needed:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
     --local-dir ~/.cache/torch/sentence_transformers/all-MiniLM-L6-v2 \
     --local-dir-use-symlinks False
   ```

## Frontend Cache Busting

CSS and JS files include version query params (`?v=12`, `?v=11`). Increment versions when making frontend changes:
- `index.html`: Update `style.css?v=X` and `script.js?v=X`

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
- Session IDs formatted as `session_1`, `session_2`, etc.
- History limited to `MAX_HISTORY` exchanges (default: 2 turns)

## Related Documentation

- `README.md`: User-facing setup and usage instructions
- `backend-refactor-tool.md`: Refactoring specifications (internal)
