"""Pytest configuration and shared fixtures for RAG system tests.

This module provides common fixtures and configuration for testing the RAG system.
Fixtures are automatically discovered by pytest and can be injected into test functions.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Source


class MockConfig:
    """Mock configuration for testing.
    
    Provides configuration values that match the production Config class
    but use test-appropriate values (e.g., test API keys).
    """
    def __init__(self):
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 100
        self.CHROMA_PATH = "./test_chroma_db"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.MAX_HISTORY = 2
        self.MOONSHOT_API_KEY = "test-api-key"
        self.MOONSHOT_MODEL = "moonshot-v1-8k"
        self.MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"


@pytest.fixture
def mock_config():
    """Fixture providing a mock configuration object.
    
    Returns:
        MockConfig: Configuration object with test values.
    """
    return MockConfig()


@pytest.fixture
def mock_vector_store():
    """Fixture providing a mock VectorStore instance.
    
    Returns:
        Mock: Configured mock VectorStore with common methods mocked.
    """
    mock_store = Mock()
    
    # Configure default return values for common methods
    mock_store.search.return_value = Mock(
        documents=["Test content about RAG"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1],
        error=None
    )
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_course_link.return_value = "https://example.com/course"
    mock_store.get_course_count.return_value = 3
    mock_store.get_existing_course_titles.return_value = ["Course A", "Course B", "Course C"]
    mock_store._resolve_course_name.return_value = "Test Course"
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": "Test Course",
            "course_link": "https://example.com/course",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/l1"},
                {"lesson_number": 2, "lesson_title": "Advanced Topics", "lesson_link": "https://example.com/l2"}
            ]
        }
    ]
    
    return mock_store


@pytest.fixture
def mock_ai_generator():
    """Fixture providing a mock AIGenerator instance.
    
    Returns:
        Mock: Configured mock AIGenerator.
    """
    mock_gen = Mock()
    mock_gen.generate_response.return_value = "This is a test response from the AI."
    return mock_gen


@pytest.fixture
def mock_session_manager():
    """Fixture providing a mock SessionManager instance.
    
    Returns:
        Mock: Configured mock SessionManager with session tracking.
    """
    mock_sm = Mock()
    mock_sm.create_session.return_value = "session_123"
    mock_sm.get_conversation_history.return_value = None
    mock_sm.add_exchange.return_value = None
    mock_sm.delete_session.return_value = True
    
    return mock_sm


@pytest.fixture
def mock_tool_manager():
    """Fixture providing a mock ToolManager instance.
    
    Returns:
        Mock: Configured mock ToolManager.
    """
    mock_tm = Mock()
    
    # Create mock sources
    mock_source = Source(text="Test Course - Lesson 1", link="https://example.com/lesson1")
    mock_tm.get_last_sources.return_value = [mock_source]
    mock_tm.reset_sources.return_value = None
    
    mock_tm.get_tool_definitions.return_value = [
        {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course content"
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_course_outline",
                "description": "Get course outline"
            }
        }
    ]
    
    mock_tm.execute_tool.return_value = "Tool execution result"
    
    return mock_tm


@pytest.fixture
def mock_rag_system(mock_vector_store, mock_ai_generator, mock_session_manager, mock_tool_manager):
    """Fixture providing a fully mocked RAGSystem instance.
    
    This fixture creates a Mock object that simulates the RAGSystem
    with all its methods and dependencies properly mocked.
    
    Args:
        mock_vector_store: Injected mock VectorStore fixture.
        mock_ai_generator: Injected mock AIGenerator fixture.
        mock_session_manager: Injected mock SessionManager fixture.
        mock_tool_manager: Injected mock ToolManager fixture.
    
    Returns:
        Mock: Configured mock RAGSystem with all dependencies.
    """
    mock_rag = Mock()
    
    # Set up mock dependencies as attributes
    mock_rag.vector_store = mock_vector_store
    mock_rag.ai_generator = mock_ai_generator
    mock_rag.session_manager = mock_session_manager
    mock_rag.tool_manager = mock_tool_manager
    
    # Configure default return values for main methods
    mock_rag.query.return_value = (
        "This is a test answer from the RAG system.",
        [Source(text="Test Course - Lesson 1", link="https://example.com/lesson1")]
    )
    
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }
    
    mock_rag.add_course_document.return_value = (Mock(title="Test Course"), 5)
    mock_rag.add_course_folder.return_value = (2, 25)
    
    return mock_rag


@pytest.fixture
def test_client(mock_rag_system):
    """Fixture providing a FastAPI TestClient with mocked RAG dependencies.
    
    This fixture creates a test client for the FastAPI application with the
    RAGSystem fully mocked to avoid initialization overhead and external dependencies.
    
    Args:
        mock_rag_system: Injected mock RAGSystem fixture.
    
    Returns:
        TestClient: Configured test client for making API requests.
    """
    from fastapi.testclient import TestClient
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    from models import Source
    
    # Create minimal test app (avoids static file mounting issues)
    test_app = FastAPI(title="Course Materials RAG System - Test")
    
    # Add CORS middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    class DeleteSessionRequest(BaseModel):
        session_id: str
    
    class DeleteSessionResponse(BaseModel):
        success: bool
        message: str
    
    # API Endpoints with mocked RAG system
    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources."""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            source_items = [
                SourceItem(text=source.text, link=source.link) 
                for source in sources
            ]
            
            return QueryResponse(
                answer=answer,
                sources=source_items,
                session_id=session_id
            )
        except Exception as e:
            error_msg = str(e)
            if "Invalid Moonshot API Key" in error_msg or "Authentication" in error_msg:
                raise HTTPException(
                    status_code=401,
                    detail="API Key 无效。请在 .env 文件中设置正确的 MOONSHOT_API_KEY，然后重启服务器。获取 API Key: https://platform.moonshot.cn/"
                )
            raise HTTPException(status_code=500, detail=error_msg)
    
    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics."""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @test_app.post("/api/session/delete", response_model=DeleteSessionResponse)
    async def delete_session(request: DeleteSessionRequest):
        """Delete a conversation session."""
        try:
            success = mock_rag_system.session_manager.delete_session(request.session_id)
            if success:
                return DeleteSessionResponse(
                    success=True,
                    message=f"Session {request.session_id} deleted successfully"
                )
            else:
                return DeleteSessionResponse(
                    success=False,
                    message=f"Session {request.session_id} not found"
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @test_app.get("/")
    async def root():
        """Root endpoint for health check."""
        return {"message": "Course Materials RAG System API", "status": "ok"}
    
    client = TestClient(test_app)
    return client


@pytest.fixture
def sample_sources():
    """Fixture providing sample source objects for testing.
    
    Returns:
        List[Source]: List of sample Source objects.
    """
    return [
        Source(text="Course A - Lesson 1", link="https://example.com/a1"),
        Source(text="Course A - Lesson 2", link="https://example.com/a2"),
        Source(text="Course B - Lesson 1", link=None),
    ]


@pytest.fixture
def sample_course_analytics():
    """Fixture providing sample course analytics data.
    
    Returns:
        dict: Dictionary with total_courses and course_titles.
    """
    return {
        "total_courses": 3,
        "course_titles": ["Introduction to Python", "Advanced RAG", "Vector Databases"]
    }
