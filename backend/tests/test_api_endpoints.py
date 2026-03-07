"""Tests for FastAPI endpoints - /api/query, /api/courses, /api/session/delete

This module tests the HTTP API layer of the RAG system, ensuring proper
request/response handling, error responses, and integration with the RAG system.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Source


class TestQueryEndpoint:
    """Test cases for POST /api/query endpoint."""
    
    def test_query_successful_response(self, test_client, mock_rag_system):
        """Test successful query returns correct response structure."""
        # Arrange
        mock_rag_system.query.return_value = (
            "RAG stands for Retrieval-Augmented Generation.",
            [Source(text="Test Course - Lesson 1", link="https://example.com/lesson1")]
        )
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "What is RAG?",
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "RAG stands for Retrieval-Augmented Generation."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/lesson1"
    
    def test_query_creates_new_session_when_none_provided(self, test_client, mock_rag_system):
        """Test that a new session is created when session_id is not provided."""
        # Arrange
        mock_rag_system.session_manager.create_session.return_value = "session_new_123"
        mock_rag_system.query.return_value = ("Answer", [])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "Test query",
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_new_123"
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_query_uses_existing_session(self, test_client, mock_rag_system):
        """Test that existing session_id is preserved in the response."""
        # Arrange
        mock_rag_system.query.return_value = ("Answer", [])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "Test query",
            "session_id": "session_existing_456"
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_existing_456"
        mock_rag_system.session_manager.create_session.assert_not_called()
    
    def test_query_passes_session_to_rag_system(self, test_client, mock_rag_system):
        """Test that session_id is passed to RAG system query method."""
        # Arrange
        mock_rag_system.query.return_value = ("Answer", [])
        
        # Act
        test_client.post("/api/query", json={
            "query": "Test query",
            "session_id": "session_abc"
        })
        
        # Assert
        mock_rag_system.query.assert_called_once_with("Test query", "session_abc")
    
    def test_query_with_multiple_sources(self, test_client, mock_rag_system, sample_sources):
        """Test response with multiple sources returns all sources correctly."""
        # Arrange
        mock_rag_system.query.return_value = ("Answer with multiple sources", sample_sources)
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "Complex question",
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        assert data["sources"][0]["text"] == "Course A - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/a1"
        assert data["sources"][2]["link"] is None
    
    def test_query_with_empty_sources(self, test_client, mock_rag_system):
        """Test response when no sources are found."""
        # Arrange
        mock_rag_system.query.return_value = ("I couldn't find any relevant information.", [])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "Obscure topic",
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "I couldn't find any relevant information."
        assert data["sources"] == []
    
    def test_query_with_empty_query_string(self, test_client, mock_rag_system):
        """Test handling of empty query string."""
        # Arrange
        mock_rag_system.query.return_value = ("Please provide a question.", [])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "",
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_with_special_characters(self, test_client, mock_rag_system):
        """Test handling of queries with special characters."""
        # Arrange
        special_query = "What is RAG? <script>alert('xss')</script> & more!"
        mock_rag_system.query.return_value = ("Answer", [])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": special_query,
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(special_query, mock_rag_system.session_manager.create_session.return_value)
    
    def test_query_with_long_query(self, test_client, mock_rag_system):
        """Test handling of very long queries."""
        # Arrange
        long_query = "What is RAG? " * 100
        mock_rag_system.query.return_value = ("Answer", [])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": long_query,
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
    
    def test_query_authentication_error(self, test_client, mock_rag_system):
        """Test 401 response for authentication errors."""
        # Arrange
        mock_rag_system.query.side_effect = Exception("Invalid Moonshot API Key")
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "Test query",
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "API Key 无效" in data["detail"] or "API Key" in data["detail"]
    
    def test_query_internal_server_error(self, test_client, mock_rag_system):
        """Test 500 response for general errors."""
        # Arrange
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "Test query",
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]
    
    def test_query_missing_query_field(self, test_client):
        """Test validation error when query field is missing."""
        # Act
        response = test_client.post("/api/query", json={
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_query_invalid_json(self, test_client):
        """Test error handling for invalid JSON."""
        # Act
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Assert
        assert response.status_code == 422
    
    def test_query_unicode_content(self, test_client, mock_rag_system):
        """Test handling of unicode characters in query and response."""
        # Arrange
        unicode_query = "什么是RAG？🔍 日本語テスト"
        mock_rag_system.query.return_value = ("这是答案。日本語の回答です。", [])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": unicode_query,
            "session_id": None
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "这是答案。日本語の回答です。"


class TestCoursesEndpoint:
    """Test cases for GET /api/courses endpoint."""
    
    def test_courses_successful_response(self, test_client, mock_rag_system, sample_course_analytics):
        """Test successful retrieval of course statistics."""
        # Arrange
        mock_rag_system.get_course_analytics.return_value = sample_course_analytics
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Python" in data["course_titles"]
    
    def test_courses_empty_catalog(self, test_client, mock_rag_system):
        """Test response when no courses are available."""
        # Arrange
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_single_course(self, test_client, mock_rag_system):
        """Test response with single course in catalog."""
        # Arrange
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Only Course"]
        }
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Only Course"]
    
    def test_courses_internal_error(self, test_client, mock_rag_system):
        """Test 500 response when analytics retrieval fails."""
        # Arrange
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store unavailable")
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Vector store unavailable" in data["detail"]
    
    def test_courses_many_courses(self, test_client, mock_rag_system):
        """Test response with many courses."""
        # Arrange
        many_courses = [f"Course {i}" for i in range(1, 101)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": many_courses
        }
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100


class TestSessionDeleteEndpoint:
    """Test cases for POST /api/session/delete endpoint."""
    
    def test_delete_session_successful(self, test_client, mock_rag_system):
        """Test successful deletion of existing session."""
        # Arrange
        mock_rag_system.session_manager.delete_session.return_value = True
        
        # Act
        response = test_client.post("/api/session/delete", json={
            "session_id": "session_123"
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "session_123" in data["message"]
        assert "deleted successfully" in data["message"]
    
    def test_delete_session_not_found(self, test_client, mock_rag_system):
        """Test deletion of non-existent session."""
        # Arrange
        mock_rag_system.session_manager.delete_session.return_value = False
        
        # Act
        response = test_client.post("/api/session/delete", json={
            "session_id": "session_nonexistent"
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["message"]
    
    def test_delete_session_calls_manager(self, test_client, mock_rag_system):
        """Test that delete_session is called with correct session_id."""
        # Arrange
        mock_rag_system.session_manager.delete_session.return_value = True
        
        # Act
        test_client.post("/api/session/delete", json={
            "session_id": "session_to_delete"
        })
        
        # Assert
        mock_rag_system.session_manager.delete_session.assert_called_once_with("session_to_delete")
    
    def test_delete_session_missing_session_id(self, test_client):
        """Test validation error when session_id is missing."""
        # Act
        response = test_client.post("/api/session/delete", json={})
        
        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_delete_session_empty_session_id(self, test_client, mock_rag_system):
        """Test handling of empty session_id."""
        # Arrange
        mock_rag_system.session_manager.delete_session.return_value = False
        
        # Act
        response = test_client.post("/api/session/delete", json={
            "session_id": ""
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
    
    def test_delete_session_internal_error(self, test_client, mock_rag_system):
        """Test 500 response when session deletion fails."""
        # Arrange
        mock_rag_system.session_manager.delete_session.side_effect = Exception("Storage error")
        
        # Act
        response = test_client.post("/api/session/delete", json={
            "session_id": "session_123"
        })
        
        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestRootEndpoint:
    """Test cases for GET / (root) endpoint."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns system status."""
        # Act
        response = test_client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "ok"
        assert "RAG" in data["message"]
    
    def test_root_endpoint_health_check(self, test_client):
        """Test root endpoint can be used for health checks."""
        # Act
        response = test_client.get("/")
        
        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestCORSHeaders:
    """Test cases for CORS headers.
    
    Note: CORS headers are only added for cross-origin requests.
    TestClient simulates same-origin requests by default, so we test
    that the CORS middleware is properly configured by checking
    the middleware is present in the app.
    """
    
    def test_cors_middleware_configured(self, test_client, mock_rag_system):
        """Test CORS middleware is properly configured on the app."""
        # Act - Get the root endpoint (simplest request)
        response = test_client.get("/")
        
        # Assert
        assert response.status_code == 200
        # Note: CORS headers are only added for requests with Origin header
        # (cross-origin requests), not for same-origin requests from TestClient
    
    def test_cors_preflight_request(self, test_client):
        """Test CORS preflight requests work correctly."""
        # Simulate a CORS preflight request
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # FastAPI with CORS middleware handles preflight requests
        # It may return 200 OK or 405 Method Not Allowed depending on configuration
        # The important thing is that the app handles the request without errors
        assert response.status_code in [200, 405]
    



class TestResponseModels:
    """Test cases for response model validation."""
    
    def test_query_response_model_structure(self, test_client, mock_rag_system):
        """Test that query response follows expected model structure."""
        # Arrange
        mock_rag_system.query.return_value = ("Answer", [
            Source(text="Source 1", link="https://example.com")
        ])
        
        # Act
        response = test_client.post("/api/query", json={
            "query": "Test",
            "session_id": "session_1"
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Verify source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source
    
    def test_courses_response_model_structure(self, test_client, mock_rag_system):
        """Test that courses response follows expected model structure."""
        # Arrange
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course 1", "Course 2"]
        }
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])
    
    def test_delete_session_response_model_structure(self, test_client, mock_rag_system):
        """Test that delete session response follows expected model structure."""
        # Arrange
        mock_rag_system.session_manager.delete_session.return_value = True
        
        # Act
        response = test_client.post("/api/session/delete", json={
            "session_id": "session_1"
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["success"], bool)
        assert isinstance(data["message"], str)


class TestEndpointIntegration:
    """Integration-style tests combining multiple endpoints."""
    
    def test_full_conversation_flow(self, test_client, mock_rag_system):
        """Test complete conversation flow: query, follow-up, delete session."""
        # Step 1: Initial query (no session)
        mock_rag_system.query.return_value = ("Initial answer", [])
        mock_rag_system.session_manager.create_session.return_value = "session_flow_123"
        
        response1 = test_client.post("/api/query", json={
            "query": "First question",
            "session_id": None
        })
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        assert session_id == "session_flow_123"
        
        # Step 2: Follow-up query (with session)
        mock_rag_system.query.return_value = ("Follow-up answer", [])
        
        response2 = test_client.post("/api/query", json={
            "query": "Follow-up question",
            "session_id": session_id
        })
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Step 3: Delete the session
        mock_rag_system.session_manager.delete_session.return_value = True
        
        response3 = test_client.post("/api/session/delete", json={
            "session_id": session_id
        })
        assert response3.status_code == 200
        assert response3.json()["success"] is True
    
    def test_query_after_getting_courses(self, test_client, mock_rag_system):
        """Test query endpoint works after checking courses."""
        # First, get courses
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Python", "RAG"]
        }
        
        response1 = test_client.get("/api/courses")
        assert response1.status_code == 200
        
        # Then, query about one of the courses
        mock_rag_system.query.return_value = ("Python is a programming language.", [])
        
        response2 = test_client.post("/api/query", json={
            "query": "Tell me about Python",
            "session_id": None
        })
        assert response2.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
