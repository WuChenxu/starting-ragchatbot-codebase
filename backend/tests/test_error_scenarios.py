"""Tests for error scenarios that could cause 'query failed'"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQueryFailureScenarios(unittest.TestCase):
    """Test specific scenarios that could cause 'query failed' errors"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def setUp(self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Set up test fixtures"""
        from rag_system import RAGSystem
        
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_db"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.MAX_HISTORY = 2
        mock_config.MOONSHOT_API_KEY = "test"
        mock_config.MOONSHOT_MODEL = "moonshot-v1-8k"
        mock_config.MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
        
        mock_vector_store_instance = Mock()
        mock_ai_instance = Mock()
        mock_session_instance = Mock()
        
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ai_generator.return_value = mock_ai_instance
        mock_session_manager.return_value = mock_session_instance
        
        self.rag_system = RAGSystem(mock_config)
        self.mock_ai = mock_ai_instance
        self.mock_session = mock_session_instance
    
    def test_query_fails_when_ai_returns_none(self):
        """Test failure when AI returns None instead of string"""
        # Arrange
        self.mock_ai.generate_response.return_value = None
        
        # Act
        answer, sources = self.rag_system.query("Test query")
        
        # Assert - should handle None gracefully
        self.assertIsNone(answer)
    
    def test_query_fails_when_ai_raises_exception(self):
        """Test failure when AI raises an exception"""
        # Arrange
        self.mock_ai.generate_response.side_effect = Exception("AI service error")
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")
        
        self.assertIn("AI service error", str(context.exception))
    
    def test_query_fails_when_session_manager_fails(self):
        """Test failure when session manager fails"""
        # Arrange
        self.mock_session.add_exchange.side_effect = Exception("Session error")
        self.mock_ai.generate_response.return_value = "Answer"
        
        # Act & Assert
        with self.assertRaises(Exception):
            self.rag_system.query("Test", session_id="session_1")
    
    def test_tool_execution_fails(self):
        """Test failure when tool execution fails - exception is now caught"""
        # Arrange
        from search_tools import ToolManager, CourseSearchTool
        
        mock_store = Mock()
        mock_store.search.side_effect = Exception("Database error")
        
        tool = CourseSearchTool(mock_store)
        
        # Act - with new error handling, exception is caught
        result = tool.execute(query="test")
        
        # Assert - should return error message instead of raising
        self.assertIn("Search failed", result)
        self.assertIn("Database error", result)
    
    def test_empty_tool_definitions(self):
        """Test behavior when no tools are registered"""
        # Arrange
        from search_tools import ToolManager
        
        tool_manager = ToolManager()
        
        # Act
        definitions = tool_manager.get_tool_definitions()
        
        # Assert
        self.assertEqual(len(definitions), 0)
        
        # Test that execute_tool returns appropriate error
        result = tool_manager.execute_tool("nonexistent_tool")
        self.assertEqual(result, "Tool 'nonexistent_tool' not found")
    
    def test_tool_manager_get_last_sources_empty(self):
        """Test get_last_sources when no tools have sources"""
        # Arrange
        from search_tools import ToolManager, CourseSearchTool
        
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)
        tool.last_sources = []  # Empty sources
        
        tool_manager = ToolManager()
        tool_manager.register_tool(tool)
        
        # Act
        sources = tool_manager.get_last_sources()
        
        # Assert
        self.assertEqual(len(sources), 0)


class TestAPIErrorHandling(unittest.TestCase):
    """Test error handling at the API level"""
    
    def test_http_exception_on_ai_error(self):
        """Test that API returns proper HTTP exception on AI error"""
        # This would test the app.py error handling
        pass


class TestContentQuerySpecificIssues(unittest.TestCase):
    """Test specific issues with content queries"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_content_query_with_special_characters(self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Test content query with special characters that might cause issues"""
        from rag_system import RAGSystem
        
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_db"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.MAX_HISTORY = 2
        mock_config.MOONSHOT_API_KEY = "test"
        mock_config.MOONSHOT_MODEL = "moonshot-v1-8k"
        mock_config.MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
        
        mock_vector_store.return_value = Mock()
        mock_ai_generator.return_value = Mock()
        mock_ai_generator.return_value.generate_response.return_value = "Answer"
        mock_session_manager.return_value = Mock()
        
        rag = RAGSystem(mock_config)
        
        # Test queries with special characters
        special_queries = [
            "What is RAG? (Really Advanced Generation)",
            "Tell me about embeddings & vectors",
            "Query with 'quotes' and \"double quotes\"",
            "Question about [brackets] and {braces}",
        ]
        
        for query in special_queries:
            # Act & Assert - should not raise exception
            try:
                answer, sources = rag.query(query)
            except Exception as e:
                self.fail(f"Query '{query}' raised exception: {e}")
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_content_query_with_very_long_text(self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Test content query with very long text"""
        from rag_system import RAGSystem
        
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_db"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.MAX_HISTORY = 2
        mock_config.MOONSHOT_API_KEY = "test"
        mock_config.MOONSHOT_MODEL = "moonshot-v1-8k"
        mock_config.MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
        
        mock_vector_store.return_value = Mock()
        mock_ai_generator.return_value = Mock()
        mock_ai_generator.return_value.generate_response.return_value = "Answer"
        mock_session_manager.return_value = Mock()
        
        rag = RAGSystem(mock_config)
        
        # Very long query
        long_query = "What is " + "RAG " * 1000 + "?"
        
        # Act & Assert - should not raise exception
        try:
            answer, sources = rag.query(long_query)
        except Exception as e:
            self.fail(f"Long query raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
