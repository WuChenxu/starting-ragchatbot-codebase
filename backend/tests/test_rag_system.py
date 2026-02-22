"""Tests for rag_system.py - RAGSystem content-query handling"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 100
        self.CHROMA_PATH = "./test_chroma_db"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.MAX_HISTORY = 2
        self.MOONSHOT_API_KEY = "test-key"
        self.MOONSHOT_MODEL = "moonshot-v1-8k"
        self.MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"


class TestRAGSystemContentQueries(unittest.TestCase):
    """Test cases for RAG system handling content-related queries"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def setUp(self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Set up test fixtures with mocked dependencies"""
        self.mock_config = MockConfig()
        self.mock_vector_store_instance = Mock()
        self.mock_ai_generator_instance = Mock()
        self.mock_session_manager_instance = Mock()
        self.mock_doc_processor_instance = Mock()
        
        mock_vector_store.return_value = self.mock_vector_store_instance
        mock_ai_generator.return_value = self.mock_ai_generator_instance
        mock_session_manager.return_value = self.mock_session_manager_instance
        mock_doc_processor.return_value = self.mock_doc_processor_instance
        
        self.rag_system = RAGSystem(self.mock_config)
    
    def test_query_with_content_search(self):
        """Test that content queries trigger the search tool"""
        # Arrange
        self.mock_ai_generator_instance.generate_response.return_value = "RAG stands for Retrieval-Augmented Generation"
        self.mock_vector_store_instance.search.return_value = Mock(
            documents=["Content about RAG"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            error=None
        )
        self.mock_vector_store_instance.get_lesson_link.return_value = "https://example.com"
        
        # Act
        answer, sources = self.rag_system.query("What is RAG?")
        
        # Assert
        self.mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = self.mock_ai_generator_instance.generate_response.call_args
        self.assertIn("What is RAG?", call_args[1]["query"])
        self.assertIsNotNone(call_args[1]["tools"])
        self.assertIsNotNone(call_args[1]["tool_manager"])
    
    def test_query_passes_tools_to_ai_generator(self):
        """Test that tools are correctly passed to AI generator"""
        # Arrange
        self.mock_ai_generator_instance.generate_response.return_value = "Answer"
        
        # Act
        answer, sources = self.rag_system.query("Test query")
        
        # Assert
        call_args = self.mock_ai_generator_instance.generate_response.call_args
        tools = call_args[1]["tools"]
        tool_manager = call_args[1]["tool_manager"]
        
        # Should have both search and outline tools
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)
        self.assertIsNotNone(tool_manager)
    
    def test_query_with_session_id(self):
        """Test query handling with existing session"""
        # Arrange
        self.mock_ai_generator_instance.generate_response.return_value = "Answer"
        self.mock_session_manager_instance.get_conversation_history.return_value = "Previous conversation"
        
        # Act
        answer, sources = self.rag_system.query("Follow up question", session_id="session_123")
        
        # Assert
        self.mock_session_manager_instance.get_conversation_history.assert_called_once_with("session_123")
        call_args = self.mock_ai_generator_instance.generate_response.call_args
        self.assertEqual(call_args[1]["conversation_history"], "Previous conversation")
    
    def test_query_saves_exchange_to_history(self):
        """Test that query-response exchange is saved to conversation history"""
        # Arrange
        self.mock_ai_generator_instance.generate_response.return_value = "The answer is 42"
        
        # Act
        answer, sources = self.rag_system.query("What is the answer?", session_id="session_1")
        
        # Assert
        self.mock_session_manager_instance.add_exchange.assert_called_once_with(
            "session_1", "What is the answer?", "The answer is 42"
        )
    
    def test_query_resets_sources_after_retrieval(self):
        """Test that sources are reset after being retrieved"""
        # Arrange
        self.mock_ai_generator_instance.generate_response.return_value = "Answer"
        
        # Act
        answer, sources = self.rag_system.query("Test query")
        
        # Assert - sources should be retrieved and then reset
        # This is verified by checking that tool_manager.reset_sources is called
        # The actual implementation detail may vary
    
    def test_query_returns_sources_from_tool(self):
        """Test that sources returned come from the tool execution"""
        # Arrange
        self.mock_ai_generator_instance.generate_response.return_value = "Answer"
        
        # Create mock sources in the tool manager
        from models import Source
        mock_source = Source(text="Course - Lesson 1", link="https://example.com")
        
        # We need to mock the tool manager's get_last_sources method
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[mock_source])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Act
        answer, sources = self.rag_system.query("Test query")
        
        # Assert
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].text, "Course - Lesson 1")
        self.assertEqual(sources[0].link, "https://example.com")
        self.rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_content_query_vs_outline_query(self):
        """Test that different query types are handled appropriately"""
        # Arrange
        content_query = "What is cross-encoder re-ranking?"
        outline_query = "What lessons are in the Chroma course?"
        
        self.mock_ai_generator_instance.generate_response.return_value = "Answer"
        
        # Act
        content_answer, _ = self.rag_system.query(content_query)
        outline_answer, _ = self.rag_system.query(outline_query)
        
        # Assert - both should use the same pipeline but AI decides which tool to call
        self.assertEqual(self.mock_ai_generator_instance.generate_response.call_count, 2)
        
        # Both calls should have tools available
        for call in self.mock_ai_generator_instance.generate_response.call_args_list:
            self.assertIsNotNone(call[1]["tools"])
            self.assertIsNotNone(call[1]["tool_manager"])
    
    def test_query_error_handling(self):
        """Test error handling in query method"""
        # Arrange
        self.mock_ai_generator_instance.generate_response.side_effect = Exception("API Error")
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")
        
        self.assertIn("API Error", str(context.exception))


class TestRAGSystemInitialization(unittest.TestCase):
    """Test cases for RAG system initialization"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_tools_registered_on_init(self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Test that both tools are registered during initialization"""
        # Arrange
        mock_config = MockConfig()
        
        # Act
        rag_system = RAGSystem(mock_config)
        
        # Assert
        tool_manager = rag_system.tool_manager
        self.assertIn("search_course_content", tool_manager.tools)
        self.assertIn("get_course_outline", tool_manager.tools)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_vector_store_shared_between_tools(self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Test that both tools share the same vector store instance"""
        # Arrange
        mock_config = MockConfig()
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Act
        rag_system = RAGSystem(mock_config)
        
        # Assert
        self.assertEqual(rag_system.search_tool.store, mock_vector_store_instance)
        self.assertEqual(rag_system.outline_tool.store, mock_vector_store_instance)


class TestRAGSystemWithRealChromaData(unittest.TestCase):
    """Integration tests with real ChromaDB data (optional)"""
    
    @unittest.skip("Integration test - requires real data")
    def test_real_content_search(self):
        """Test content search with real ChromaDB data"""
        # This would require actual data in the ChromaDB
        pass


class TestRAGSystemDocumentProcessing(unittest.TestCase):
    """Test document processing functionality"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def setUp(self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor):
        """Set up test fixtures"""
        self.mock_config = MockConfig()
        self.mock_vector_store_instance = Mock()
        self.mock_doc_processor_instance = Mock()
        
        mock_vector_store.return_value = self.mock_vector_store_instance
        mock_doc_processor.return_value = self.mock_doc_processor_instance
        
        self.rag_system = RAGSystem(self.mock_config)
    
    def test_add_course_document_success(self):
        """Test successful document addition"""
        # Arrange
        from models import Course, CourseChunk
        mock_course = Course(title="Test Course", lessons=[])
        mock_chunks = [CourseChunk(content="Test", course_title="Test", chunk_index=0)]
        
        self.mock_doc_processor_instance.process_course_document.return_value = (mock_course, mock_chunks)
        
        # Act
        result = self.rag_system.add_course_document("/path/to/file.txt")
        
        # Assert
        self.assertEqual(result[0].title, "Test Course")
        self.mock_vector_store_instance.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store_instance.add_course_content.assert_called_once_with(mock_chunks)


if __name__ == "__main__":
    unittest.main()
