"""Integration tests for the complete RAG pipeline"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCompleteQueryPipeline(unittest.TestCase):
    """End-to-end tests for the complete query pipeline"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_content_query_full_pipeline(
        self, mock_session_manager, mock_ai_generator, mock_vector_store, mock_doc_processor
    ):
        """Test complete pipeline for a content query"""
        # This test simulates the full flow from query to response
        from models import Source
        from rag_system import RAGSystem

        # Setup mocks
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

        # Create RAG system
        rag = RAGSystem(mock_config)

        # Mock AI response
        mock_ai_instance.generate_response.return_value = (
            "Cross-encoder re-ranking is a technique..."
        )

        # Mock sources from tool
        mock_source = Source(text="Chroma Course - Lesson 4", link="https://example.com")
        rag.tool_manager.get_last_sources = Mock(return_value=[mock_source])

        # Execute query
        answer, sources = rag.query("What is cross-encoder re-ranking?")

        # Verify
        self.assertEqual(answer, "Cross-encoder re-ranking is a technique...")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].text, "Chroma Course - Lesson 4")

        # Verify tools were passed
        call_args = mock_ai_instance.generate_response.call_args
        self.assertIn("tools", call_args[1])
        self.assertIn("tool_manager", call_args[1])


class TestErrorScenarios(unittest.TestCase):
    """Test error handling scenarios"""

    def test_vector_store_connection_error(self):
        """Test handling of vector store connection errors"""
        # Test that appropriate errors are propagated
        pass

    def test_ai_service_unavailable(self):
        """Test handling when AI service is unavailable"""
        pass


if __name__ == "__main__":
    unittest.main()
