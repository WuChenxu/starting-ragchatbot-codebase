"""Tests for MAX_RESULTS validation and edge cases"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults


class TestMaxResultsValidation(unittest.TestCase):
    """Test cases for MAX_RESULTS parameter validation"""
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.os.path.exists')
    def test_init_with_max_results_zero_raises_error(self, mock_exists, mock_client):
        """Test that VectorStore raises ValueError when MAX_RESULTS is 0"""
        # Arrange
        mock_exists.return_value = True
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            VectorStore(
                chroma_path="./test_db",
                embedding_model="all-MiniLM-L6-v2",
                max_results=0
            )
        
        error_msg = str(context.exception)
        self.assertIn("MAX_RESULTS must be a positive integer", error_msg)
        self.assertIn("got 0", error_msg)
        self.assertIn(">= 1", error_msg)
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.os.path.exists')
    def test_init_with_max_results_negative_raises_error(self, mock_exists, mock_client):
        """Test that VectorStore raises ValueError when MAX_RESULTS is negative"""
        # Arrange
        mock_exists.return_value = True
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            VectorStore(
                chroma_path="./test_db",
                embedding_model="all-MiniLM-L6-v2",
                max_results=-5
            )
        
        error_msg = str(context.exception)
        self.assertIn("MAX_RESULTS must be a positive integer", error_msg)
        self.assertIn("got -5", error_msg)
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.os.path.exists')
    def test_init_with_valid_max_results_succeeds(self, mock_exists, mock_client):
        """Test that VectorStore initializes successfully with valid MAX_RESULTS"""
        # Arrange
        mock_exists.return_value = True
        
        # Act
        store = VectorStore(
            chroma_path="./test_db",
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        # Assert
        self.assertEqual(store.max_results, 5)
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.os.path.exists')
    def test_init_with_max_results_one_succeeds(self, mock_exists, mock_client):
        """Test that VectorStore initializes successfully with MAX_RESULTS=1 (boundary)"""
        # Arrange
        mock_exists.return_value = True
        
        # Act
        store = VectorStore(
            chroma_path="./test_db",
            embedding_model="all-MiniLM-L6-v2",
            max_results=1
        )
        
        # Assert
        self.assertEqual(store.max_results, 1)


class TestSearchLimitValidation(unittest.TestCase):
    """Test cases for search limit parameter validation in search() method"""
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.os.path.exists')
    def setUp(self, mock_exists, mock_client):
        """Set up test fixtures"""
        mock_exists.return_value = True
        
        # Create a mock for the embedding function
        with patch('vector_store.LocalSentenceTransformerEmbeddingFunction'):
            self.store = VectorStore(
                chroma_path="./test_db",
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )
    
    def test_search_with_limit_zero_returns_error(self):
        """Test that search() returns error when limit is 0"""
        # Act
        results = self.store.search(query="test", limit=0)
        
        # Assert
        self.assertIsNotNone(results.error)
        self.assertIn("Invalid search limit", results.error)
        self.assertIn("must be a positive integer", results.error)
        self.assertEqual(len(results.documents), 0)
    
    def test_search_with_limit_negative_returns_error(self):
        """Test that search() returns error when limit is negative"""
        # Act
        results = self.store.search(query="test", limit=-3)
        
        # Assert
        self.assertIsNotNone(results.error)
        self.assertIn("Invalid search limit", results.error)
        self.assertIn("-3", results.error)
        self.assertEqual(len(results.documents), 0)
    
    def test_search_with_valid_limit_succeeds(self):
        """Test that search() works with valid limit"""
        # Arrange
        self.store.course_content = Mock()
        self.store.course_content.query.return_value = {
            'documents': [['result1', 'result2']],
            'metadatas': [[{'course_title': 'Test', 'lesson_number': 1}, {'course_title': 'Test', 'lesson_number': 2}]],
            'distances': [[0.1, 0.2]]
        }
        
        # Act
        results = self.store.search(query="test", limit=3)
        
        # Assert
        self.assertIsNone(results.error)
        self.assertEqual(len(results.documents), 2)
        # Verify n_results was passed correctly
        call_args = self.store.course_content.query.call_args
        self.assertEqual(call_args[1]['n_results'], 3)


class TestRAGSystemWithInvalidMaxResults(unittest.TestCase):
    """Test RAGSystem behavior with invalid MAX_RESULTS configuration"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_rag_system_fails_gracefully_with_max_results_zero(self, mock_session_manager, 
                                                               mock_ai_generator, mock_vector_store, 
                                                               mock_doc_processor):
        """Test that RAGSystem initialization fails gracefully with MAX_RESULTS=0"""
        # Arrange
        from rag_system import RAGSystem
        
        mock_config = Mock()
        mock_config.MAX_RESULTS = 0  # Invalid
        mock_config.CHROMA_PATH = "./test_db"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        
        mock_vector_store.side_effect = ValueError("MAX_RESULTS must be a positive integer")
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RAGSystem(mock_config)
        
        self.assertIn("MAX_RESULTS", str(context.exception))


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def test_config_default_max_results_is_valid(self):
        """Test that default MAX_RESULTS in config is valid"""
        from config import Config
        
        # Act
        config = Config()
        
        # Assert
        self.assertGreater(config.MAX_RESULTS, 0)


if __name__ == "__main__":
    unittest.main()
