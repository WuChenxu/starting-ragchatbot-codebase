"""Tests for search_tools.py - CourseSearchTool and CourseOutlineTool"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Source
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool.execute method"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_with_successful_search(self):
        """Test execute method returns formatted results on successful search"""
        # Arrange
        mock_results = SearchResults(
            documents=["This is content about RAG systems", "Another document about embeddings"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Act
        result = self.search_tool.execute(query="RAG systems", course_name="Test")

        # Assert
        self.assertIn("[Test Course - Lesson 1]", result)
        self.assertIn("This is content about RAG systems", result)
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertEqual(self.search_tool.last_sources[0].text, "Test Course - Lesson 1")
        self.assertEqual(self.search_tool.last_sources[0].link, "https://example.com/lesson1")

    def test_execute_with_empty_results(self):
        """Test execute method handles empty results gracefully"""
        # Arrange
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(query="nonexistent topic")

        # Assert
        self.assertIn("No relevant content found", result)
        self.assertEqual(len(self.search_tool.last_sources), 0)

    def test_execute_with_error(self):
        """Test execute method handles search errors from results"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(query="test query")

        # Assert - now wrapped with "Search error:" prefix
        self.assertIn("Search error", result)
        self.assertIn("Database connection failed", result)

    def test_execute_with_course_filter(self):
        """Test execute method passes course filter correctly"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.search_tool.execute(query="MCP architecture", course_name="MCP")

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="MCP architecture", course_name="MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self):
        """Test execute method passes lesson filter correctly"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.search_tool.execute(query="test", course_name="Test", lesson_number=3)

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="test", course_name="Test", lesson_number=3
        )

    def test_execute_preserves_source_links(self):
        """Test that source links are correctly preserved for UI"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course A", "lesson_number": 5}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://learn.example.com/lesson5"

        # Act
        result = self.search_tool.execute(query="test")

        # Assert
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(self.search_tool.last_sources[0].link, "https://learn.example.com/lesson5")

    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted for OpenAI"""
        # Act
        definition = self.search_tool.get_tool_definition()

        # Assert
        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "search_course_content")
        self.assertIn("query", definition["function"]["parameters"]["properties"])
        self.assertIn("course_name", definition["function"]["parameters"]["properties"])
        self.assertIn("lesson_number", definition["function"]["parameters"]["properties"])
        self.assertEqual(definition["function"]["parameters"]["required"], ["query"])

    def test_execute_catches_exceptions(self):
        """Test execute method catches and returns exceptions gracefully"""
        # Arrange
        self.mock_vector_store.search.side_effect = Exception("Connection timeout")

        # Act
        result = self.search_tool.execute(query="test")

        # Assert
        self.assertIn("Search failed", result)
        self.assertIn("Connection timeout", result)


class TestCourseOutlineTool(unittest.TestCase):
    """Test cases for CourseOutlineTool.execute method"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.outline_tool = CourseOutlineTool(self.mock_vector_store)

    def test_execute_successful_outline_retrieval(self):
        """Test successful course outline retrieval"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = "MCP Course"
        self.mock_vector_store.get_course_link.return_value = "https://example.com/mcp"
        self.mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "MCP Course",
                "course_link": "https://example.com/mcp",
                "lessons": [
                    {
                        "lesson_number": 1,
                        "lesson_title": "Introduction",
                        "lesson_link": "https://example.com/l1",
                    },
                    {
                        "lesson_number": 2,
                        "lesson_title": "Architecture",
                        "lesson_link": "https://example.com/l2",
                    },
                ],
            }
        ]

        # Act
        result = self.outline_tool.execute(course_name="MCP")

        # Assert
        self.assertIn("Course Title: MCP Course", result)
        self.assertIn("Course Link: https://example.com/mcp", result)
        self.assertIn("Total Lessons: 2", result)
        self.assertIn("Lesson 1: Introduction", result)
        self.assertIn("Lesson 2: Architecture", result)

    def test_execute_course_not_found(self):
        """Test handling when course is not found"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = None

        # Act
        result = self.outline_tool.execute(course_name="NonExistent")

        # Assert
        self.assertIn("No course found matching 'NonExistent'", result)

    def test_execute_catches_exceptions(self):
        """Test execute method catches and returns exceptions gracefully"""
        # Arrange
        self.mock_vector_store._resolve_course_name.side_effect = Exception("Database error")

        # Act
        result = self.outline_tool.execute(course_name="Test")

        # Assert
        self.assertIn("Failed to retrieve course outline", result)
        self.assertIn("Database error", result)

    def test_execute_sources_tracking(self):
        """Test that outline tool tracks sources correctly"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = "Test Course"
        self.mock_vector_store.get_course_link.return_value = "https://example.com/course"
        self.mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Test Course",
                "lessons": [
                    {
                        "lesson_number": 1,
                        "lesson_title": "Lesson 1",
                        "lesson_link": "https://example.com/l1",
                    }
                ],
            }
        ]

        # Act
        result = self.outline_tool.execute(course_name="Test")

        # Assert
        self.assertEqual(len(self.outline_tool.last_sources), 2)  # Course + 1 lesson
        self.assertEqual(self.outline_tool.last_sources[0].text, "Test Course")
        self.assertEqual(self.outline_tool.last_sources[0].link, "https://example.com/course")


class TestToolManager(unittest.TestCase):
    """Test cases for ToolManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_tool = Mock()
        self.mock_tool.get_tool_definition.return_value = {
            "type": "function",
            "function": {"name": "test_tool"},
        }
        self.mock_tool.execute.return_value = "test result"
        self.mock_tool.last_sources = []

    def test_register_tool(self):
        """Test tool registration"""
        # Act
        self.tool_manager.register_tool(self.mock_tool)

        # Assert
        self.assertIn("test_tool", self.tool_manager.tools)

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        definitions = self.tool_manager.get_tool_definitions()

        # Assert
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["function"]["name"], "test_tool")

    def test_execute_tool(self):
        """Test executing a registered tool"""
        # Arrange
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        result = self.tool_manager.execute_tool("test_tool", param="value")

        # Assert
        self.mock_tool.execute.assert_called_once_with(param="value")
        self.assertEqual(result, "test result")

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        # Act
        result = self.tool_manager.execute_tool("nonexistent")

        # Assert
        self.assertEqual(result, "Tool 'nonexistent' not found")

    def test_get_last_sources(self):
        """Test retrieving sources from tools"""
        # Arrange
        mock_source = Source(text="Test", link="https://example.com")
        self.mock_tool.last_sources = [mock_source]
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        sources = self.tool_manager.get_last_sources()

        # Assert
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].text, "Test")

    def test_reset_sources(self):
        """Test resetting sources on all tools"""
        # Arrange
        self.mock_tool.last_sources = [Source(text="Test", link=None)]
        self.tool_manager.register_tool(self.mock_tool)

        # Act
        self.tool_manager.reset_sources()

        # Assert
        self.assertEqual(len(self.mock_tool.last_sources), 0)


if __name__ == "__main__":
    unittest.main()
