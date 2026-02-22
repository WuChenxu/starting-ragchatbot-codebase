"""Tests for ai_generator.py - AIGenerator and tool calling"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class MockToolCall:
    """Mock class for tool calls"""
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = Mock()
        self.function.name = name
        self.function.arguments = json.dumps(arguments)


class MockMessage:
    """Mock class for chat completion message"""
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock class for completion choice"""
    def __init__(self, message):
        self.message = message


class MockCompletion:
    """Mock class for chat completion"""
    def __init__(self, choices):
        self.choices = choices


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.model = "moonshot-v1-8k"
        self.base_url = "https://api.moonshot.cn/v1"
        
        with patch('ai_generator.OpenAI') as mock_openai:
            self.mock_client = Mock()
            mock_openai.return_value = self.mock_client
            self.generator = AIGenerator(self.api_key, self.model, self.base_url)
    
    def test_init(self):
        """Test AIGenerator initialization"""
        # Assert
        self.assertEqual(self.generator.model, self.model)
        self.assertEqual(self.generator.base_params["model"], self.model)
        self.assertEqual(self.generator.base_params["temperature"], 0)
        self.assertEqual(self.generator.base_params["max_tokens"], 800)
    
    def test_generate_response_without_tools(self):
        """Test generate_response without tool usage"""
        # Arrange
        mock_message = MockMessage(content="This is the answer")
        mock_completion = MockCompletion([MockChoice(mock_message)])
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Act
        result = self.generator.generate_response("What is RAG?")
        
        # Assert
        self.assertEqual(result, "This is the answer")
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["messages"][0]["role"], "system")
        self.assertEqual(call_args[1]["messages"][1]["role"], "user")
        self.assertIn("What is RAG?", call_args[1]["messages"][1]["content"])
    
    def test_generate_response_with_conversation_history(self):
        """Test generate_response with conversation history"""
        # Arrange
        mock_message = MockMessage(content="Answer with history")
        mock_completion = MockCompletion([MockChoice(mock_message)])
        self.mock_client.chat.completions.create.return_value = mock_completion
        history = "User: Previous question\nAssistant: Previous answer"
        
        # Act
        result = self.generator.generate_response("Follow up question", conversation_history=history)
        
        # Assert
        call_args = self.mock_client.chat.completions.create.call_args
        self.assertIn("Previous conversation:", call_args[1]["messages"][0]["content"])
        self.assertIn(history, call_args[1]["messages"][0]["content"])
    
    def test_generate_response_with_tool_call(self):
        """Test generate_response when AI decides to call a tool"""
        # Arrange
        # First response - AI wants to call tool
        tool_call = MockToolCall("call_1", "search_course_content", {"query": "RAG", "course_name": "MCP"})
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])
        
        # Second response - final answer after tool execution
        second_message = MockMessage(content="RAG stands for Retrieval-Augmented Generation")
        second_completion = MockCompletion([MockChoice(second_message)])
        
        self.mock_client.chat.completions.create.side_effect = [first_completion, second_completion]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about RAG"
        
        # Define tools
        tools = [{
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course content"
            }
        }]
        
        # Act
        result = self.generator.generate_response(
            "What is RAG?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "RAG stands for Retrieval-Augmented Generation")
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="RAG", course_name="MCP")
    
    def test_generate_response_tool_not_found(self):
        """Test generate_response handles tool not found error gracefully"""
        # Arrange
        tool_call = MockToolCall("call_1", "nonexistent_tool", {})
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])
        
        second_message = MockMessage(content="Tool not available")
        second_completion = MockCompletion([MockChoice(second_message)])
        
        self.mock_client.chat.completions.create.side_effect = [first_completion, second_completion]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool 'nonexistent_tool' not found"
        
        # Act
        result = self.generator.generate_response(
            "Test query",
            tools=[],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Tool not available")
    
    def test_generate_response_api_error(self):
        """Test generate_response handles API errors"""
        # Arrange
        from openai import AuthenticationError
        self.mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=Mock(),
            body=None
        )
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            self.generator.generate_response("Test query")
        
        self.assertIn("Invalid Moonshot API Key", str(context.exception))
    
    def test_system_prompt_contains_tool_guidelines(self):
        """Test that system prompt includes tool usage guidelines"""
        # Arrange
        mock_message = MockMessage(content="Test")
        mock_completion = MockCompletion([MockChoice(mock_message)])
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Act
        self.generator.generate_response("Test")
        
        # Assert
        call_args = self.mock_client.chat.completions.create.call_args
        system_content = call_args[1]["messages"][0]["content"]
        
        self.assertIn("search_course_content", system_content)
        self.assertIn("get_course_outline", system_content)
        self.assertIn("Tool Usage Guidelines", system_content)
    
    def test_handle_tool_execution_multiple_tools(self):
        """Test _handle_tool_execution with multiple tool calls"""
        # Arrange
        tool_call_1 = MockToolCall("call_1", "search_course_content", {"query": "topic1"})
        tool_call_2 = MockToolCall("call_2", "get_course_outline", {"course_name": "MCP"})
        
        first_message = MockMessage(content=None, tool_calls=[tool_call_1, tool_call_2])
        first_completion = MockCompletion([MockChoice(first_message)])
        
        second_message = MockMessage(content="Combined answer")
        second_completion = MockCompletion([MockChoice(second_message)])
        
        self.mock_client.chat.completions.create.side_effect = [first_completion, second_completion]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        # Act
        result = self.generator.generate_response(
            "Test query",
            tools=[],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Combined answer")
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
    
    def test_tools_passed_to_api(self):
        """Test that tools are correctly passed to the API"""
        # Arrange
        mock_message = MockMessage(content="Test")
        mock_completion = MockCompletion([MockChoice(mock_message)])
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        tools = [
            {"type": "function", "function": {"name": "tool1"}},
            {"type": "function", "function": {"name": "tool2"}}
        ]
        
        # Act
        self.generator.generate_response("Test", tools=tools)
        
        # Assert
        call_args = self.mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["tools"], tools)
        self.assertEqual(call_args[1]["tool_choice"], "auto")


class TestAIGeneratorCourseSearchToolIntegration(unittest.TestCase):
    """Integration tests for AI generator calling CourseSearchTool"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.model = "moonshot-v1-8k"
        
        with patch('ai_generator.OpenAI') as mock_openai:
            self.mock_client = Mock()
            mock_openai.return_value = self.mock_client
            self.generator = AIGenerator(self.api_key, self.model)
    
    def test_content_query_triggers_search_tool(self):
        """Test that content-related queries trigger search_course_content tool"""
        # Arrange
        tool_call = MockToolCall(
            "call_1", 
            "search_course_content", 
            {"query": "embeddings", "course_name": "Advanced Retrieval"}
        )
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])
        
        second_message = MockMessage(content="Embeddings are vector representations...")
        second_completion = MockCompletion([MockChoice(second_message)])
        
        self.mock_client.chat.completions.create.side_effect = [first_completion, second_completion]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "[Course - Lesson 1]\nEmbeddings are..."
        
        # Define search tool
        tools = [{
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials"
            }
        }]
        
        # Act
        result = self.generator.generate_response(
            "What are embeddings in the Advanced Retrieval course?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        mock_tool_manager.execute_tool.assert_called_once()
        call_args = mock_tool_manager.execute_tool.call_args
        self.assertEqual(call_args[0][0], "search_course_content")
        # Should pass query parameter
        self.assertIn("query", call_args[1])
    
    def test_outline_query_triggers_outline_tool(self):
        """Test that outline-related queries trigger get_course_outline tool"""
        # Arrange
        tool_call = MockToolCall(
            "call_1", 
            "get_course_outline", 
            {"course_name": "MCP"}
        )
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])
        
        second_message = MockMessage(content="The MCP course has 11 lessons...")
        second_completion = MockCompletion([MockChoice(second_message)])
        
        self.mock_client.chat.completions.create.side_effect = [first_completion, second_completion]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course Title: MCP..."
        
        tools = [{
            "type": "function",
            "function": {
                "name": "get_course_outline",
                "description": "Get course outline"
            }
        }]
        
        # Act
        result = self.generator.generate_response(
            "What lessons are in the MCP course?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        mock_tool_manager.execute_tool.assert_called_once_with("get_course_outline", course_name="MCP")
    
    def test_tool_results_included_in_final_prompt(self):
        """Test that tool execution results are included in the final prompt to AI"""
        # Arrange
        tool_call = MockToolCall("call_1", "search_course_content", {"query": "test"})
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])
        
        second_message = MockMessage(content="Final answer based on search")
        second_completion = MockCompletion([MockChoice(second_message)])
        
        self.mock_client.chat.completions.create.side_effect = [first_completion, second_completion]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result: found relevant content"
        
        # Act
        result = self.generator.generate_response(
            "Test query",
            tools=[],
            tool_manager=mock_tool_manager
        )
        
        # Assert - Check that the second API call includes tool result
        second_call_args = self.mock_client.chat.completions.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        
        # Should have system, user, assistant (with tool call), and tool messages
        self.assertTrue(any(m["role"] == "tool" for m in messages))
        tool_message = [m for m in messages if m["role"] == "tool"][0]
        self.assertEqual(tool_message["content"], "Search result: found relevant content")


if __name__ == "__main__":
    unittest.main()
