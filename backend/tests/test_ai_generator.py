"""Tests for ai_generator.py - AIGenerator and tool calling"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, call, patch

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

        with patch("ai_generator.OpenAI") as mock_openai:
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
        self.assertEqual(self.generator.MAX_TOOL_ROUNDS, 2)

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
        result = self.generator.generate_response(
            "Follow up question", conversation_history=history
        )

        # Assert
        call_args = self.mock_client.chat.completions.create.call_args
        self.assertIn("Previous conversation:", call_args[1]["messages"][0]["content"])
        self.assertIn(history, call_args[1]["messages"][0]["content"])

    def test_generate_response_with_single_tool_call(self):
        """Test generate_response when AI makes one tool call then answers"""
        # Arrange
        # First response - AI wants to call tool
        tool_call = MockToolCall(
            "call_1", "search_course_content", {"query": "RAG", "course_name": "MCP"}
        )
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
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_course_content",
                    "description": "Search course content",
                },
            }
        ]

        # Act
        result = self.generator.generate_response(
            "What is RAG?", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "RAG stands for Retrieval-Augmented Generation")
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG", course_name="MCP"
        )

    def test_generate_response_with_sequential_tool_calls(self):
        """Test sequential tool calling - 2 rounds of tool calls"""
        # Arrange
        # Round 1: AI calls get_course_outline
        tool_call_1 = MockToolCall("call_1", "get_course_outline", {"course_name": "MCP"})
        first_message = MockMessage(content=None, tool_calls=[tool_call_1])
        first_completion = MockCompletion([MockChoice(first_message)])

        # Round 2: AI calls search_course_content based on outline results
        tool_call_2 = MockToolCall(
            "call_2",
            "search_course_content",
            {"query": "RAG implementation", "course_name": "MCP", "lesson_number": 3},
        )
        second_message = MockMessage(content=None, tool_calls=[tool_call_2])
        second_completion = MockCompletion([MockChoice(second_message)])

        # Final response
        third_message = MockMessage(
            content="Lesson 3 of MCP covers RAG implementation using vector databases."
        )
        third_completion = MockCompletion([MockChoice(third_message)])

        self.mock_client.chat.completions.create.side_effect = [
            first_completion,
            second_completion,
            third_completion,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: MCP\nLesson 3: RAG Implementation",  # First tool result
            "RAG uses vector databases for retrieval...",  # Second tool result
        ]

        tools = [
            {"type": "function", "function": {"name": "get_course_outline"}},
            {"type": "function", "function": {"name": "search_course_content"}},
        ]

        # Act
        result = self.generator.generate_response(
            "What does lesson 3 of MCP cover?", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(
            result, "Lesson 3 of MCP covers RAG implementation using vector databases."
        )
        # Should make 3 API calls: round 1, round 2, final
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)
        # Should execute 2 tools
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify tools were called in sequence
        calls = mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0][0], "get_course_outline")
        self.assertEqual(calls[1][0][0], "search_course_content")

    def test_sequential_tools_available_in_both_rounds(self):
        """Test that tools are passed in both round 1 and round 2 API calls"""
        # Arrange
        tool_call_1 = MockToolCall("call_1", "search_course_content", {"query": "topic1"})
        first_message = MockMessage(content=None, tool_calls=[tool_call_1])
        first_completion = MockCompletion([MockChoice(first_message)])

        tool_call_2 = MockToolCall("call_2", "search_course_content", {"query": "topic2"})
        second_message = MockMessage(content=None, tool_calls=[tool_call_2])
        second_completion = MockCompletion([MockChoice(second_message)])

        final_message = MockMessage(content="Combined answer")
        final_completion = MockCompletion([MockChoice(final_message)])

        self.mock_client.chat.completions.create.side_effect = [
            first_completion,
            second_completion,
            final_completion,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        # Act
        self.generator.generate_response("Test", tools=tools, tool_manager=mock_tool_manager)

        # Assert
        api_calls = self.mock_client.chat.completions.create.call_args_list

        # Round 1: tools should be present
        self.assertIn("tools", api_calls[0][1])
        self.assertIn("tool_choice", api_calls[0][1])

        # Round 2: tools should still be present
        self.assertIn("tools", api_calls[1][1])
        self.assertIn("tool_choice", api_calls[1][1])

        # Final call: tools should NOT be present
        self.assertNotIn("tools", api_calls[2][1])
        self.assertNotIn("tool_choice", api_calls[2][1])

    def test_max_rounds_termination(self):
        """Test that tool calling stops after MAX_TOOL_ROUNDS even if more tool calls requested"""
        # Arrange - AI keeps requesting tool calls
        tool_call_1 = MockToolCall("call_1", "search_course_content", {"query": "topic1"})
        first_message = MockMessage(content=None, tool_calls=[tool_call_1])
        first_completion = MockCompletion([MockChoice(first_message)])

        tool_call_2 = MockToolCall("call_2", "search_course_content", {"query": "topic2"})
        second_message = MockMessage(content=None, tool_calls=[tool_call_2])
        second_completion = MockCompletion([MockChoice(second_message)])

        # Third response - AI would want another tool but max rounds reached
        third_message = MockMessage(content="Final answer after max rounds")
        third_completion = MockCompletion([MockChoice(third_message)])

        self.mock_client.chat.completions.create.side_effect = [
            first_completion,
            second_completion,
            third_completion,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        # Act
        result = self.generator.generate_response(
            "Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "Final answer after max rounds")
        # Should make exactly 3 API calls (2 tool rounds + final response)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)
        # Should execute exactly 2 tools
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

    def test_no_tool_calls_termination(self):
        """Test termination when AI response has no tool_use blocks"""
        # Arrange
        first_message = MockMessage(content="Direct answer without tools")
        first_completion = MockCompletion([MockChoice(first_message)])
        self.mock_client.chat.completions.create.return_value = first_completion

        mock_tool_manager = Mock()
        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        # Act
        result = self.generator.generate_response(
            "Simple question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "Direct answer without tools")
        # Should only make 1 API call
        self.mock_client.chat.completions.create.assert_called_once()
        # Should not execute any tools
        mock_tool_manager.execute_tool.assert_not_called()

    def test_tool_execution_error_termination(self):
        """Test termination when tool execution fails"""
        # Arrange
        tool_call = MockToolCall("call_1", "search_course_content", {"query": "test"})
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])

        final_message = MockMessage(content="Error occurred while searching")
        final_completion = MockCompletion([MockChoice(final_message)])

        self.mock_client.chat.completions.create.side_effect = [first_completion, final_completion]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        # Act
        result = self.generator.generate_response(
            "Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "Error occurred while searching")
        # Should make 2 API calls (tool round + final synthesis)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)

    def test_conversation_context_preserved_across_rounds(self):
        """Test that conversation context is preserved between tool calling rounds"""
        # Arrange
        captured_calls = []

        def capture_and_return(*args, **kwargs):
            # Capture a copy of the messages at call time
            messages = kwargs.get("messages", [])
            captured_calls.append(
                {
                    "messages_count": len(messages),
                    "message_roles": [m.get("role") for m in messages],
                    "has_tools": "tools" in kwargs,
                }
            )
            return next(responses_iter)

        tool_call_1 = MockToolCall("call_1", "get_course_outline", {"course_name": "MCP"})
        first_message = MockMessage(content=None, tool_calls=[tool_call_1])
        first_completion = MockCompletion([MockChoice(first_message)])

        tool_call_2 = MockToolCall("call_2", "search_course_content", {"query": "Lesson 4 content"})
        second_message = MockMessage(content=None, tool_calls=[tool_call_2])
        second_completion = MockCompletion([MockChoice(second_message)])

        final_message = MockMessage(content="Answer based on both searches")
        final_completion = MockCompletion([MockChoice(final_message)])

        responses = [first_completion, second_completion, final_completion]
        responses_iter = iter(responses)

        self.mock_client.chat.completions.create.side_effect = capture_and_return

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Outline result", "Search result"]

        tools = [
            {"type": "function", "function": {"name": "get_course_outline"}},
            {"type": "function", "function": {"name": "search_course_content"}},
        ]

        # Act
        self.generator.generate_response(
            "Complex query", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(len(captured_calls), 3, "Should make 3 API calls")

        # Verify message counts grow across rounds
        self.assertEqual(captured_calls[0]["messages_count"], 2, "Round 1: system + user")
        self.assertEqual(captured_calls[1]["messages_count"], 4, "Round 2: + assistant + tool")
        self.assertEqual(captured_calls[2]["messages_count"], 6, "Final: + assistant + tool")

        # Verify tools are available in rounds 1 and 2, not in final
        self.assertTrue(captured_calls[0]["has_tools"], "Round 1 should have tools")
        self.assertTrue(captured_calls[1]["has_tools"], "Round 2 should have tools")
        self.assertFalse(captured_calls[2]["has_tools"], "Final should not have tools")

        # Verify message roles in final call
        self.assertIn("system", captured_calls[2]["message_roles"])
        self.assertIn("user", captured_calls[2]["message_roles"])
        self.assertIn("assistant", captured_calls[2]["message_roles"])
        self.assertIn("tool", captured_calls[2]["message_roles"])

    def test_generate_response_api_error(self):
        """Test generate_response handles API errors"""
        # Arrange
        from openai import AuthenticationError

        self.mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key", response=Mock(), body=None
        )

        # Act & Assert
        with self.assertRaises(Exception) as context:
            self.generator.generate_response("Test query")

        self.assertIn("Invalid Moonshot API Key", str(context.exception))

    def test_system_prompt_contains_sequential_guidelines(self):
        """Test that system prompt includes sequential tool usage guidelines"""
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
        self.assertIn("Sequential Search Support", system_content)
        self.assertIn("up to 2 tool calls", system_content)
        self.assertNotIn("One search per query maximum", system_content)


class TestAIGeneratorComparisonQueries(unittest.TestCase):
    """Test cases for comparison queries requiring multiple tool calls"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.model = "moonshot-v1-8k"

        with patch("ai_generator.OpenAI") as mock_openai:
            self.mock_client = Mock()
            mock_openai.return_value = self.mock_client
            self.generator = AIGenerator(self.api_key, self.model)

    def test_comparison_query_across_two_courses(self):
        """Test comparing content across two courses triggers two searches"""
        # Arrange
        # Round 1: Search course A
        tool_call_1 = MockToolCall(
            "call_1", "search_course_content", {"query": "RAG implementation", "course_name": "MCP"}
        )
        first_message = MockMessage(content=None, tool_calls=[tool_call_1])
        first_completion = MockCompletion([MockChoice(first_message)])

        # Round 2: Search course B
        tool_call_2 = MockToolCall(
            "call_2",
            "search_course_content",
            {"query": "RAG implementation", "course_name": "Chroma"},
        )
        second_message = MockMessage(content=None, tool_calls=[tool_call_2])
        second_completion = MockCompletion([MockChoice(second_message)])

        # Final comparison answer
        final_message = MockMessage(
            content="MCP focuses on agent-based RAG while Chroma emphasizes vector storage."
        )
        final_completion = MockCompletion([MockChoice(final_message)])

        self.mock_client.chat.completions.create.side_effect = [
            first_completion,
            second_completion,
            final_completion,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "MCP uses agents for RAG orchestration",
            "Chroma provides vector database storage",
        ]

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        # Act
        result = self.generator.generate_response(
            "Compare RAG implementation between MCP and Chroma courses",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        calls = mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][1]["course_name"], "MCP")
        self.assertEqual(calls[1][1]["course_name"], "Chroma")

    def test_multipart_question_outline_then_content(self):
        """Test multi-part question: get outline then search specific lesson"""
        # Arrange
        # Round 1: Get outline
        tool_call_1 = MockToolCall("call_1", "get_course_outline", {"course_name": "Computer Use"})
        first_message = MockMessage(content=None, tool_calls=[tool_call_1])
        first_completion = MockCompletion([MockChoice(first_message)])

        # Round 2: Search specific lesson content
        tool_call_2 = MockToolCall(
            "call_2",
            "search_course_content",
            {"query": "computer use", "course_name": "Computer Use", "lesson_number": 2},
        )
        second_message = MockMessage(content=None, tool_calls=[tool_call_2])
        second_completion = MockCompletion([MockChoice(second_message)])

        final_message = MockMessage(
            content="Computer Use has 5 lessons. Lesson 2 covers browser automation."
        )
        final_completion = MockCompletion([MockChoice(final_message)])

        self.mock_client.chat.completions.create.side_effect = [
            first_completion,
            second_completion,
            final_completion,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: Computer Use\nLesson 1: Introduction\nLesson 2: Browser Automation",
            "Lesson 2 covers playwright and browser automation techniques",
        ]

        tools = [
            {"type": "function", "function": {"name": "get_course_outline"}},
            {"type": "function", "function": {"name": "search_course_content"}},
        ]

        # Act
        result = self.generator.generate_response(
            "What lessons are in Computer Use and what does Lesson 2 cover?",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        calls = mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0][0], "get_course_outline")
        self.assertEqual(calls[1][0][0], "search_course_content")


class TestAIGeneratorLessonNumberExtraction(unittest.TestCase):
    """Test cases for lesson number parameter extraction and usage"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.model = "moonshot-v1-8k"

        with patch("ai_generator.OpenAI") as mock_openai:
            self.mock_client = Mock()
            mock_openai.return_value = self.mock_client
            self.generator = AIGenerator(self.api_key, self.model)

    def test_lesson_number_passed_for_specific_lesson_query(self):
        """Test that lesson_number parameter is passed when querying specific lesson"""
        # Arrange
        tool_call = MockToolCall(
            "call_1",
            "search_course_content",
            {"query": "lesson 5 content", "course_name": "MCP", "lesson_number": 5},
        )
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])

        final_message = MockMessage(content="Lesson 5 covers creating MCP client")
        final_completion = MockCompletion([MockChoice(final_message)])

        self.mock_client.chat.completions.create.side_effect = [first_completion, final_completion]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Lesson 5 content about MCP client"

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        # Act
        self.generator.generate_response(
            "What was covered in lesson 5 of the MCP course?",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assert - Verify tool was called with lesson_number parameter
        mock_tool_manager.execute_tool.assert_called_once()
        call_args = mock_tool_manager.execute_tool.call_args
        self.assertEqual(call_args[0][0], "search_course_content")
        self.assertIn("lesson_number", call_args[1])
        self.assertEqual(call_args[1]["lesson_number"], 5)
        self.assertEqual(call_args[1]["course_name"], "MCP")

    def test_lesson_number_extracted_from_various_formats(self):
        """Test lesson number extraction from different query formats"""
        test_cases = [
            ("What is in lesson 3 of MCP?", 3),
            ("Tell me about lesson 10", 10),
            ("Lesson 1 content", 1),
            ("What does lesson 8 cover?", 8),
        ]

        for query, expected_lesson in test_cases:
            with self.subTest(query=query, expected_lesson=expected_lesson):
                # Arrange
                tool_call = MockToolCall(
                    "call_1",
                    "search_course_content",
                    {
                        "query": f"lesson {expected_lesson}",
                        "course_name": "MCP",
                        "lesson_number": expected_lesson,
                    },
                )
                first_message = MockMessage(content=None, tool_calls=[tool_call])
                first_completion = MockCompletion([MockChoice(first_message)])

                final_message = MockMessage(content=f"Lesson {expected_lesson} content")
                final_completion = MockCompletion([MockChoice(final_message)])

                self.mock_client.chat.completions.create.side_effect = [
                    first_completion,
                    final_completion,
                ]

                mock_tool_manager = Mock()
                mock_tool_manager.execute_tool.return_value = f"Lesson {expected_lesson} content"

                tools = [{"type": "function", "function": {"name": "search_course_content"}}]

                # Act
                self.generator.generate_response(query, tools=tools, tool_manager=mock_tool_manager)

                # Assert
                call_args = mock_tool_manager.execute_tool.call_args
                self.assertEqual(
                    call_args[1].get("lesson_number"),
                    expected_lesson,
                    f"Query '{query}' should extract lesson_number={expected_lesson}",
                )

                # Reset for next iteration
                self.mock_client.reset_mock()

    def test_system_prompt_contains_lesson_number_guideline(self):
        """Test that system prompt includes lesson_number usage guideline"""
        # Arrange
        mock_message = MockMessage(content="Test")
        mock_completion = MockCompletion([MockChoice(mock_message)])
        self.mock_client.chat.completions.create.return_value = mock_completion

        # Act
        self.generator.generate_response("Test")

        # Assert
        call_args = self.mock_client.chat.completions.create.call_args
        system_content = call_args[1]["messages"][0]["content"]

        self.assertIn("lesson_number", system_content)
        self.assertIn("lesson 5", system_content.lower())
        self.assertIn("ALWAYS use the `lesson_number` parameter", system_content)

    def test_no_lesson_number_for_general_course_query(self):
        """Test that general course queries don't necessarily include lesson_number"""
        # Arrange
        tool_call = MockToolCall(
            "call_1", "search_course_content", {"query": "MCP architecture", "course_name": "MCP"}
        )
        first_message = MockMessage(content=None, tool_calls=[tool_call])
        first_completion = MockCompletion([MockChoice(first_message)])

        final_message = MockMessage(content="MCP architecture information")
        final_completion = MockCompletion([MockChoice(final_message)])

        self.mock_client.chat.completions.create.side_effect = [first_completion, final_completion]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "MCP architecture content"

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        # Act
        self.generator.generate_response(
            "Tell me about MCP architecture", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - For general queries, lesson_number may not be present
        call_args = mock_tool_manager.execute_tool.call_args
        self.assertEqual(call_args[0][0], "search_course_content")
        # General query about architecture shouldn't require lesson_number

    def test_system_prompt_contains_formatting_guidelines(self):
        """Test that system prompt includes formatting guidelines for lesson content"""
        # Arrange
        mock_message = MockMessage(content="Test")
        mock_completion = MockCompletion([MockChoice(mock_message)])
        self.mock_client.chat.completions.create.return_value = mock_completion

        # Act
        self.generator.generate_response("Test")

        # Assert
        call_args = self.mock_client.chat.completions.create.call_args
        system_content = call_args[1]["messages"][0]["content"]

        self.assertIn("Well-formatted", system_content)
        self.assertIn("numbered lists", system_content.lower())
        self.assertIn("1. 2. 3.", system_content)
        self.assertIn("Formatting Guidelines for Lesson Content", system_content)


if __name__ == "__main__":
    unittest.main()
