import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from models import Source
from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI tool definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the course content",
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """
        try:
            # Use the vector store's unified search interface
            results = self.store.search(
                query=query, course_name=course_name, lesson_number=lesson_number
            )

            # Handle errors
            if results.error:
                return f"Search error: {results.error}"

            # Handle empty results
            if results.is_empty():
                filter_info = ""
                if course_name:
                    filter_info += f" in course '{course_name}'"
                if lesson_number:
                    filter_info += f" in lesson {lesson_number}"
                return f"No relevant content found{filter_info}."

            # Format and return results
            return self._format_results(results)
        except Exception as e:
            return f"Search failed: {str(e)}"

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources: List[Source] = []  # Track sources for the UI with links

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Build source text
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"

            # Fetch lesson link from vector store
            lesson_link = None
            if lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)

            # Track source for the UI
            sources.append(Source(text=source_text, link=lesson_link))

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outline information including title, link, and lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources for the UI

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI tool definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "get_course_outline",
                "description": "Retrieve the complete outline of a course including title, course link, and all lessons with their numbers and titles. Use this when the user asks about course structure, what lessons are in a course, or the outline of a specific course.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                        }
                    },
                    "required": ["course_name"],
                },
            },
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the course outline retrieval tool.

        Args:
            course_name: Course name/title to retrieve outline for

        Returns:
            Formatted course outline information
        """
        try:
            # Resolve course name to actual title
            course_title = self.store._resolve_course_name(course_name)

            if not course_title:
                return f"No course found matching '{course_name}'."

            # Get course metadata
            course_link = self.store.get_course_link(course_title)

            # Get all courses metadata to find lessons
            all_courses = self.store.get_all_courses_metadata()

            course_data = None
            for course in all_courses:
                if course.get("title") == course_title:
                    course_data = course
                    break

            if not course_data:
                return f"Could not retrieve outline for course '{course_title}'."

            # Format and return results
            return self._format_outline(course_title, course_link, course_data)
        except Exception as e:
            return f"Failed to retrieve course outline: {str(e)}"

    def _format_outline(
        self, course_title: str, course_link: Optional[str], course_data: Dict[str, Any]
    ) -> str:
        """Format course outline information"""
        # Reset sources
        self.last_sources = []

        # Add course link as source
        if course_link:
            self.last_sources.append(Source(text=course_title, link=course_link))

        # Build outline
        lines = []
        lines.append(f"Course Title: {course_title}")

        if course_link:
            lines.append(f"Course Link: {course_link}")

        # Add lesson list
        lessons = course_data.get("lessons", [])
        if lessons:
            lines.append(f"\nTotal Lessons: {len(lessons)}")
            lines.append("\nLesson List:")

            for lesson in lessons:
                lesson_num = lesson.get("lesson_number", "N/A")
                lesson_title = lesson.get("lesson_title", "Untitled")
                lesson_link = lesson.get("lesson_link")

                lines.append(f"  Lesson {lesson_num}: {lesson_title}")

                # Track lesson links as sources
                if lesson_link:
                    source_text = f"{course_title} - Lesson {lesson_num}"
                    self.last_sources.append(Source(text=source_text, link=lesson_link))
        else:
            lines.append("\nNo lessons available for this course.")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("function", {}).get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its function definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for OpenAI tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> List[Source]:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
