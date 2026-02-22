import json
from openai import OpenAI, AuthenticationError
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Moonshot AI (Kimi) API for generating responses"""
    
    # Maximum number of sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within course materials
2. **get_course_outline**: Retrieve complete course outline including title, course link, and lesson list

Tool Usage Guidelines:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
  - When users ask about a specific lesson (e.g., "lesson 5", "lesson 3"), ALWAYS use the `lesson_number` parameter to filter by that lesson
  - Example: Query "What was covered in lesson 5 of MCP?" → Use `course_name="MCP"`, `lesson_number=5`, and `query` describing what to find
- **get_course_outline**: Use when users ask about course structure, what lessons are in a course, or the outline of a specific course. When using this tool, return the course title, course link, the number and title of each lesson in your response. Format each lesson on a single line as: "Lesson N: Title".
- **Sequential Search Support**: You may make up to 2 tool calls in sequence when:
  - Comparing content across multiple courses (e.g., "How do the RAG implementations differ between courses A and B?")
  - Answering multi-part questions requiring different types of information
  - The first search yields insufficient results for a complete answer
  - Cross-referencing course outlines with specific content
  - Finding other courses covering the same topic: First get outline of source course to identify the topic, then search other courses for that topic
- **Cross-Course Comparison Rules**:
  - "Other courses" refers to DIFFERENT course titles (e.g., "MCP" vs "Building Towards Computer Use")
  - When asked "are there other courses covering X": First identify X from the source course, then search across other course names
  - Do NOT reference different lessons of the same course as "other courses"
- **When to stop**: After receiving tool results, analyze whether you have sufficient information to answer directly. If yes, provide the final answer immediately without additional searches.
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific content questions**: Use search_course_content first, then answer
- **Course outline/structure questions**: Use get_course_outline first, then answer with the course title, course link, lesson numbers and titles
- **Multi-course or complex questions**: Make sequential tool calls as needed (max 2 rounds), then synthesize into a unified answer
- **No meta-commentary in final output**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results" or "I searched for..."

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
5. **Well-formatted** - Use numbered lists (1. 2. 3.) for multi-point content, with each specific content on its own line

Formatting Guidelines for Lesson Content:
- When describing what was covered in a lesson, use numbered lists for clarity
- Each specific topic or concept should be on its own line
- Example format:
  1. **Topic Name**: Description of the specific content covered
  2. **Topic Name**: Description of the specific content covered

Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.moonshot.cn/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential rounds of tool calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare messages for OpenAI format
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        try:
            # Sequential tool calling loop - up to MAX_TOOL_ROUNDS rounds
            for round_num in range(self.MAX_TOOL_ROUNDS + 1):
                # Prepare API call parameters
                api_params = {
                    **self.base_params,
                    "messages": messages
                }
                
                # Add tools if available and we haven't reached max rounds yet
                # Tools are available in rounds 0 and 1, not in the final round
                if tools and round_num < self.MAX_TOOL_ROUNDS:
                    api_params["tools"] = tools
                    api_params["tool_choice"] = "auto"
                
                # Get response from Kimi
                response = self.client.chat.completions.create(**api_params)
                message = response.choices[0].message
                
                # Check if AI wants to make tool calls
                if message.tool_calls and tool_manager and round_num < self.MAX_TOOL_ROUNDS:
                    # Execute tools and prepare for next round
                    should_continue = self._execute_tool_round(
                        messages=messages,
                        assistant_message=message,
                        tool_manager=tool_manager
                    )
                    
                    # Termination condition (c): Tool execution failed
                    if not should_continue:
                        # Make one final API call without tools to synthesize error response
                        final_response = self.client.chat.completions.create(
                            **self.base_params,
                            messages=messages
                        )
                        return final_response.choices[0].message.content or ""
                    
                    # Continue to next round
                    continue
                else:
                    # Termination condition (b): No tool calls requested
                    # Return the response directly
                    return message.content or ""
            
            # Termination condition (a): Max rounds completed
            # Return the last response
            return message.content or ""
            
        except AuthenticationError as e:
            error_msg = """
[API Authentication Error] 

Please check your MOONSHOT_API_KEY in the .env file.

To get a valid API key:
1. Visit https://platform.moonshot.cn/
2. Sign up or log in to your account
3. Go to API Keys section and create a new key
4. Update your .env file with the new key:
   MOONSHOT_API_KEY=your-actual-api-key-here

Then restart the server.
"""
            print(error_msg)
            raise Exception("Invalid Moonshot API Key. Please check your .env file configuration.") from e
        except Exception as e:
            print(f"Error calling Kimi API: {e}")
            raise
    
    def _execute_tool_round(self, messages: List[Dict], assistant_message, tool_manager) -> bool:
        """
        Execute a single round of tool calls and prepare messages for next round.
        
        Args:
            messages: Current conversation messages (modified in place)
            assistant_message: The assistant message containing tool calls
            tool_manager: Manager to execute tools
            
        Returns:
            True if all tools executed successfully, False if any tool failed
        """
        # Add assistant's message with tool calls
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]
        })
        
        # Execute all tool calls and collect results
        has_error = False
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            try:
                tool_result = tool_manager.execute_tool(function_name, **function_args)
            except Exception as e:
                tool_result = f"Error executing tool '{function_name}': {str(e)}"
                has_error = True
            
            # Add tool result as a tool message (OpenAI format)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })
        
        return not has_error
