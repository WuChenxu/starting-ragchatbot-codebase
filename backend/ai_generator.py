from openai import OpenAI, AuthenticationError
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Moonshot AI (Kimi) API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within course materials
2. **get_course_outline**: Retrieve complete course outline including title, course link, and lesson list

Tool Usage Guidelines:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
- **get_course_outline**: Use when users ask about course structure, what lessons are in a course, or the outline of a specific course. When using this tool, return the course title, course link, the number and title of each lesson in your response. Format each lesson on a single line as: "Lesson N: Title".
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific content questions**: Use search_course_content first, then answer
- **Course outline/structure questions**: Use get_course_outline first, then answer with the course title, course link, lesson numbers and titles
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
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
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if available (OpenAI format)
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        try:
            # Get response from Kimi
            response = self.client.chat.completions.create(**api_params)
            
            # Handle tool execution if needed
            message = response.choices[0].message
            if message.tool_calls and tool_manager:
                return self._handle_tool_execution(
                    messages=messages,
                    assistant_message=message,
                    tool_manager=tool_manager
                )
            
            # Return direct response
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
    
    def _handle_tool_execution(self, messages: List[Dict], assistant_message, tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            messages: Current conversation messages
            assistant_message: The assistant message containing tool calls
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
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
        import json
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            try:
                tool_result = tool_manager.execute_tool(function_name, **function_args)
            except Exception as e:
                tool_result = f"Error executing tool '{function_name}': {str(e)}"
            
            # Add tool result as a tool message (OpenAI format)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })
        
        # Get final response
        final_response = self.client.chat.completions.create(
            **self.base_params,
            messages=messages
        )
        
        return final_response.choices[0].message.content or ""
