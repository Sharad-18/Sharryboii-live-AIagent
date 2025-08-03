"""
Enhanced LangGraph workflow with tool integration - FIXED
"""

from typing import TypedDict, List, Tuple, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import re
import uuid
from langgraph.checkpoint.memory import InMemorySaver
# Import processing modules
from speech_to_text import record_audio, transcribe_audio
from ai_agents import ask_agent
from text_to_speech import text_to_speech_with_eleven_lab,text_to_speech_with_gtts
from tools import tools  # Import our enhanced tools

memory = InMemorySaver()

class AgentState(TypedDict):
    """Enhanced state definition for the AI Assistant workflow"""
    messages: List[BaseMessage]
    chat_history: List[Tuple[str, str]]
    audio_file: str
    current_user_input: str
    current_response: str
    error_message: Optional[str]
    session_active: bool
    processing_audio: bool
    detected_intent: Optional[str]
    tool_results: Optional[str]
    needs_tool_execution: bool


class EnhancedAIAssistantWorkflow:
    """Enhanced LangGraph workflow for AI Assistant with tool integration"""
    
    def __init__(self, audio_file: str = "audio_question.mp3"):
        self.audio_file = audio_file
        self.workflow = self._create_workflow()
        self.intent_patterns = self._setup_intent_patterns()
        self.thread_id = str(uuid.uuid4())  # Generate unique thread ID
    
    def _setup_intent_patterns(self) -> dict:
        """Setup patterns for intent detection"""
        return {
            "vision": [
                r"what do you see", r"analyze.*image", r"look at", r"describe.*image",
                r"what.*in.*image", r"camera", r"webcam", r"picture", r"photo"
            ],
            "weather": [
                r"weather", r"temperature", r"forecast", r"rain", r"sunny", r"cloudy",
                r"hot", r"cold", r"climate"
            ],
            "search": [
                r"search", r"look up", r"find.*information", r"google", r"what is",
                r"tell me about", r"research"
            ],
            "news": [
                r"news", r"headlines", r"current events", r"what.*happening",
                r"latest news"
            ],
            "time": [
                r"time", r"what time", r"current time", r"clock", r"date"
            ],
            "system": [
                r"system info", r"computer", r"cpu", r"memory", r"disk space",
                r"performance"
            ],
            "calculator": [
                r"calculate", r"math", r"\d+.*[\+\-\*\/].*\d+", r"compute",
                r"addition", r"subtract", r"multiply", r"divide"
            ],
            "files": [
                r"list files", r"show files", r"directory", r"folder contents",
                r"what files"
            ]
        }
    
    def _detect_intent(self, user_input: str) -> Optional[str]:
        """Detect user intent from input text"""
        user_input_lower = user_input.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return intent
        
        return None
    
    def _extract_parameters(self, user_input: str, intent: str) -> dict:
        """Extract parameters for tool execution based on intent"""
        params = {}
        user_input_lower = user_input.lower()
        
        if intent == "weather":
            # Extract city name
            city_match = re.search(r"weather.*in\s+([a-zA-Z\s]+)", user_input_lower)
            if city_match:
                params["city"] = city_match.group(1).strip()
            else:
                params["city"] = "Jaipur"  # Default
            params["original_query"] = user_input
        
        elif intent == "search":
            # Extract search query - be more flexible
            search_patterns = [
                r"search\s+(?:for\s+)?(.+)",
                r"look\s+up\s+(.+)",
                r"find.*about\s+(.+)",
                r"what\s+is\s+(.+)",
                r"tell\s+me\s+about\s+(.+)"
            ]
            query_found = False
            for pattern in search_patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    params["query"] = match.group(1).strip()
                    query_found = True
                    break
            
            # If no specific pattern matched, use the entire input as search query
            if not query_found:
                params["query"] = user_input.strip()
            params["original_query"] = user_input
        
        elif intent == "calculator":
            # Extract mathematical expression
            math_match = re.search(r"([\d\+\-\*\/\.\(\)\s]+)", user_input)
            if math_match:
                params["expression"] = math_match.group(1).strip()
            else:
                # If no clear math expression, pass the whole query
                params["expression"] = user_input.strip()
            params["original_query"] = user_input
        
        elif intent == "vision":
            # IMPROVED: Always use the user's actual query
            # Remove common prefixes to clean up the query
            vision_prefixes = [
                r"^(can you |please |could you )?",
                r"^(what do you see|analyze.*image|describe.*image|look at)",
                r"^(in this image|in the image)",
            ]
            
            cleaned_query = user_input
            for prefix_pattern in vision_prefixes:
                cleaned_query = re.sub(prefix_pattern, "", cleaned_query, flags=re.IGNORECASE).strip()
            
            # If after cleaning we have a meaningful query, use it
            if len(cleaned_query) > 3:  # Avoid very short queries
                params["query"] = cleaned_query
            else:
                # Only fallback to default if the query is too short or empty
                params["query"] = "What do you see in this image?"
            params["original_query"] = user_input
        
        elif intent == "news":
            # Extract specific news topic if mentioned
            news_patterns = [
                r"news about\s+(.+)",
                r"latest.*news.*on\s+(.+)",
                r"(.+)\s+news"
            ]
            for pattern in news_patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    params["topic"] = match.group(1).strip()
                    break
            params["original_query"] = user_input
        
        elif intent == "files":
            # Extract directory path if specified
            path_patterns = [
                r"files in\s+(.+)",
                r"list files.*in\s+(.+)",
                r"show.*files.*from\s+(.+)"
            ]
            for pattern in path_patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    params["path"] = match.group(1).strip()
                    break
            params["original_query"] = user_input
        
        else:
            # For any other intent, preserve the original query
            params["original_query"] = user_input
        
        return params
    
    def _create_workflow(self):
        """Create and configure the enhanced LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("record_audio", self._audio_recording_node)
        workflow.add_node("transcribe", self._transcription_node)
        workflow.add_node("intent_detection", self._intent_detection_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("ai_response", self._ai_response_node)
        workflow.add_node("text_to_speech", self._text_to_speech_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("record_audio")
        
        # Add sequential edges
        workflow.add_edge("record_audio", "transcribe")
        workflow.add_edge("transcribe", "intent_detection")
        
        # Add conditional edges from intent detection
        workflow.add_conditional_edges(
            "intent_detection",
            self._should_use_tools,
            {
                "use_tools": "tool_execution",
                "direct_response": "ai_response",
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("tool_execution", "ai_response")
        workflow.add_edge("ai_response", "text_to_speech")
        
        # Add conditional edges from text_to_speech
        workflow.add_conditional_edges(
            "text_to_speech",
            self._should_continue,
            {
                "end": END,
                "error_handler": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "error_handler",
            lambda state: "end",
            {"end": END}
        )
        
        return workflow.compile(checkpointer=memory)
    
    def _audio_recording_node(self, state: AgentState) -> AgentState:
        """Node for recording audio input"""
        try:
            record_audio(file_path=state.get("audio_file", self.audio_file))
            state["processing_audio"] = True
            state["error_message"] = None
            return state
        except Exception as e:
            state["error_message"] = f"Audio recording error: {str(e)}"
            state["processing_audio"] = False
            return state
    
    def _transcription_node(self, state: AgentState) -> AgentState:
        """Node for transcribing audio to text"""
        try:
            if state["processing_audio"]:
                audio_filepath = state.get("audio_file", self.audio_file)
                user_input = transcribe_audio(audio_filepath)
                state["current_user_input"] = user_input
                state["messages"].append(HumanMessage(content=user_input))
                
                # Check for exit condition
                if "goodbye" in user_input.lower():
                    state["session_active"] = False
            
            return state
        except Exception as e:
            state["error_message"] = f"Transcription error: {str(e)}"
            return state
    
    def _intent_detection_node(self, state: AgentState) -> AgentState:
        """Node for detecting user intent and determining if tools are needed"""
        try:
            user_input = state["current_user_input"]
            if user_input:
                detected_intent = self._detect_intent(user_input)
                state["detected_intent"] = detected_intent
                state["needs_tool_execution"] = detected_intent is not None
            
            return state
        except Exception as e:
            state["error_message"] = f"Intent detection error: {str(e)}"
            return state
    
    def _tool_execution_node(self, state: AgentState) -> AgentState:
        """Node for executing appropriate tools based on detected intent"""
        try:
            intent = state["detected_intent"]
            user_input = state["current_user_input"]
            
            if intent and user_input:
                # Extract parameters for the tool
                params = self._extract_parameters(user_input, intent)
                
                # Execute the appropriate tool with better error handling
                result = None
                
                if intent == "weather":
                    result = tools.execute_tool("weather", **{k: v for k, v in params.items() if k != "original_query"})
                elif intent == "search":
                    result = tools.execute_tool("search", **{k: v for k, v in params.items() if k != "original_query"})
                elif intent == "vision":
                    result = tools.execute_tool("vision", **{k: v for k, v in params.items() if k != "original_query"})
                elif intent == "news":
                    # Handle both general and topic-specific news
                    tool_params = {k: v for k, v in params.items() if k != "original_query"}
                    if "topic" in tool_params:
                        result = tools.execute_tool("news", topic=tool_params["topic"])
                    else:
                        result = tools.execute_tool("news")
                elif intent == "time":
                    result = tools.execute_tool("time")
                elif intent == "system":
                    result = tools.execute_tool("system")
                elif intent == "calculator":
                    result = tools.execute_tool("calculator", **{k: v for k, v in params.items() if k != "original_query"})
                elif intent == "files":
                    # Handle directory-specific file listing
                    tool_params = {k: v for k, v in params.items() if k != "original_query"}
                    if "path" in tool_params:
                        result = tools.execute_tool("files", path=tool_params["path"])
                    else:
                        result = tools.execute_tool("files")
                else:
                    result = f"Intent '{intent}' detected but tool execution not implemented"
                
                # Add context about the original query for better AI response
                if result and params.get("original_query"):
                    result = f"User asked: '{params['original_query']}'\nTool result: {result}"
                
                state["tool_results"] = result
            
            return state
        except Exception as e:
            state["error_message"] = f"Tool execution error: {str(e)}"
            return state
    
    def _ai_response_node(self, state: AgentState) -> AgentState:
        """Node for generating AI response, incorporating tool results if available"""
        try:
            if state["current_user_input"] and state["session_active"]:
                # Prepare context for AI agent
                context = state["current_user_input"]
                
                # Add tool results to context if available
                if state.get("tool_results"):
                    context = f"User query: {context}\n\nTool results: {state['tool_results']}\n\nPlease provide a natural response incorporating this information."
                
                # Generate AI response
                response = ask_agent(user_query=context)
                state["current_response"] = response
                state["messages"].append(AIMessage(content=response))
                
                # Update chat history
                state["chat_history"].append([state["current_user_input"], response])
            
            return state
        except Exception as e:
            state["error_message"] = f"AI response error: {str(e)}"
            return state
    
    # def _text_to_speech_node(self, state: AgentState) -> AgentState:
    #     """Node for converting response to speech"""
    #     try:
    #         if state["current_response"]:
    #             text_to_speech_with_eleven_lab(
    #                 input_text=state["current_response"],
    #                 output_file="final.mp3"
    #             )
    #         return state
    #     except Exception as e:
    #         state["error_message"] = f"TTS error: {str(e)}"
    #         return state
    def _text_to_speech_node(self, state: AgentState) -> AgentState:
        """Node for converting response to speech with fallback logic"""
        try:
            if state["current_response"]:
                response_text = state["current_response"]
                
                # First, try ElevenLabs TTS
                try:
                    text_to_speech_with_eleven_lab(
                        input_text=response_text,
                        output_file="final.mp3"
                    )
                    state["tts_service_used"] = "elevenlabs"
                    
                except Exception as elevenlabs_error:
                    elevenlabs_error_str = str(elevenlabs_error)
                    
                    # Check if it's a quota/limit error
                    if any(keyword in elevenlabs_error_str.lower() for keyword in 
                        ['quota', 'limit', 'credits', 'exceeded', '401', 'unauthorized']):
                        
                        print(f"ElevenLabs quota exceeded, falling back to gTTS: {elevenlabs_error_str}")
                        
                        # Fallback to gTTS
                        try:
                            text_to_speech_with_gtts(
                                input_text=response_text,
                                output_file="final.mp3"
                            )
                            state["tts_service_used"] = "gtts"
                            
                            # Add informational message to chat history if available
                            if "chat_history" in state:
                                state["chat_history"].append([
                                    "System", 
                                    "🔄 Switched to Google TTS due to ElevenLabs quota limit"
                                ])
                        
                        except Exception as gtts_error:
                            # Both TTS services failed
                            error_msg = f"Both TTS services failed - ElevenLabs: {elevenlabs_error_str}, gTTS: {str(gtts_error)}"
                            state["error_message"] = f"TTS error: {error_msg}"
                            
                            # Add error message to chat history if available
                            if "chat_history" in state:
                                state["chat_history"].append([
                                    "System", 
                                    "🔇 Audio output unavailable. Continuing with text-only responses."
                                ])
                    else:
                        # Non-quota error from ElevenLabs, re-raise the original error
                        state["error_message"] = f"TTS error: {elevenlabs_error_str}"
            
            return state
            
        except Exception as e:
            # Catch-all for any unexpected errors
            state["error_message"] = f"TTS error: {str(e)}"
            return state
    
    def _should_use_tools(self, state: AgentState) -> str:
        """Conditional edge function to determine if tools should be used"""
        if state.get("error_message"):
            return "error"
        if state.get("needs_tool_execution", False):
            return "use_tools"
        return "direct_response"
    
    def _should_continue(self, state: AgentState) -> str:
        """Conditional edge function to determine next step"""
        if not state["session_active"]:
            return "end"
        if state["error_message"]:
            return "error_handler"
        return "end"
    
    def _error_handler_node(self, state: AgentState) -> AgentState:
        """Node for handling errors gracefully"""
        error_msg = state["error_message"]
        print(f"Error handled: {error_msg}")
        
        # Add error message to chat history
        state["chat_history"].append(["System", f"Error: {error_msg}"])
        
        # Reset error state
        state["error_message"] = None
        state["processing_audio"] = False
        state["needs_tool_execution"] = False
        
        return state
    
    def create_initial_state(self, chat_history: List[Tuple[str, str]] = None) -> AgentState:
        """Create initial state for workflow execution"""
        return AgentState(
            messages=[],
            chat_history=chat_history or [],
            audio_file=self.audio_file,
            current_user_input="",
            current_response="",
            error_message=None,
            session_active=True,
            processing_audio=False,
            detected_intent=None,
            tool_results=None,
            needs_tool_execution=False
        )
    
    def invoke(self, state: AgentState) -> AgentState:
        """Execute the workflow with given state - FIXED with proper config"""
        # Create configuration with thread_id for checkpointer
        config = {"configurable": {"thread_id": self.thread_id}}
        return self.workflow.invoke(state, config=config)
    
    def invoke_with_custom_thread(self, state: AgentState, thread_id: str = None) -> AgentState:
        """Execute the workflow with a custom thread ID"""
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}
        return self.workflow.invoke(state, config=config)
    
    def get_workflow_state(self, thread_id: str = None) -> dict:
        """Get the current state of a workflow by thread_id"""
        if thread_id is None:
            thread_id = self.thread_id
        
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return self.workflow.get_state(config)
        except Exception as e:
            print(f"Error getting workflow state: {e}")
            return None
    
    def reset_workflow_thread(self):
        """Reset the workflow by generating a new thread ID"""
        self.thread_id = str(uuid.uuid4())
        print(f"Workflow reset with new thread ID: {self.thread_id}")
    
    def get_available_tools_info(self) -> str:
        """Get information about available tools"""
        tool_info = tools.get_available_tools()
        result = "🛠️ Available AI Assistant Tools:\n\n"
        
        for tool, description in tool_info.items():
            result += f"• **{tool.title()}**: {description}\n"
        
        result += "\n💡 **Usage Examples**:\n"
        result += "• 'What's the weather in New York?'\n"
        result += "• 'Search for Python tutorials'\n"
        result += "• 'What do you see?' (uses camera)\n"
        result += "• 'Calculate 25 * 4 + 10'\n"
        result += "• 'What time is it?'\n"
        result += "• 'Show me system information'\n"
        result += "• 'Get latest technology news'\n"
        
        return result


# # Example usage function to demonstrate proper invocation
# def run_workflow_example():
#     """Example of how to properly run the workflow"""
    
#     # Create workflow instance
#     workflow = EnhancedAIAssistantWorkflow()
    
#     # Create initial state
#     initial_state = workflow.create_initial_state()
    
#     try:
#         # Execute workflow - now with proper configuration
#         result = workflow.invoke(initial_state)
#         print("Workflow completed successfully!")
#         print(f"Final response: {result.get('current_response', 'No response')}")
        
#     except Exception as e:
#         print(f"Workflow execution error: {e}")
    
#     # Example of using custom thread ID
#     try:
#         custom_thread_id = "my-custom-session-123"
#         result2 = workflow.invoke_with_custom_thread(initial_state, custom_thread_id)
#         print("Custom thread workflow completed!")
        
#     except Exception as e:
#         print(f"Custom thread workflow error: {e}")


# if __name__ == "__main__":
#     run_workflow_example()