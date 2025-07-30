"""
LangGraph workflow definition for AI Assistant
"""

from typing import TypedDict, List, Tuple, Optional
from langgraph.graph import StateGraph, END
# from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import processing modules
from speech_to_text import record_audio, transcribe_audio
from ai_agents import ask_agent
from text_to_speech import text_to_speech_with_eleven_lab


class AgentState(TypedDict):
    """State definition for the AI Assistant workflow"""
    messages: List[BaseMessage]
    chat_history: List[Tuple[str, str]]
    audio_file: str
    current_user_input: str
    current_response: str
    error_message: Optional[str]
    session_active: bool
    processing_audio: bool


class AIAssistantWorkflow:
    """LangGraph workflow for AI Assistant"""
    
    def __init__(self, audio_file: str = "audio_question.mp3"):
        self.audio_file = audio_file
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create and configure the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("record_audio", self._audio_recording_node)
        workflow.add_node("transcribe", self._transcription_node)
        workflow.add_node("ai_response", self._ai_response_node)
        workflow.add_node("text_to_speech", self._text_to_speech_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("record_audio")
        
        # Add sequential edges
        workflow.add_edge("record_audio", "transcribe")
        workflow.add_edge("transcribe", "ai_response")
        workflow.add_edge("ai_response", "text_to_speech")
        
        # Add conditional edges
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
        
        return workflow.compile()
    
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
    
    def _ai_response_node(self, state: AgentState) -> AgentState:
        """Node for generating AI response"""
        try:
            if state["current_user_input"] and state["session_active"]:
                response = ask_agent(user_query=state["current_user_input"])
                state["current_response"] = response
                state["messages"].append(AIMessage(content=response))
                
                # Update chat history
                state["chat_history"].append([state["current_user_input"], response])
            
            return state
        except Exception as e:
            state["error_message"] = f"AI response error: {str(e)}"
            return state
    
    def _text_to_speech_node(self, state: AgentState) -> AgentState:
        """Node for converting response to speech"""
        try:
            if state["current_response"]:
                text_to_speech_with_eleven_lab(
                    input_text=state["current_response"],
                    output_file="final.mp3"
                )
            return state
        except Exception as e:
            state["error_message"] = f"TTS error: {str(e)}"
            return state
    
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
            processing_audio=False
        )
    
    def invoke(self, state: AgentState) -> AgentState:
        """Execute the workflow with given state"""
        return self.workflow.invoke(state)