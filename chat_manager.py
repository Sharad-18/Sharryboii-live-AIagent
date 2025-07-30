"""
Chat state management module for AI Assistant
"""

import time
from typing import List, Tuple, Generator
from workflow import AIAssistantWorkflow


class ChatManager:
    """Manages chat state and continuous audio processing"""
    
    def __init__(self, audio_file: str = "audio_question.mp3"):
        self.workflow = AIAssistantWorkflow(audio_file)
        self.chat_history: List[Tuple[str, str]] = []
        self.is_processing = False
        self.session_active = True
    
    def process_audio_cycle(self) -> List[Tuple[str, str]]:
        """Single cycle of audio processing using LangGraph"""
        if self.is_processing:
            return self.chat_history
        
        self.is_processing = True
        
        try:
            # Create initial state for this cycle
            initial_state = self.workflow.create_initial_state(self.chat_history.copy())
            
            # Run the workflow synchronously for one complete cycle
            final_state = self.workflow.invoke(initial_state)
            
            # Update chat history
            if final_state.get("chat_history"):
                self.chat_history = final_state["chat_history"]
            
            # Check if session should end
            if not final_state.get("session_active", True):
                self.session_active = False
                self.chat_history.append(["System", "Session ended. Say hello to start again!"])
            
        except Exception as e:
            print(f"Workflow error: {e}")
            self.chat_history.append(["System", f"Error: {str(e)}"])
        
        finally:
            self.is_processing = False
        
        return self.chat_history
    
    def continuous_audio_processing(self) -> Generator[List[Tuple[str, str]], None, None]:
        """Continuous audio processing that yields chat updates"""
        while self.session_active:
            try:
                # Process one audio cycle
                updated_history = self.process_audio_cycle()
                
                # Yield the updated chat history
                yield updated_history
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
                # Check if session is still active
                if not self.session_active:
                    break
                    
            except Exception as e:
                print(f"Continuous processing error: {e}")
                self.chat_history.append(["System", f"Processing error: {str(e)}"])
                yield self.chat_history
                break
    
    def clear_chat_history(self) -> List[Tuple[str, str]]:
        """Clear the chat history"""
        self.chat_history = []
        self.session_active = True
        return []
    
    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Get current chat history"""
        return self.chat_history
    
    def add_system_message(self, message: str) -> None:
        """Add a system message to chat history"""
        self.chat_history.append(["System", message])
    
    def is_session_active(self) -> bool:
        """Check if session is active"""
        return self.session_active
    
    def restart_session(self) -> None:
        """Restart the session"""
        self.session_active = True
        self.is_processing = False
        self.add_system_message("Session restarted. Ready for conversation!")
    
    def get_status(self) -> dict:
        """Get detailed status information"""
        return {
            "session_active": self.session_active,
            "is_processing": self.is_processing,
            "chat_history_length": len(self.chat_history),
            "last_message": self.chat_history[-1] if self.chat_history else None
        }