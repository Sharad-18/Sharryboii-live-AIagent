"""
Enhanced chat state management module with tool integration - UPDATED
"""

import time
from typing import List, Tuple, Generator, Optional
from workflow import EnhancedAIAssistantWorkflow
from tools import tools


class EnhancedChatManager:
    """Enhanced chat manager with tool integration and better state management"""
    
    def __init__(self, audio_file: str = "audio_question.mp3"):
        self.workflow = EnhancedAIAssistantWorkflow(audio_file)
        self.chat_history: List[Tuple[str, str]] = []
        self.is_processing = False
        self.session_active = True
        self.tools_enabled = True
        self.last_intent = None
        self.conversation_count = 0
        self.current_thread_id = None  # Track current thread ID
        self.tts_enabled = True  # NEW: TTS toggle
        self.max_tts_chars = 500  # NEW: TTS character limit
    
    def process_audio_cycle(self) -> List[Tuple[str, str]]:
        """Single cycle of enhanced audio processing using LangGraph with tools"""
        if self.is_processing:
            return self.chat_history
        
        self.is_processing = True
        
        try:
            # Create initial state for this cycle
            initial_state = self.workflow.create_initial_state(self.chat_history.copy())
            
            # Run the enhanced workflow with proper thread management
            if self.current_thread_id:
                final_state = self.workflow.invoke_with_custom_thread(
                    initial_state, 
                    self.current_thread_id
                )
            else:
                final_state = self.workflow.invoke(initial_state)
                self.current_thread_id = self.workflow.thread_id
            
            # Handle potential TTS errors gracefully
            self._handle_tts_response(final_state)
            
            # Update chat history
            if final_state.get("chat_history"):
                self.chat_history = final_state["chat_history"]
                self.conversation_count = len(self.chat_history)
            
            # Store last detected intent for analytics
            if final_state.get("detected_intent"):
                self.last_intent = final_state["detected_intent"]
            
            # Check if session should end
            if not final_state.get("session_active", True):
                self.session_active = False
                self.chat_history.append([
                    "System", 
                    "👋 Session ended. Say 'hello' or restart to begin again!"
                ])
            
            # Handle any errors that occurred during processing
            if final_state.get("error_message"):
                self._handle_workflow_error(final_state["error_message"])
            
        except Exception as e:
            error_msg = str(e)
            print(f"Enhanced workflow error: {error_msg}")
            
            # Check for specific TTS quota error
            if "quota_exceeded" in error_msg.lower() or "credits" in error_msg.lower():
                self._handle_tts_quota_error(error_msg)
            else:
                self.chat_history.append(["System", f"🚨 Workflow Error: {error_msg}"])
        
        finally:
            self.is_processing = False
        
        return self.chat_history
    
    def _handle_tts_response(self, final_state: dict) -> None:
        """Handle TTS response with error checking"""
        if not self.tts_enabled:
            return
            
        current_response = final_state.get("current_response", "")
        if current_response and len(current_response) > self.max_tts_chars:
            # Truncate response for TTS
            truncated = self._truncate_for_tts(current_response)
            self.chat_history.append([
                "System", 
                f"ℹ️ Response truncated for TTS (original: {len(current_response)} chars, truncated: {len(truncated)} chars)"
            ])
    
    def _handle_tts_quota_error(self, error_msg: str) -> None:
        """Handle TTS quota exceeded errors"""
        self.tts_enabled = False  # Automatically disable TTS
        self.chat_history.append([
            "System", 
            "🔇 TTS quota exceeded. Audio output disabled for this session. Text responses will continue."
        ])
        self.chat_history.append([
            "System", 
            "💡 To continue with audio: 1) Upgrade your ElevenLabs plan, 2) Use shorter responses, or 3) Wait for quota reset."
        ])
    
    def _handle_workflow_error(self, error_msg: str) -> None:
        """Handle general workflow errors"""
        if "TTS error" in error_msg and ("quota" in error_msg or "credits" in error_msg):
            self._handle_tts_quota_error(error_msg)
        else:
            self.chat_history.append(["System", f"⚠️ {error_msg}"])
    
    def _truncate_for_tts(self, text: str) -> str:
        """Truncate text to fit TTS character limits"""
        if len(text) <= self.max_tts_chars:
            return text
        
        # Try to truncate at sentence boundary
        sentences = text.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= self.max_tts_chars - 20:
                truncated += sentence + '. '
            else:
                break
        
        if not truncated:  # If even first sentence is too long
            truncated = text[:self.max_tts_chars-3] + "..."
        
        return truncated.strip()
    
    def continuous_audio_processing(self) -> Generator[List[Tuple[str, str]], None, None]:
        """Enhanced continuous audio processing with better error handling"""
        retry_count = 0
        max_retries = 3
        
        # Add welcome message
        if not self.chat_history:
            self.add_system_message(f"""
🎉 **Enhanced AI Assistant Ready!**

I now have access to powerful tools:
🔍 **Web Search** - Find information online
🌤️ **Weather** - Get current weather & forecasts  
📸 **Vision** - Analyze images from your camera
🧮 **Calculator** - Perform mathematical calculations
⏰ **Time & System** - Get current time and system info
📁 **File Manager** - Browse and read files
📰 **News** - Get latest headlines

🎤 **Audio**: {'Enabled' if self.tts_enabled else 'Disabled (quota/error)'}
🧵 **Thread ID**: {self.current_thread_id or 'New session'}

Just speak naturally and I'll use the right tools automatically!
            """)
            yield self.chat_history
        
        while self.session_active and retry_count < max_retries:
            try:
                # Process one audio cycle
                updated_history = self.process_audio_cycle()
                
                # Yield the updated chat history
                yield updated_history
                
                # Reset retry count on successful processing
                retry_count = 0
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
                # Check if session is still active
                if not self.session_active:
                    break
                    
            except Exception as e:
                retry_count += 1
                error_str = str(e)
                print(f"Continuous processing error (attempt {retry_count}): {error_str}")
                
                # Special handling for TTS quota errors
                if "quota_exceeded" in error_str.lower():
                    self._handle_tts_quota_error(error_str)
                    yield self.chat_history
                    continue  # Don't count TTS errors as retries
                
                error_msg = f"🔄 Processing error (attempt {retry_count}/{max_retries}): {error_str}"
                self.chat_history.append(["System", error_msg])
                yield self.chat_history
                
                if retry_count >= max_retries:
                    self.chat_history.append([
                        "System", 
                        "❌ Maximum retry attempts reached. Please restart the session."
                    ])
                    self.session_active = False
                    yield self.chat_history
                    break
                
                # Wait before retrying
                time.sleep(2)
    
    def clear_chat_history(self) -> List[Tuple[str, str]]:
        """Clear the chat history and reset session"""
        self.chat_history = []
        self.session_active = True
        self.conversation_count = 0
        self.last_intent = None
        # Reset workflow thread
        self.workflow.reset_workflow_thread()
        self.current_thread_id = self.workflow.thread_id
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
    
    def restart_session(self) -> Tuple[List[Tuple[str, str]], str]:
        """Restart the session with enhanced welcome message"""
        self.session_active = True
        self.is_processing = False
        self.conversation_count = 0
        self.last_intent = None
        self.tts_enabled = True  # Re-enable TTS on restart
        
        # Reset workflow thread
        self.workflow.reset_workflow_thread()
        self.current_thread_id = self.workflow.thread_id
        
        # Clear history and add enhanced welcome
        self.chat_history = []
        welcome_msg = f"""
🔄 **Session Restarted Successfully!**

🛠️ **Available Tools**: {len(tools.get_available_tools())} tools ready
📊 **Previous Conversations**: {self.conversation_count} messages processed
🧵 **New Thread ID**: {self.current_thread_id}
🎤 **TTS Status**: {'Enabled' if self.tts_enabled else 'Disabled'}

{self.workflow.get_available_tools_info()}

Ready for your next question! 🎤
        """
        
        self.add_system_message(welcome_msg)
        return self.chat_history, "✅ Session restarted with enhanced capabilities!"
    
    def toggle_tools(self) -> str:
        """Toggle tool execution on/off"""
        self.tools_enabled = not self.tools_enabled
        status = "enabled" if self.tools_enabled else "disabled"
        message = f"🛠️ Tools are now {status}"
        self.add_system_message(message)
        return message
    
    def toggle_tts(self) -> str:
        """Toggle TTS on/off"""
        self.tts_enabled = not self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        message = f"🎤 Text-to-Speech is now {status}"
        self.add_system_message(message)
        return message
    
    def set_tts_limit(self, max_chars: int) -> str:
        """Set TTS character limit"""
        self.max_tts_chars = max_chars
        message = f"🎤 TTS character limit set to {max_chars}"
        self.add_system_message(message)
        return message
    
    def get_tools_info(self) -> str:
        """Get detailed information about available tools"""
        return self.workflow.get_available_tools_info()
    
    def execute_manual_tool(self, tool_name: str, **kwargs) -> str:
        """Manually execute a tool and add result to chat"""
        try:
            result = tools.execute_tool(tool_name, **kwargs)
            self.chat_history.append([f"Tool: {tool_name.title()}", result])
            return result
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            self.chat_history.append(["System", error_msg])
            return error_msg
    
    def get_workflow_state(self) -> Optional[dict]:
        """Get current workflow state"""
        return self.workflow.get_workflow_state(self.current_thread_id)
    
    def get_enhanced_status(self) -> dict:
        """Get detailed status information including tool analytics"""
        return {
            "session_active": self.session_active,
            "is_processing": self.is_processing,
            "tools_enabled": self.tools_enabled,
            "tts_enabled": self.tts_enabled,
            "tts_char_limit": self.max_tts_chars,
            "chat_history_length": len(self.chat_history),
            "conversation_count": self.conversation_count,
            "last_intent": self.last_intent,
            "available_tools": len(tools.get_available_tools()),
            "current_thread_id": self.current_thread_id,
            "workflow_thread_id": self.workflow.thread_id,
            "last_message": self.chat_history[-1] if self.chat_history else None,
            "tool_registry_status": "Active" if tools else "Error"
        }
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.chat_history:
            return "No conversation history available."
        
        user_messages = [msg[0] for msg in self.chat_history if msg[0] not in ["System", "AI"]]
        ai_messages = [msg[1] for msg in self.chat_history if msg[0] not in ["System"]]
        
        summary = f"""
📊 **Conversation Summary**
💬 Total exchanges: {len(user_messages)}
🤖 AI responses: {len(ai_messages)}
🎯 Last detected intent: {self.last_intent or 'None'}
🛠️ Tools status: {'Enabled' if self.tools_enabled else 'Disabled'}
🎤 TTS status: {'Enabled' if self.tts_enabled else 'Disabled'}
📈 Session active: {'Yes' if self.session_active else 'No'}
🧵 Thread ID: {self.current_thread_id or 'None'}
        """
        
        return summary
    
    def export_chat_history(self, filename: str = None) -> str:
        """Export chat history to a file"""
        try:
            if not filename:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("SharryBoii AI Assistant - Enhanced Chat History\n")
                f.write(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Thread ID: {self.current_thread_id}\n")
                f.write(f"Tools Enabled: {self.tools_enabled}\n")
                f.write(f"TTS Enabled: {self.tts_enabled}\n")
                f.write("=" * 50 + "\n\n")
                
                for speaker, message in self.chat_history:
                    f.write(f"{speaker}: {message}\n\n")
            
            return f"✅ Chat history exported to {filename}"
            
        except Exception as e:
            return f"❌ Export error: {str(e)}"


# Backward compatibility
ChatManager = EnhancedChatManager