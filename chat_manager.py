"""
Enhanced chat state management module with tool integration
"""

import time
from typing import List, Tuple, Generator
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
    
    def process_audio_cycle(self) -> List[Tuple[str, str]]:
        """Single cycle of enhanced audio processing using LangGraph with tools"""
        if self.is_processing:
            return self.chat_history
        
        self.is_processing = True
        
        try:
            # Create initial state for this cycle
            initial_state = self.workflow.create_initial_state(self.chat_history.copy())
            
            # Run the enhanced workflow
            final_state = self.workflow.invoke(initial_state)
            
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
            
        except Exception as e:
            print(f"Enhanced workflow error: {e}")
            self.chat_history.append(["System", f"🚨 Workflow Error: {str(e)}"])
        
        finally:
            self.is_processing = False
        
        return self.chat_history
    
    def continuous_audio_processing(self) -> Generator[List[Tuple[str, str]], None, None]:
        """Enhanced continuous audio processing with better error handling"""
        retry_count = 0
        max_retries = 3
        
        # Add welcome message
        if not self.chat_history:
            self.add_system_message("""
🎉 **Enhanced AI Assistant Ready!**

I now have access to powerful tools:
🔍 **Web Search** - Find information online
🌤️ **Weather** - Get current weather & forecasts  
📸 **Vision** - Analyze images from your camera
🧮 **Calculator** - Perform mathematical calculations
⏰ **Time & System** - Get current time and system info
📁 **File Manager** - Browse and read files
📰 **News** - Get latest headlines

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
                print(f"Continuous processing error (attempt {retry_count}): {e}")
                
                error_msg = f"🔄 Processing error (attempt {retry_count}/{max_retries}): {str(e)}"
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
        
        # Clear history and add enhanced welcome
        self.chat_history = []
        welcome_msg = f"""
🔄 **Session Restarted Successfully!**

🛠️ **Available Tools**: {len(tools.get_available_tools())} tools ready
📊 **Previous Conversations**: {self.conversation_count} messages processed

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
    
    def get_enhanced_status(self) -> dict:
        """Get detailed status information including tool analytics"""
        return {
            "session_active": self.session_active,
            "is_processing": self.is_processing,
            "tools_enabled": self.tools_enabled,
            "chat_history_length": len(self.chat_history),
            "conversation_count": self.conversation_count,
            "last_intent": self.last_intent,
            "available_tools": len(tools.get_available_tools()),
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
📈 Session active: {'Yes' if self.session_active else 'No'}
        """
        
        return summary
    
    def export_chat_history(self, filename: str = None) -> str:
        """Export chat history to a file"""
        try:
            if not filename:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("SharryBoii AI Assistant - Chat History\n")
                f.write(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                for speaker, message in self.chat_history:
                    f.write(f"{speaker}: {message}\n\n")
            
            return f"✅ Chat history exported to {filename}"
            
        except Exception as e:
            return f"❌ Export error: {str(e)}"


# Backward compatibility
ChatManager = EnhancedChatManager