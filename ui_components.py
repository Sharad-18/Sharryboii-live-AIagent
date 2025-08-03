"""
UI components module for AI Assistant Gradio interface
"""

import gradio as gr
from typing import Callable, Any
from camera import WebcamManager
from chat_manager import ChatManager


class UIComponents:
    """Handles all UI components and their interactions"""
    
    def __init__(self):
        self.webcam_manager = WebcamManager()
        self.chat_manager = ChatManager()
        self.demo = None
    
    def create_header(self) -> gr.Markdown:
        """Create the main header"""
        return gr.Markdown("""
        <h1 style='color: orange; text-align: center; font-size: 4em;'> 
            👧🏼 SharryBoii-V1.1 –AI Assistant 
        </h1>
        <p style='text-align: center; font-size: 1.2em; color: #666;'>
            Powered by LangGraph for robust conversation flow
        </p>
        """)
    
    def create_webcam_section(self) -> tuple:
        """Create webcam section with controls"""
        with gr.Column(scale=1):
            gr.Markdown("## 📹 Webcam Feed")
            
            with gr.Row():
                start_btn = gr.Button("🟢 Start Camera", variant="primary")
                stop_btn = gr.Button("🔴 Stop Camera", variant="secondary")
            
            webcam_output = gr.Image(
                label="Live Feed",
                streaming=True,
                show_label=False,
                width=640,
                height=480
            )
            
            # Timer for webcam refresh (~30 FPS)
            webcam_timer = gr.Timer(0.033)
        
        return start_btn, stop_btn, webcam_output, webcam_timer
    
    def create_chat_section(self) -> tuple:
        """Create chat section with controls"""
        with gr.Column(scale=1):
            gr.Markdown("## 💬 Intelligent Chat Interface")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=False,
                avatar_images=("👤", "🤖")
            )
            
            gr.Markdown("""
            *🎤 **LangGraph-powered continuous listening** - speak anytime!*
            
            **Features:**
            - Robust state management
            - Advanced error handling  
            - Structured conversation flow
            - Conditional routing
            """)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                restart_btn = gr.Button("Restart Session", variant="primary")
                
            status_display = gr.Textbox(
                label="Status", 
                value="Ready to start conversation",
                interactive=False
            )
        
        return chatbot, clear_btn, restart_btn, status_display
    
    def setup_event_handlers(self, components: dict) -> None:
        """Setup all event handlers for UI components"""
        
        # Webcam event handlers
        components['start_btn'].click(
            fn=self.webcam_manager.start_webcam,
            outputs=components['webcam_output']
        )
        
        components['stop_btn'].click(
            fn=self.webcam_manager.stop_webcam,
            outputs=components['webcam_output']
        )
        
        components['webcam_timer'].tick(
            fn=self.webcam_manager.get_frame,
            outputs=components['webcam_output'],
            show_progress=False
        )
        
        # Chat event handlers
        components['clear_btn'].click(
            fn=self.chat_manager.clear_chat_history,
            outputs=components['chatbot']
        )
        
        components['restart_btn'].click(
            fn=self._restart_session,
            outputs=[components['chatbot'], components['status_display']]
        )
        
        # Auto-start continuous audio processing
        self.demo.load(
            fn=self.chat_manager.continuous_audio_processing,
            outputs=components['chatbot']
        )
    
    def _restart_session(self):
        """Restart the chat session"""
        self.chat_manager.restart_session()
        return self.chat_manager.get_chat_history(), "Session restarted successfully!"
    
    def _update_status(self) -> str:
        """Update status display"""
        status = self.chat_manager.get_status()
        if status['is_processing']:
            return "🎤 Processing audio..."
        elif status['session_active']:
            return f"Active - {status['chat_history_length']} messages"
        else:
            return "Session ended"
    
    def create_interface(self) -> gr.Blocks:
        """Create the complete Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft(), title="SharryBoii AI Assistant") as demo:
            self.demo = demo
            
            # Header
            self.create_header()
            
            with gr.Row():
                # Webcam section
                start_btn, stop_btn, webcam_output, webcam_timer = self.create_webcam_section()
                
                # Chat section  
                chatbot, clear_btn, restart_btn, status_display = self.create_chat_section()
            
            # Store components for event handlers
            components = {
                'start_btn': start_btn,
                'stop_btn': stop_btn,
                'webcam_output': webcam_output,
                'webcam_timer': webcam_timer,
                'chatbot': chatbot,
                'clear_btn': clear_btn,
                'restart_btn': restart_btn,
                'status_display': status_display
            }
            
            # Setup event handlers
            self.setup_event_handlers(components)
        
        return demo
    
    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface"""
        if self.demo is None:
            self.demo = self.create_interface()
        
        default_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": True,
            "debug": True
        }
        
        # Update with any provided kwargs
        default_kwargs.update(kwargs)
        
        self.demo.launch(**default_kwargs)