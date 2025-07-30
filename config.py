"""
Configuration settings for AI Assistant
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    audio_file: str = "audio_question.mp3"
    output_file: str = "final.mp3"
    sample_rate: int = 16000
    chunk_size: int = 1024


@dataclass
class CameraConfig:
    """Camera configuration"""
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class UIConfig:
    """UI configuration"""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = True
    debug: bool = True
    theme: str = "soft"
    webcam_refresh_rate: float = 0.033  # ~30 FPS


@dataclass
class APIConfig:
    """API configuration"""
    groq_api_key: Optional[str] = None
    eleven_labs_api_key: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment variables if not provided
        if self.groq_api_key is None:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
        if self.eleven_labs_api_key is None:
            self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")


@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    max_retries: int = 3
    retry_delay: float = 1.0
    processing_timeout: int = 30
    exit_keywords: list = None
    
    def __post_init__(self):
        if self.exit_keywords is None:
            self.exit_keywords = ["goodbye", "bye", "exit", "quit", "stop"]


class AppConfig:
    """Main application configuration"""
    
    def __init__(self):
        self.audio = AudioConfig()
        self.camera = CameraConfig()
        self.ui = UIConfig()
        self.api = APIConfig()
        self.workflow = WorkflowConfig()
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check API keys
        if not self.api.groq_api_key:
            errors.append("GROQ_API_KEY is not set")
        
        # Check audio settings
        if self.audio.sample_rate <= 0:
            errors.append("Audio sample rate must be positive")
        
        # Check camera settings
        if self.camera.width <= 0 or self.camera.height <= 0:
            errors.append("Camera dimensions must be positive")
        
        # Check UI settings
        if not (1024 <= self.ui.server_port <= 65535):
            errors.append("Server port must be between 1024 and 65535")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("=== AI Assistant Configuration ===")
        print(f"Audio file: {self.audio.audio_file}")
        print(f"Camera: {self.camera.width}x{self.camera.height}@{self.camera.fps}fps")
        print(f"Server: {self.ui.server_name}:{self.ui.server_port}")
        print(f"GROQ API Key: {'✓ Set' if self.api.groq_api_key else '✗ Missing'}")
        print(f"Eleven Labs API Key: {'✓ Set' if self.api.eleven_labs_api_key else '✗ Missing'}")
        print("==================================")


# Global configuration instance
config = AppConfig()