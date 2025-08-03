"""
Enhanced main application entry point for SharryBoii AI Assistant
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


from ui_components import UIComponents
from config import config
from tools import tools
import logging
# except ImportError as e:
#     print(f"Import error: {e}")
#     print("Please ensure all required modules are in the same directory:")
#     print("- updated_workflow.py")
#     print("- camera.py") 
#     print("- updated_chat_manager.py")
#     print("- ui_components.py")
#     print("- config.py")
#     print("- tools.py")
#     print("- speech_to_text.py")
#     print("- ai_agents.py")
#     print("- text_to_speech.py")
#     sys.exit(1)


def setup_logging():
    """Setup enhanced logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.api.groq_api_key and 'INFO' or 'WARNING'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_assistant.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress some verbose logging
    logging.getLogger('gradio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'gradio',
        'opencv-python', 
        'langgraph',
        'langchain',
        'numpy',
        'requests',
        'psutil',
        'groq'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'langgraph':
                import langgraph
            elif package == 'langchain':
                import langchain
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_api_keys():
    """Check if required API keys are configured"""
    api_status = {
        "GROQ_API_KEY": bool(config.api.groq_api_key),
        "ELEVEN_LABS_API_KEY": bool(config.api.eleven_labs_api_key),
        "OPENWEATHER_API_KEY": bool(os.getenv("OPENWEATHER_API_KEY"))
    }
    
    print("\n🔑 API Key Status:")
    for key, status in api_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {key}: {'Configured' if status else 'Missing'}")
    
    # Only GROQ is required, others are optional
    if not api_status["GROQ_API_KEY"]:
        print("\n Warning: GROQ_API_KEY is required for AI responses")
        print("   Set it in your .env file or environment variables")
        return False
    
    if not api_status["ELEVEN_LABS_API_KEY"]:
        print("\n💡 Note: Text-to-speech will use fallback without ELEVEN_LABS_API_KEY")
    
    if not api_status["OPENWEATHER_API_KEY"]:
        print("💡 Note: Weather features disabled without OPENWEATHER_API_KEY")
    
    return True


def test_tools():
    """Test tool functionality"""
    print("\n🛠️  Testing AI Assistant Tools...")
    
    # Test basic tools
    try:
        # Test system info
        system_result = tools.execute_tool("system")
        print("System tool: Working")
        
        # Test calculator
        calc_result = tools.execute_tool("calculator", expression="2+2")
        print("Calculator tool: Working")
        
        # Test weather (might fail without API key)
        try:
            weather_result = tools.execute_tool("weather", city="London")
            print("Weather tool: Working")
        except:
            print("Weather tool: Requires API key")
        
        # Test search
        try:
            search_result = tools.execute_tool("search", query="test")
            print("Search tool: Working")
        except:
            print("Search tool: Limited functionality")
        
        print(f"🎯 Total tools available: {len(tools.get_available_tools())}")
        
    except Exception as e:
        print(f"Tool testing error: {e}")
        return False
    
    return True


def print_startup_banner():
    """Print enhanced startup banner"""
    banner = f"""
╭─────────────────────────────────────────────────────────────╮
│                                                             │
│  🤖 SharryBoii - AI Assistant v2.0                │
│                                                             │
│  🎯 Features:                                               │
│     • LangGraph-powered conversation flow                   │
│     • Multi-tool integration ({len(tools.get_available_tools())} tools available)           │
│     • Real-time webcam integration                          │
│     • Voice conversation with TTS                           │
│     • Weather, search, vision, and more!                    │
│                                                             │
│  🌐 Server: {config.ui.server_name}:{config.ui.server_port}                               │
│  🔧 Debug: {'Enabled' if config.ui.debug else 'Disabled'}                                │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
    """
    print(banner)


def main():
    """Enhanced main application function"""
    print_startup_banner()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    # print("🔍 Checking dependencies...")
    # if not check_dependencies():
    #     logger.error("Missing dependencies. Please install required packages.")
    #     sys.exit(1)
    
    # Check API keys
    print("🔑 Checking API configuration...")
    if not check_api_keys():
        logger.error("Required API keys missing.")
        response = input("\nContinue anyway? Some features will be limited. (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Validate configuration
    print("Validating configuration...")
    if not config.validate():
        logger.error("Configuration validation failed.")
        sys.exit(1)
    
    # Test tools
    print("🛠️  Testing tool functionality...")
    test_tools()
    
    # Print configuration
    config.print_config()
    
    try:
        # Create UI components
        logger.info("Initializing enhanced UI components...")
        ui = UIComponents()
        
        # Create interface
        logger.info("🖥️  Creating Gradio interface...")
        demo = ui.create_interface()
        
        # Launch application
        logger.info("🚀 Launching enhanced AI assistant...")
        print(f"\n🌟 Starting server at http://{config.ui.server_name}:{config.ui.server_port}")
        print("💡 Tip: Try saying things like:")
        print("   • 'What's the weather in New York?'")
        print("   • 'Search for Python tutorials'")
        print("   • 'What do you see?' (uses camera)")
        print("   • 'Calculate 25 times 4'")
        print("   • 'What time is it?'")
        
        ui.launch(
            server_name=config.ui.server_name,
            server_port=config.ui.server_port,
            share=config.ui.share,
            debug=config.ui.debug
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Thanks for using SharryBoii Enhanced AI Assistant!")
        print("Your conversation history has been preserved.")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Critical Error: {e}")
        print("Check the log file 'ai_assistant.log' for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()