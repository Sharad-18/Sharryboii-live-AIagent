"""
Main application entry point for SharryBoii AI Assistant
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from ui_components import UIComponents
    from config import config
    import logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the same directory:")
    print("- workflow.py")
    print("- camera.py") 
    print("- chat_manager.py")
    print("- ui_components.py")
    print("- config.py")
    print("- speech_to_text.py")
    print("- ai_agents.py")
    print("- text_to_speech.py")
    sys.exit(1)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_assistant.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'gradio',
        'opencv-python', 
        'langgraph',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'langgraph':
                import langgraph
            # elif package == 'langchain':
            #     import langchain
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Main application function"""
    print("🚀 Starting SharryBoii AI Assistant...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    # if not check_dependencies():
    #     logger.error("Missing dependencies. Please install required packages.")
    #     sys.exit(1)
    
    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed.")
        sys.exit(1)
    
    # Print configuration
    config.print_config()
    
    try:
        # Create UI components
        logger.info("Initializing UI components...")
        ui = UIComponents()
        
        # Create interface
        logger.info("Creating Gradio interface...")
        demo = ui.create_interface()
        
        # Launch application
        logger.info("Launching application...")
        ui.launch(
            server_name=config.ui.server_name,
            server_port=config.ui.server_port,
            share=config.ui.share,
            debug=config.ui.debug
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Goodbye! Thanks for using SharryBoii AI Assistant!")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()