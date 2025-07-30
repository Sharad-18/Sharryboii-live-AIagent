# agents.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os

from tools import tools  # ToolRegistry instance

# Load environment variables
load_dotenv()
gemini_api = os.getenv("GEMINI_API_KEY")

# ✨ Agent persona prompt
system_prompts = """
You are SharryBoii - a witty, clever, and helpful assistant.
Here’s how you operate:
- FIRST and FOREMOST, figure out from the query asked whether it requires a look via the webcam to be answered. If yes, call the analyze_image tool immediately and proceed.
- Don’t ask for permission to look through the webcam; just call the tool straight away.
- When a question could only be answered by taking a photo, always use the analyze_image tool.
- Always present the results in a natural, witty, and human-sounding way — like Dora herself is speaking.
Your job is to make every interaction feel smart, snappy, and personable. Got it? Let’s charm your master!
"""

# 🔮 Gemini model setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    api_key=gemini_api
)

# 🛠️ LangChain-compatible tool wrappers

@tool
def analyze_image(query: str) -> str:
    """Use the webcam to analyze an image with a query."""
    return tools.vision.analyze_image(query)

@tool
def get_weather(city: str = "London", country: str = None) -> str:
    """Get current weather for a city (and optional country)."""
    return tools.weather.get_current_weather(city, country)

@tool
def get_forecast(city: str = "London", days: int = 3) -> str:
    """Get multi-day forecast for a city."""
    return tools.weather.get_weather_forecast(city, days)

@tool
def search_web(query: str) -> str:
    """Search the internet using DuckDuckGo."""
    return tools.search.search_web(query)

@tool
def get_news(topic: str = "technology") -> str:
    """Get the latest news headlines for a topic."""
    return tools.search.get_news_headlines(topic)

@tool
def get_system_info() -> str:
    """Get system information: OS, CPU, memory, etc."""
    return tools.system.get_system_info()

@tool
def get_time() -> str:
    """Get the current system time."""
    return tools.system.get_current_time()

@tool
def set_reminder(message: str, minutes: int = 5) -> str:
    """Set a reminder with a message for a future time."""
    return tools.system.set_reminder(message, minutes)

@tool
def do_math(expression: str) -> str:
    """Calculate a math expression (e.g., 2+2*5)."""
    return tools.calculator.calculate(expression)

@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory (default: current folder)."""
    return tools.files.list_files(directory)

@tool
def read_file(filename: str, max_chars: int = 1000) -> str:
    """Read a file's content (first 1000 chars by default)."""
    return tools.files.read_file(filename, max_chars)

# 🧰 Collect all tools
toolset = [
    analyze_image,
    get_weather,
    get_forecast,
    search_web,
    get_news,
    get_system_info,
    get_time,
    set_reminder,
    do_math,
    list_files,
    read_file,
]

# 💬 Entry function
def ask_agent(user_query: str) -> str:
    """Ask the assistant a question and get a witty, helpful reply."""
    agent = create_react_agent(
        model=llm,
        tools=toolset,
        prompt=system_prompts
    )

    input_messages = {"messages": [{"role": "user", "content": user_query}]}
    response = agent.invoke(input_messages)
    return response['messages'][-1].content

# Example test
# print(ask_agent("Do I look tired today?"))  # Will use webcam
# print(ask_agent("What's the weather like in Tokyo?"))

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.prebuilt import create_react_agent
# from dotenv import load_dotenv
# import os
# from tools import VisionTool,
# load_dotenv()
# gemini_api=os.getenv("GEMINI_API_KEY")

# system_prompts="""
# You are SharryBoii - a witty, clever, and helpful assistant.
# Here’s how you operate:
#         - FIRST and FOREMOST, figure out from the query asked whether it requires a look via the webcam to be answered, if yes call the analyze_image_with_query tool for it and proceed.
#         - Dont ask for permission to look through the webcam, or say that you need to call the tool to take a peek, call it straight away, ALWAYS call the required tools have access to take a picture.
#         - When the user asks something which could only be answered by taking a photo, then call the analyze_image_with_query tool.
#         - Always present the results (if they come from a tool) in a natural, witty, and human-sounding way — like Dora herself is speaking, not a machine.
#     Your job is to make every interaction feel smart, snappy, and personable. Got it? Let’s charm your master!"
# """
# llm=ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.7,
#     api_key=gemini_api
# )

# def ask_agent(user_query:str)->str:
#     """Asks the agent a question and returns the answer."""
#     agent = create_react_agent(
#         model=llm,
#         tools=[analyze_image],
#         prompt=system_prompts
#         )

#     input_messages = {"messages": [{"role": "user", "content": user_query}]}

#     response = agent.invoke(input_messages)

#     return response['messages'][-1].content


# # print(ask_agent(user_query="Do I have a beard?"))