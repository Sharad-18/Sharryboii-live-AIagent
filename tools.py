"""
Enhanced tools module for AI Assistant with vision, weather, search, and more
"""

import cv2
import base64
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from groq import Groq
import subprocess
import platform
import psutil

load_dotenv()


class VisionTool:
    """Computer vision analysis tool"""
    
    def __init__(self):
        self.client = Groq()
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    def capture_image(self) -> str:
        """Captures one frame from the default webcam, resize it, encodes it as Base64 JPEG"""
        for idx in range(4):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                # Let camera warm up
                for _ in range(10):
                    cap.read()
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    continue
                    
                # Save a copy for debugging
                cv2.imwrite('sample.jpg', frame)
                
                # Encode as JPEG
                ret, buf = cv2.imencode('.jpg', frame)
                if ret:
                    return base64.b64encode(buf).decode('utf-8')
        
        raise RuntimeError("No camera found or unable to capture image.")
    
    def analyze_image(self, query: str) -> str:
        """Analyze captured image with given query using Groq's vision API"""
        try:
            img_b64 = self.capture_image()
            
            if not query or not img_b64:
                return "Error: both 'query' and image must be provided."
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            }]
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Vision analysis error: {str(e)}"


class WeatherTool:
    """Weather information tool"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, city: str, country: str = None) -> str:
        """Get current weather for a city"""
        try:
            if not self.api_key:
                return "Weather API key not configured. Set OPENWEATHER_API_KEY environment variable."
            
            location = f"{city},{country}" if country else city
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather = {
                "location": f"{data['name']}, {data['sys']['country']}",
                "temperature": f"{data['main']['temp']}°C",
                "feels_like": f"{data['main']['feels_like']}°C",
                "description": data['weather'][0]['description'].title(),
                "humidity": f"{data['main']['humidity']}%",
                "pressure": f"{data['main']['pressure']} hPa",
                "wind_speed": f"{data['wind']['speed']} m/s"
            }
            
            return f"""Current weather in {weather['location']}:
🌡️ Temperature: {weather['temperature']} (feels like {weather['feels_like']})
🌤️ Conditions: {weather['description']}
💧 Humidity: {weather['humidity']}
🌪️ Wind: {weather['wind_speed']}
📊 Pressure: {weather['pressure']}"""
            
        except requests.exceptions.RequestException as e:
            return f"Weather API error: {str(e)}"
        except KeyError as e:
            return f"Weather data parsing error: {str(e)}"
        except Exception as e:
            return f"Weather error: {str(e)}"
    
    def get_weather_forecast(self, city: str, days: int = 3) -> str:
        """Get weather forecast for a city"""
        try:
            if not self.api_key:
                return "Weather API key not configured."
            
            url = f"{self.base_url}/forecast"
            params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric",
                "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast_text = f"📅 {days}-day forecast for {data['city']['name']}:\n\n"
            
            current_date = None
            for item in data['list'][:days*8:8]:  # Take one per day
                date = datetime.fromtimestamp(item['dt'])
                if date.date() != current_date:
                    current_date = date.date()
                    forecast_text += f"📆 {date.strftime('%A, %B %d')}:\n"
                    forecast_text += f"   🌡️ {item['main']['temp']}°C - {item['weather'][0]['description'].title()}\n"
                    forecast_text += f"   💧 Humidity: {item['main']['humidity']}%\n\n"
            
            return forecast_text
            
        except Exception as e:
            return f"Weather forecast error: {str(e)}"


class SearchTool:
    """Internet search tool using DuckDuckGo"""
    
    def __init__(self):
        self.search_url = "https://api.duckduckgo.com/"
    
    def search_web(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo API"""
        try:
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Get instant answer if available
            if data.get("Answer"):
                results.append(f"💡 Quick Answer: {data['Answer']}")
            
            # Get abstract if available
            if data.get("Abstract"):
                results.append(f"📝 Summary: {data['Abstract']}")
            
            # Get related topics
            if data.get("RelatedTopics"):
                results.append("🔗 Related Topics:")
                for topic in data["RelatedTopics"][:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(f"   • {topic['Text']}")
            
            if not results:
                return f"No specific results found for '{query}'. Try rephrasing your search."
            
            return f"🔍 Search results for '{query}':\n\n" + "\n\n".join(results)
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def get_news_headlines(self, topic: str = "technology") -> str:
        """Get news headlines (simplified version)"""
        try:
            # Using a simple news API alternative
            query = f"{topic} news today"
            return self.search_web(query, max_results=3)
        except Exception as e:
            return f"News error: {str(e)}"


class SystemTool:
    """System information and control tool"""
    
    def get_system_info(self) -> str:
        """Get basic system information"""
        try:
            info = {
                "OS": f"{platform.system()} {platform.release()}",
                "CPU": platform.processor(),
                "CPU Usage": f"{psutil.cpu_percent(interval=1)}%",
                "Memory": f"{psutil.virtual_memory().percent}% used",
                "Disk": f"{psutil.disk_usage('/').percent}% used",
                "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = "💻 System Information:\n"
            for key, value in info.items():
                result += f"   {key}: {value}\n"
            
            return result
            
        except Exception as e:
            return f"System info error: {str(e)}"
    
    def get_current_time(self, timezone: str = None) -> str:
        """Get current time"""
        try:
            now = datetime.now()
            return f"🕐 Current time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}"
        except Exception as e:
            return f"Time error: {str(e)}"
    
    def set_reminder(self, message: str, minutes: int) -> str:
        """Set a simple reminder (placeholder - would need proper implementation)"""
        future_time = datetime.now() + timedelta(minutes=minutes)
        return f"⏰ Reminder set for {future_time.strftime('%I:%M %p')}: {message}"


class CalculatorTool:
    """Simple calculator tool"""
    
    def calculate(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Remove any potentially dangerous characters
            allowed_chars = "0123456789+-*/.() "
            cleaned_expression = ''.join(c for c in expression if c in allowed_chars)
            
            if not cleaned_expression:
                return "Invalid mathematical expression"
            
            result = eval(cleaned_expression)
            return f"🧮 {expression} = {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Calculation error: {str(e)}"


class FileManagerTool:
    """File management tool"""
    
    def list_files(self, directory: str = ".") -> str:
        """List files in a directory"""
        try:
            files = os.listdir(directory)
            files.sort()
            
            result = f"📁 Files in '{directory}':\n"
            for file in files[:20]:  # Limit to 20 files
                path = os.path.join(directory, file)
                if os.path.isdir(path):
                    result += f"   📂 {file}/\n"
                else:
                    size = os.path.getsize(path)
                    result += f"   📄 {file} ({size} bytes)\n"
            
            if len(files) > 20:
                result += f"   ... and {len(files) - 20} more files\n"
            
            return result
            
        except Exception as e:
            return f"File listing error: {str(e)}"
    
    def read_file(self, filename: str, max_chars: int = 1000) -> str:
        """Read content from a text file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read(max_chars)
            
            result = f"📄 Content of '{filename}':\n\n{content}"
            
            if len(content) == max_chars:
                result += f"\n\n... (truncated to {max_chars} characters)"
            
            return result
            
        except Exception as e:
            return f"File reading error: {str(e)}"


class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self.vision = VisionTool()
        self.weather = WeatherTool()
        self.search = SearchTool()
        self.system = SystemTool()
        self.calculator = CalculatorTool()
        self.files = FileManagerTool()
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools and their descriptions"""
        return {
            "vision": "Analyze images from webcam with AI vision",
            "weather": "Get current weather and forecasts",
            "search": "Search the internet for information",
            "news": "Get latest news headlines",
            "system": "Get system information and current time",
            "calculator": "Perform mathematical calculations",
            "files": "List and read files",
            "reminder": "Set simple reminders"
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool with given parameters"""
        try:
            if tool_name == "vision":
                query = kwargs.get("query", "What do you see in this image?")
                return self.vision.analyze_image(query)
            
            elif tool_name == "weather":
                city = kwargs.get("city", "London")
                country = kwargs.get("country")
                return self.weather.get_current_weather(city, country)
            
            elif tool_name == "forecast":
                city = kwargs.get("city", "London")
                days = kwargs.get("days", 3)
                return self.weather.get_weather_forecast(city, days)
            
            elif tool_name == "search":
                query = kwargs.get("query", "")
                if not query:
                    return "Please provide a search query"
                return self.search.search_web(query)
            
            elif tool_name == "news":
                topic = kwargs.get("topic", "technology")
                return self.search.get_news_headlines(topic)
            
            elif tool_name == "system":
                return self.system.get_system_info()
            
            elif tool_name == "time":
                return self.system.get_current_time()
            
            elif tool_name == "calculator":
                expression = kwargs.get("expression", "")
                if not expression:
                    return "Please provide a mathematical expression"
                return self.calculator.calculate(expression)
            
            elif tool_name == "files":
                directory = kwargs.get("directory", ".")
                return self.files.list_files(directory)
            
            elif tool_name == "read_file":
                filename = kwargs.get("filename", "")
                if not filename:
                    return "Please provide a filename"
                return self.files.read_file(filename)
            
            elif tool_name == "reminder":
                message = kwargs.get("message", "")
                minutes = kwargs.get("minutes", 5)
                return self.system.set_reminder(message, minutes)
            
            else:
                available = ", ".join(self.get_available_tools().keys())
                return f"Unknown tool '{tool_name}'. Available tools: {available}"
        
        except Exception as e:
            return f"Tool execution error: {str(e)}"


# Global tool registry instance
tools = ToolRegistry()


# Convenience functions for backward compatibility
def capture_image() -> str:
    """Backward compatibility function"""
    return tools.vision.capture_image()


def analyze_image(query: str) -> str:
    """Backward compatibility function"""
    return tools.vision.analyze_image(query)


# Example usage and testing
if __name__ == "__main__":
    print("🛠️  Testing AI Assistant Tools...")
    
    # Test system info
    print("\n" + tools.execute_tool("system"))
    
    # Test calculator
    print("\n" + tools.execute_tool("calculator", expression="2 + 2 * 3"))
    
    # Test weather (requires API key)
    print("\n" + tools.execute_tool("weather", city="New York"))
    
    # Test search
    print("\n" + tools.execute_tool("search", query="Python programming"))
    
    # List available tools
    print("\n🔧 Available tools:")
    for tool, description in tools.get_available_tools().items():
        print(f"   {tool}: {description}")
# import cv2
# import base64
# from dotenv import load_dotenv

# load_dotenv()

# def capture_image()->str:
#     """ captures one frame from the default webcame, resize it, encodes it as Base64 JPEG (raw string) and returns it"""
#     for idx in range(4):
#         cap=cv2.VideoCapture(idx,cv2.CAP_AVFOUNDATION)
#         if cap.isOpened():
#             for _ in range(10):
#                 cap.read()
#             ret,frame=cap.read()
#             cap.release()
#             if not ret:
#                 continue
#             cv2.imwrite('sample.jpg',frame)
#             rer,buf=cv2.imencode('.jpg',frame)
#             if ret:
#                 return base64.b64encode(buf).decode('utf-8')
#     raise RuntimeError("No camera found or unable to capture image.")

# from groq import Groq

# def analyze_image(query:str)->str:
#     """Expects a string with 'query Capture the image and sends the qury and 
#     the image and sends the qury and the image to Groq's vision chat API and returns the analysis.
#     """
#     img_b64=capture_image()
#     model="meta-llama/llama-4-maverick-17b-128e-instruct"
#     if not query or not img_b64:
#         return "Error: both 'qury' and 'img_b64' must be provided." 
#     client = Groq()
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text", 
#                     "text": query
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{img_b64}",
#                     },
#                 },
#             ],
#         }]
#     chat_completion=client.chat.completions.create(
#         messages=messages,
#         model=model
#     )

#     return chat_completion.choices[0].message.content
# # query = "How many people do you see?"
# # print(analyze_image(query))