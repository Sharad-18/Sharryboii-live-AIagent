# 🤖 SharryBoii – The Witty AI Assistant with Vision, Tools, and Personality

**SharryBoii** is your AI-powered sidekick that *sees*, *thinks*, and *responds with sass*.  
Built using LangChain + LangGraph, integrated with computer vision, search, weather, file I/O, system stats, and more — all wrapped in a natural, chatty interface.

---

## ✨ Features

- 🎥 **Webcam Vision** – Ask it:  
  > “Do I look tired today?”  
  > “Do I have a beard right now?”  
  Captures a live image via webcam and responds using Groq’s LLaMA Vision model.

- 🌦️ **Real-Time Weather**  
- 🔎 **Internet Search + News Headlines**  
- 🧮 **Math Evaluator / Calculator**  
- 🧠 **Dynamic Tool Routing (ReAct via LangGraph)**  
- 💾 **File Reading & Directory Listing**  
- 💻 **System Stats (CPU, Memory, Time)**  
- ⏰ **Reminders (Prototype)**

---

## 🧠 Powered By

- [LangChain](https://www.langchain.com/)
- [LangGraph](https://www.langgraph.dev/)
- [Groq Vision (LLaMA 4)](https://groq.com/)
- [Google Gemini API](https://ai.google.dev/)
- OpenWeather API
- DuckDuckGo Instant Answer API
- Python + OpenCV + psutil

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Sharad-18/Sharryboii-live-AIagent.git
uv sync
```
That’s it. This command:
<br>
Creates a virtual environment (if one doesn't exist)
<br>
Installs all dependencies from uv.lock
<br>
Sets up everything exactly as expected
