import cv2
import base64
from dotenv import load_dotenv

load_dotenv()

def capture_image()->str:
    """ captures one frame from the default webcame, resize it, encodes it as Base64 JPEG (raw string) and returns it"""
    for idx in range(4):
        cap=cv2.VideoCapture(idx,cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            for _ in range(10):
                cap.read()
            ret,frame=cap.read()
            cap.release()
            if not ret:
                continue
            cv2.imwrite('sample.jpg',frame)
            rer,buf=cv2.imencode('.jpg',frame)
            if ret:
                return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError("No camera found or unable to capture image.")

from groq import Groq

def analyze_image(query:str)->str:
    """Expects a string with 'query Capture the image and sends the qury and 
    the image and sends the qury and the image to Groq's vision chat API and returns the analysis.
    """
    img_b64=capture_image()
    model="meta-llama/llama-4-maverick-17b-128e-instruct"
    if not query or not img_b64:
        return "Error: both 'qury' and 'img_b64' must be provided." 
    client = Groq()
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content
# query = "How many people do you see?"
# print(analyze_image(query))