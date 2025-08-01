import os
import elevenlabs
from elevenlabs.client import ElevenLabs
import subprocess
import platform
from dotenv import load_dotenv
load_dotenv()
ElevenLabs_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech_with_eleven_lab(input_text, output_file) -> str:
    client=ElevenLabs(api_key=ElevenLabs_API_KEY)
    audio = client.text_to_speech.convert(text=input_text, voice_id=os.getenv("Voice_id"),model_id="eleven_multilingual_v2",
                                          output_format="mp3_22050_32",)
    elevenlabs.save(audio,output_file)
    os_name=platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_file])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_file])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


from gtts import gTTS

def text_to_speech_with_gtts(input_text, output_file):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_file)
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_file])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_file])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


# input_text = "Hi, I am doing fine, how are you? This is a test for AI with text to speech."
# output_filepath = "test_text_to_speech.mp3"
# text_to_speech_with_eleven_lab(input_text, output_filepath)
# text_to_speech_with_gtts(input_text, output_filepath)