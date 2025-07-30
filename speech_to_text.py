import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path,timeout=20,phrase_time_limit=None):
    """
    Function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_lfimit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source,duration=1)
            logging.info("Start speaking now...")

            # record the audio 
            audio_data= recognizer.listen(source=source,timeout=timeout,phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            # convert audio to MP3 format
            wav_data=audio_data.get_wav_data()
            audio_segment=AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3",bitrate="128k")
            logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while recording audio: {e}")
import os
from groq import Groq

def transcribe_audio(file_path):
    """
    Function to transcribe audio from an MP3 file using Groq's vision chat API.

    Args:
    file_path (str): Path to the MP3 file to be transcribed.

    Returns:
    str: Transcription of the audio.
    """
    GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    client=Groq(api_key=GROQ_API_KEY)
    stt_model="whisper-large-v3"
    audio_file=open(file_path, "rb")
    transcription=client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
    )

    return transcription.text
# audio_filepath = "test_speech_to_text.mp3"
# record_audio(audio_filepath)
# try:
#     transcript = transcribe_audio(audio_filepath)
#     print("Transcription:", transcript)
# except FileNotFoundError:
#     print("Error: Audio file not found.")
# except Exception as e:
#     print(f"Transcription failed: {e}")