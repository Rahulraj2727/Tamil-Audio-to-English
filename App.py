import os
import gradio as gr
import requests
import io
from PIL import Image
from groq import Groq

# API Keys
GROQ_API_KEY = "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"
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Initialize Groq API client
groq_client = Groq(api_key=GROQ_API_KEY)

# Function: Tamil Audio → Tamil Text
def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.wav", audio_file.read()), model="whisper-large-v3", language="ta"
            )
        return transcription.text
    except Exception as e:
        return f"Error: {e}"

# Function: Tamil Text → English Translation
def translate_tamil_to_english(tamil_text):
    try:
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": f"Translate to English: {tamil_text}"}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Function: English Text → Image
def generate_image_from_text(english_text):
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
            json={"inputs": english_text},
        )
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        return f"Error: {e}"

# Function: English Text → Further Text Generation
def generate_text_from_prompt(english_text):
    try:
        response = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": english_text}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Main Processing Function
def process_audio_file(audio_file_path):
    tamil_text = transcribe_audio(audio_file_path)
    english_text = translate_tamil_to_english(tamil_text)
    generated_image = generate_image_from_text(english_text)
    generated_text = generate_text_from_prompt(english_text)
    return tamil_text, english_text, generated_image, generated_text

# Gradio Interface
iface = gr.Interface(
    fn=process_audio_file,
    inputs=gr.Audio(type="filepath", label="Upload Tamil Audio File"),
    outputs=[
        gr.Textbox(label="Transcribed Tamil Text"),
        gr.Textbox(label="Translated English Text"),
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Generated Text"),
    ],
    title="Tamil Audio to AI Processing Pipeline",
)

iface.launch()
