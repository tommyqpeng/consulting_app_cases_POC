# -*- coding: utf-8 -*-
"""
Updated on June 20, 2025
For flat image structure and separate encrypted metadata file
"""

import json
import requests
import os
import re
from cryptography.fernet import Fernet
import streamlit as st

# --- Decrypt Encrypted JSON File ---
def decrypt_file(encrypted_path, decryption_key):
    """
    Decrypts an encrypted JSON file using Fernet and returns parsed metadata as a dict.
    """
    with open(encrypted_path, "rb") as f:
        encrypted = f.read()
    fernet = Fernet(decryption_key)
    decrypted = fernet.decrypt(encrypted)
    return json.loads(decrypted)

# --- Get specific question ---
def get_question(case_data, case_id, question_id):
    """
    Fetches the question dict given case_data, case_id, and question_id.
    """
    return case_data[str(case_id)]["questions"][str(question_id)]

# --- Build Prompt for Model ---
def build_prompt(question_text, rubric, examples=None, user_input="", generation_instructions=""):
    """
    Builds the full prompt for the model including rubric, examples (if any), and candidate input.
    """
    retrieved_text = ""
    if examples:
        retrieved_text = "\n".join(
            f"Past Answer: {item['answer']}\nFeedback Given: {item['feedback']}\n"
            for item in examples
        )

    return f"""
Case Question:
{question_text}

Candidate's Answer:
{user_input}

{"Historical Examples:\n" + retrieved_text if retrieved_text else ""}

Rubric:
{rubric}

{generation_instructions}
"""

# --- Call DeepSeek API ---
def generate_feedback(prompt, system_role, api_key, temperature=0.4):
    """
    Sends the prompt and system role to the DeepSeek API with specified temperature.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[DeepSeek API Error] {e}")
        print("Response text:", getattr(response, "text", "No response"))
        return None

# --- Transcribe Audio with Deepgram ---
def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    response = requests.post(
        "https://api.deepgram.com/v1/listen",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav"
        },
        data=audio_bytes
    )
    if response.status_code == 200:
        return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    else:
        raise RuntimeError(f"Transcription failed: {response.text}")

# --- Render Question with Embedded Images ---
def render_question_with_images(text, image_dir="images"):
    """
    Parses question text with {{img:filename}} placeholders and renders using Streamlit.
    """
    parts = re.split(r"(\{\{img:[^}]+\}\})", text)

    for part in parts:
        if part.startswith("{{img:") and part.endswith("}}"):
            img_file = part[6:-2].strip()
            img_path = os.path.join(image_dir, img_file)
            if os.path.exists(img_path):
                st.image(img_path)
            else:
                st.warning(f"⚠️ Image not found: {img_file}")
        else:
            st.markdown(part)
