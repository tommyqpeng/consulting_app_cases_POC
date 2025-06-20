# -*- coding: utf-8 -*-
"""
Case interview app with voice/text input, feedback generation,
and FAISS-based retrieval of historical examples.
"""

import streamlit as st
import gspread
import json
import re
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from util_functions import (
    decrypt_file,
    transcribe_audio,
    build_prompt,
    generate_feedback,
    render_question_with_images
)
from faiss_lookup import EncryptedAnswerRetriever
from st_audiorec import st_audiorec

# --- Config and Secrets ---
APP_PASSWORD = st.secrets["APP_PASSWORD"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
DECRYPTION_KEY = st.secrets["DECRYPTION_KEY"].encode()
DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
ENCRYPTED_PATH = "case_questions.json.encrypted"
FAISS_INDEX_PATH = "faiss_index.encrypted"
FAISS_META_PATH = "metadata.encrypted"

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GSHEET_CREDS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(st.secrets["AnswerStorage_Sheet_ID"]).sheet1

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "submitted_questions" not in st.session_state:
    st.session_state.submitted_questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# --- Load Data ---
case_data = decrypt_file(ENCRYPTED_PATH, DECRYPTION_KEY)

retriever = EncryptedAnswerRetriever(
    encrypted_index_path=FAISS_INDEX_PATH,
    encrypted_meta_path=FAISS_META_PATH,
    decryption_key=DECRYPTION_KEY
)

# --- UI Title ---
st.title("Case Interview Submission")

# --- Password Authentication ---
if not st.session_state.authenticated:
    password = st.text_input("Enter access password", type="password")
    if st.button("Submit Password"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.warning("Incorrect password.")
            st.stop()

# --- Welcome Message and Email Info ---
st.markdown("""
### How It Works

You will answer one case interview question at a time, using either voice or text.

Once you've completed all questions, your responses will be reviewed by an **ex-McKinsey interview coach**.

**You’ll receive personalized written feedback within 48 hours via email.**
""")

# --- Name and Email ---
if not st.session_state.user_name or not st.session_state.user_email:
    st.subheader("Your Details")
    st.session_state.user_name = st.text_input("Your name")
    st.session_state.user_email = st.text_input("Your email address")
    email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"

    if not st.session_state.user_name or not st.session_state.user_email:
        st.info("Please enter your name and email to continue.")
        st.stop()
    if not re.match(email_pattern, st.session_state.user_email):
        st.warning("Please enter a valid email address.")
        st.stop()

# --- Select Case with Button Grid ---
if "selected_case_id" not in st.session_state:
    st.subheader("Choose a Case")
    case_ids = sorted(case_data.keys())
    cols = st.columns(min(len(case_ids), 3))

    for i, cid in enumerate(case_ids):
        case_title = case_data[cid]["case_title"]
        with cols[i % 3]:
            if st.button(f"Start: {case_title}", key=f"casebtn_{cid}"):
                st.session_state.selected_case_id = cid
                st.rerun()
    st.stop()

# --- Active Case ---
case_id = st.session_state.selected_case_id
case = case_data[case_id]
questions = list(case["questions"].items())

# --- All Questions Completed ---
if st.session_state.current_question >= len(questions):
    st.success("✅ You have completed all questions. Thank you!")
    st.stop()

# --- Show Case Text ---
st.markdown(f"### Case: {case['case_title']}")
st.markdown(case["case_text"])

# --- Current Question ---
question_id, question_obj = questions[st.session_state.current_question]
st.markdown("---")
st.markdown(f"#### Question {question_id}")
render_question_with_images(question_obj["question_text"], image_dir="images")

# --- Input Method ---
input_method = st.radio("Choose input method:", ["Text", "Voice"])
user_input = ""

if input_method == "Text":
    user_input = st.text_area("Write your answer here:", height=200)

else:
    uploaded_file = st.file_uploader("Upload .wav or .m4a file", type=["wav", "m4a"])
    audio_bytes = st_audiorec() or (uploaded_file.read() if uploaded_file else None)
    if audio_bytes:
        with st.spinner("Transcribing..."):
            try:
                user_input = transcribe_audio(audio_bytes, DEEPGRAM_API_KEY)
                st.text_area("Transcript (edit if needed)", value=user_input, height=200, key="transcript_edit")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()
    else:
        st.info("Please record or upload an audio file.")
        st.stop()

# --- Submit Button ---
if st.button("Submit Answer") and user_input.strip():
    with st.spinner("Submitting..."):
        try:
            examples = retriever.get_nearest_neighbors(
                query=user_input,
                case_id=case_id,
                question_id=question_id,
                n=3
            )

            if not examples:
                st.info("No relevant past examples found — feedback will be based solely on your response.")

            prompt = build_prompt(
                question_text=question_obj["question_text"],
                rubric=question_obj["rubric"],
                examples=examples,
                user_input=user_input,
                generation_instructions=question_obj["generation_instructions"]
            )

            feedback = generate_feedback(prompt, case["system_role"], DEEPSEEK_API_KEY)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            sheet.append_row([
                timestamp,
                st.session_state.user_name,
                st.session_state.user_email,
                case_id,
                question_id,
                user_input.strip(),
                feedback.strip()
            ])

            st.session_state.submitted_questions.append(question_id)
            st.session_state.current_question += 1
            st.success("✅ Submitted!")
            st.rerun()

        except Exception as e:
            st.error(f"Submission failed: {e}")
