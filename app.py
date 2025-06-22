# -*- coding: utf-8 -*-
"""
Case interview submission app with FAISS retrieval, DeepSeek feedback, and Google Sheet logging.
"""

import streamlit as st
import gspread
import json
import re
import uuid
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
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
DECRYPTION_KEY = st.secrets["DECRYPTION_KEY"].encode()
DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
ENCRYPTED_PATH = "case_questions.json.encrypted"
FAISS_INDEX_PATH = "faiss_index.encrypted"
FAISS_META_PATH = "metadata.encrypted"
CASE_PASSWORDS = st.secrets["CASE_PASSWORDS"]

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GSHEET_CREDS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(st.secrets["AnswerStorage_Sheet_ID"]).sheet1

# --- Session State Init ---
for key, default in {
    "submitted_questions": [],
    "current_question": 0,
    "user_name": "",
    "user_email": "",
    "details_submitted": False,
    "input_method_chosen": False,
    "selected_input_method": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Load Data ---
case_data = decrypt_file(ENCRYPTED_PATH, DECRYPTION_KEY)

retriever = EncryptedAnswerRetriever(
    encrypted_index_path=FAISS_INDEX_PATH,
    encrypted_meta_path=FAISS_META_PATH,
    decryption_key=DECRYPTION_KEY
)

# --- UI Title ---
st.title("Case Interview Submission")

# --- Welcome ---
st.markdown("""
### How It Works

You will answer one case interview question at a time, using either voice or text.

Once you've completed all questions, your responses will be reviewed by an **ex-McKinsey interview coach**.

**You’ll receive personalized written feedback within 48 hours via email.**
""")

# --- User Info ---
if not st.session_state.details_submitted:
    st.subheader("Your Details")
    name = st.text_input("Your name")
    email = st.text_input("Your email address")
    email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"

    if st.button("Continue"):
        if not name or not email:
            st.warning("Please enter your name and email.")
            st.stop()
        if not re.match(email_pattern, email):
            st.warning("Please enter a valid email address.")
            st.stop()

        st.session_state.user_name = name
        st.session_state.user_email = email
        st.session_state.details_submitted = True
        st.rerun()
    st.stop()

# --- Case Selection ---
if "selected_case_id" not in st.session_state:
    st.subheader("Choose a Case")
    case_ids = sorted(case_data.keys())
    cols = st.columns(min(len(case_ids), 3))

    for i, cid in enumerate(case_ids):
        case_title = case_data[cid]["case_title"]
        with cols[i % 3]:
            if st.button(f"{case_title}", key=f"casebtn_{cid}"):
                st.session_state.selected_case_id = cid
                st.session_state.input_method_chosen = False
                st.session_state.current_question = 0
                st.rerun()
    st.stop()

# --- Case Password ---
case_id = st.session_state.selected_case_id
if f"authenticated_{case_id}" not in st.session_state:
    st.subheader("Enter Case Password")
    case_password = st.text_input("Password for this case", type="password")
    if st.button("Unlock Case"):
        if CASE_PASSWORDS.get(case_id) == case_password:
            st.session_state[f"authenticated_{case_id}"] = True
            st.rerun()
        else:
            st.warning("Incorrect password for this case.")
            st.stop()
    st.stop()

# --- Ask for Input Method at Start of Case ---
if not st.session_state.input_method_chosen:
    st.subheader("Choose How You Will Answer Questions")
    st.session_state.selected_input_method = st.radio("Input Method:", ["Text", "Voice"])
    if st.button("Start Case"):
        st.session_state.input_method_chosen = True
        st.rerun()
    st.stop()

# --- Active Case ---
case = case_data[case_id]
questions = list(case["questions"].items())

# --- All Questions Completed ---
if st.session_state.current_question >= len(questions):
    st.success("You have completed all questions. Thank you!")
    st.stop()

# --- Show Case Info ---
st.markdown(f"### Case: {case['case_title']}")
st.markdown(case["case_text"])

# --- Current Question ---
question_id, question_obj = questions[st.session_state.current_question]
st.markdown("---")
st.markdown(f"#### Question {question_id}")
render_question_with_images(question_obj["question_text"], image_dir="images")

# --- Display Previous Answer ---
prev_key = f"submitted_answer_{case_id}_{question_id}"
if prev_key in st.session_state:
    st.markdown("**Your previous answer:**")
    st.markdown(f"> {st.session_state[prev_key]}")

# --- Input Area ---
input_method = st.session_state.selected_input_method
user_input = ""

if input_method == "Text":
    input_key = f"input_text_{case_id}_{question_id}_{uuid.uuid4()}"
    user_input = st.text_area("Your new answer:", height=200, key=input_key)

elif input_method == "Voice":
    uploaded_file = st.file_uploader("Upload .wav or .m4a file", type=["wav", "m4a"])
    audio_bytes = st_audiorec() or (uploaded_file.read() if uploaded_file else None)
    if audio_bytes:
        with st.spinner("Transcribing..."):
            try:
                transcript = transcribe_audio(audio_bytes, DEEPGRAM_API_KEY)
                st.session_state[f"audio_input_{case_id}_{question_id}"] = transcript
                user_input = st.text_area("Transcript (edit if needed)", value=transcript, height=200)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()
    else:
        st.info("Please record or upload an audio file.")
        st.stop()

# --- Submit ---
if st.button("Submit Answer"):
    user_input = user_input.strip()
    if not user_input:
        st.warning("Please enter a response before submitting.")
        st.stop()

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
            submission_id = str(uuid.uuid4())

            sheet.append_row([
                submission_id,
                timestamp,
                st.session_state.user_name,
                st.session_state.user_email,
                case_id,
                question_id,
                user_input,
                prompt.strip(),
                feedback.strip()
            ])

            st.session_state[prev_key] = user_input
            st.session_state.submitted_questions.append(question_id)
            st.session_state.current_question += 1
            st.success("Submitted!")
            st.rerun()

        except Exception as e:
            st.error(f"Submission failed: {e}")
