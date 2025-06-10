# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
import yt_dlp  # For downloading and converting audio from YouTube
import gdown    # For downloading from Google Drive
import re
import subprocess
import json
from datetime import datetime
import openai
from docx import Document
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Media Transcriber with Whisper",
    layout="wide",
    initial_sidebar_state="auto"
)

st.warning(
    """
**Dependency Alert:** This app relies on `ffmpeg` installed via `packages.txt` to extract audio from videos.
Ensure `ffmpeg` is available in the environment and monitor logs for any conversion errors.
""",
    icon="ℹ️"
)

# --- Helper Functions ---

def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be a valid filename."""
    if not filename:
        return "transcript"
    base = os.path.splitext(filename)[0]
    san = re.sub(r'[<>:"/\\|?*\s]+', '_', base)
    return san.strip('_-') or "transcript"

@st.cache_data
def load_api_key(key_name: str) -> str:
    try:
        key = st.secrets[key_name]
        if not key or key.startswith("YOUR_"):
            st.error(f"Error: {key_name} missing/invalid.", icon="🚨")
            return ""
        return key
    except Exception as e:
        st.error(f"Secrets error: {e}", icon="🚨")
        return ""

# Load OpenAI API key
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.stop()
openai.api_key = OPENAI_API_KEY

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Hindi": "hi",
    "Japanese": "ja",
    "Russian": "ru",
    "Chinese (Mandarin, Simplified)": "zh-CN"
}


def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio from a YouTube URL and converts it to WAV.
    Returns (audio_path, title) or (None, None).
    """
    try:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_wav = temp.name
        temp.close()
    except Exception as e:
        st.error(f"Temp file error: {e}", icon="❌")
        return None, None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_wav,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'quiet': True,
        'no_warnings': True
    }
    st.info("Downloading YouTube audio...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return output_wav, info.get('title', 'transcript')
    except Exception as e:
        st.error(f"yt-dlp error: {e}", icon="❌")
        return None, None


def download_media(url: str) -> tuple[str | None, str | None]:
    """
    Downloads media from YouTube or Google Drive and returns a WAV audio file and title.
    """
    if "drive.google.com" in url:
        match = re.search(r"/(?:d/|open\?id=)([a-zA-Z0-9_-]+)", url)
        if not match:
            st.error("Invalid Google Drive URL.", icon="❌")
            return None, None
        file_id = match.group(1)
        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_path = tmp.name
            tmp.close()
            st.info("Downloading video from Drive...")
            gdown.download(dl_url, video_path, quiet=False)
        except Exception as e:
            st.error(f"Drive download error: {e}", icon="❌")
            return None, None

        # Extract audio via ffmpeg
        audio_path = video_path.rsplit('.', 1)[0] + ".wav"
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.exists(audio_path):
            st.error("Audio extraction failed.", icon="❌")
            return None, None
        return audio_path, os.path.basename(video_path)
    else:
        return download_audio_yt_dlp(url)


def transcribe_with_whisper(file_path: str, language: str) -> str:
    """Transcribes audio via OpenAI Whisper."""
    st.info("Transcribing with Whisper...")
    try:
        with open(file_path, "rb") as f:
            resp = openai.Audio.transcribe(model="whisper-1", file=f, language=language)
        return resp.get('text', '')
    except Exception as e:
        st.error(f"Whisper error: {e}", icon="❌")
        return ""

# --- MAIN UI ---
st.title("🎬 Media Transcriber")
media_url = st.text_input("Enter YouTube or Drive URL:", placeholder="https://youtu.be/... or drive.google.com/...")
language = st.selectbox("Choose language:", options=list(SUPPORTED_LANGUAGES.keys()), index=0)

if st.button("Generate Transcript"):
    if not media_url:
        st.error("Please enter a valid URL.", icon="⚠️")
    else:
        audio_path, title = download_media(media_url)
        if audio_path:
            transcript = transcribe_with_whisper(audio_path, SUPPORTED_LANGUAGES[language])
            if transcript:
                st.subheader("Transcript Text")
                st.text_area("", transcript, height=300)
                # Optional: download as .txt
                fname = sanitize_filename(title) + ".txt"
                st.download_button("Download Transcript", transcript, file_name=fname, mime="text/plain")

st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by OpenAI Whisper, yt-dlp, gdown, and ffmpeg | Loaded at {current_time}")
