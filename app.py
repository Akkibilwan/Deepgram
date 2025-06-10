# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
import yt_dlp      # For YouTube downloads
import re
import subprocess
import logging
import requests     # For Google Drive downloads
from datetime import datetime
import openai

# ————— Logging —————
logging.basicConfig(level=logging.INFO)

# ————— Page configuration —————
st.set_page_config(
    page_title="Media Transcriber with Whisper",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 Media Transcriber")
st.write("Use a YouTube or Google Drive URL to extract audio and get a Whisper transcript.")

st.warning(
    """
**Dependency Alert:** This app requires `ffmpeg` (via packages.txt) to extract audio from video files.  
Ensure `ffmpeg` is present and monitor the logs for any errors.
""",
    icon="ℹ️"
)

# ————— Helper functions —————

def sanitize_filename(name: str) -> str:
    """Convert a title into a safe filename."""
    base = os.path.splitext(name or "transcript")[0]
    safe = re.sub(r'[<>:"/\\|?*\s]+', '_', base)
    return safe.strip('_-') or "transcript"

@st.cache_data
def load_api_key(key_name: str) -> str:
    try:
        key = st.secrets[key_name]
        if not key or key.startswith("YOUR_"):
            st.error(f"⚠️ {key_name} missing or invalid.")
            return ""
        return key
    except Exception as e:
        st.error(f"Error loading secret `{key_name}`: {e}")
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
    "Chinese (Simplified)": "zh-CN"
}


def download_audio_yt(url: str) -> tuple[str|None, str|None]:
    """Download and convert YouTube audio to WAV."""
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = tmp.name; tmp.close()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': wav_path,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'quiet': True
        }
        st.info("Downloading YouTube audio...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        return wav_path, info.get('title', 'transcript')
    except Exception as e:
        st.error(f"YouTube download error: {e}")
        logging.exception("YT download failed")
        return None, None


def download_drive_video(file_id: str, dest_path: str):
    """Download Google Drive file by ID, handling confirmation tokens."""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    params = {'id': file_id}
    response = session.get(URL, params=params, stream=True)
    # Look for confirm token (for large files)
    for key, val in response.cookies.items():
        if key.startswith('download_warning'):
            params['confirm'] = val
            response = session.get(URL, params=params, stream=True)
            break
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_media(url: str) -> tuple[str|None, str|None]:
    """Detect URL type (YouTube vs Drive) and download accordingly."""
    if "drive.google.com" in url:
        m = re.search(r"/(?:d/|open\?id=)([\w-]+)", url)
        if not m:
            st.error("Invalid Google Drive URL")
            return None, None
        file_id = m.group(1)
        tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        st.info("Downloading video from Google Drive...")
        try:
            download_drive_video(file_id, tmp_vid)
        except Exception as e:
            st.error(f"Drive download error: {e}")
            logging.exception("Drive download failed")
            return None, None
        # Extract audio
        wav_path = tmp_vid.rsplit('.',1)[0] + ".wav"
        cmd = ["ffmpeg", "-i", tmp_vid, "-vn",
               "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(wav_path):
            st.error("ffmpeg audio extraction failed.")
            logging.error(proc.stderr.decode())
            return None, None
        return wav_path, os.path.basename(tmp_vid)
    else:
        return download_audio_yt(url)


def transcribe_whisper(audio_path: str, lang_code: str) -> str:
    """Transcribe WAV via OpenAI Whisper."""
    st.info("Transcribing with Whisper...")
    try:
        with open(audio_path, 'rb') as f:
            resp = openai.Audio.transcribe('whisper-1', file=f, language=lang_code)
        return resp.get('text','')
    except Exception as e:
        st.error(f"Whisper transcription error: {e}")
        logging.exception("Whisper failed")
        return ''

# ————— Main UI —————
url_input = st.text_input("Enter YouTube or Drive URL:")
lang = st.selectbox("Select language:", list(SUPPORTED_LANGUAGES.keys()), index=0)

if st.button("Generate Transcript"):
    st.write("🔍 Processing URL:", url_input)
    if not url_input:
        st.error("Please provide a URL.")
    else:
        audio_file, title = download_media(url_input)
        st.write("✅ Audio file:", audio_file)
        if audio_file:
            transcript = transcribe_whisper(audio_file, SUPPORTED_LANGUAGES[lang])
            if transcript:
                st.subheader("Transcript")
                st.text_area("", transcript, height=300)
                fname = sanitize_filename(title) + ".txt"
                st.download_button("Download Transcript", transcript, file_name=fname, mime="text/plain")

st.markdown("---")
st.caption(f"Powered by Whisper, yt-dlp, requests & ffmpeg — {datetime.now():%Y-%m-%d %H:%M:%S}")
