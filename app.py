# -*- coding: utf-8 -*-
import streamlit as st
import os
import io
import tempfile
import yt_dlp  # For downloading and converting audio
import re
import subprocess
import json
from datetime import datetime
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document
import openai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Transcriber (SRT Output)",
    layout="wide",
    initial_sidebar_state="auto"
)

st.warning(
    """
**Dependency Alert:** This app relies on `ffmpeg` being installed via `packages.txt`. 
It uses ffmpeg to convert the downloaded audio to WAV. Please monitor logs for any ffmpeg errors.
""",
    icon="ℹ️"
)

# --- Helper Functions ---

def format_time(seconds: float) -> str:
    """Formats a time in seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

def build_single_srt(transcript: str, duration: float) -> str:
    """Builds a single SRT block covering the entire duration."""
    start_time = "00:00:00,000"
    end_time = format_time(duration)
    return f"1\n{start_time} --> {end_time}\n{transcript.strip()}\n\n"

@st.cache_data
def load_api_key(key_name: str) -> str:
    try:
        key = st.secrets[key_name]
        if not key or key == f"YOUR_{key_name}_HERE" or len(key) < 20:
            st.error(f"Error: {key_name} missing/invalid.", icon="🚨")
            return ""
        return key
    except Exception as e:
        st.error(f"Secrets error: {e}", icon="🚨")
        return ""

# Load API keys and initialize clients
DEEPGRAM_API_KEY = load_api_key("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    st.stop()
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key missing. Please add OPENAI_API_KEY to your secrets.", icon="🚨")
    st.stop()

@st.cache_resource
def get_deepgram_client(api_key):
    try:
        config = DeepgramClientOptions(verbose=False)
        return DeepgramClient(api_key, config)
    except Exception as e:
        st.error(f"Deepgram client init error: {e}", icon="🚨")
        st.stop()

deepgram = get_deepgram_client(DEEPGRAM_API_KEY)
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key

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

def sanitize_filename(filename: str) -> str:
    if not filename:
        return "transcript"
    base = os.path.splitext(filename)[0]
    san = re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+', '_', base)
    return san.strip('_-') or "transcript"

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio from a YouTube URL and converts it to WAV.
    Returns (file_path, video_title).
    """
    video_title = "audio_transcript"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="") as temp_audio:
            temp_audio_path = temp_audio.name
    except Exception as e:
        st.error(f"Failed to create temporary file: {e}", icon="❌")
        return None, None

    output_template = temp_audio_path + ".wav"
    # Updated yt-dlp options with a custom User-Agent header.
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'noplaylist': True,
        'quiet': False,
        'no_warnings': True,
        'socket_timeout': 45,
        'retries': 2,
        'overwrites': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.188 Safari/537.36'
        },
        # Uncomment and update the following line if you need to use cookies:
        # 'cookies': 'path/to/your/cookies.txt'
    }

    st.info("Downloading and converting audio to WAV... (requires ffmpeg)")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', video_title)
    except Exception as e:
        st.error(f"yt-dlp error: {e}", icon="❌")
        if os.path.exists(output_template):
            os.remove(output_template)
        return None, None

    actual_filepath = output_template
    if not os.path.exists(actual_filepath):
        candidate = output_template + ".wav"
        if os.path.exists(candidate):
            actual_filepath = candidate

    if not os.path.exists(actual_filepath) or os.path.getsize(actual_filepath) == 0:
        st.error("Download/Conversion failed: output file missing/empty.", icon="❌")
        if os.path.exists(actual_filepath):
            os.remove(actual_filepath)
        return None, None

    st.success(f"Audio download & conversion completed: '{video_title}' "
               f"({os.path.getsize(actual_filepath)/1024/1024:.2f} MB).")
    return actual_filepath, video_title

def get_audio_duration(file_path: str) -> float:
    """
    Uses ffprobe to return the duration (in seconds) of the audio file.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except Exception as e:
        st.error(f"Error getting duration: {e}", icon="❌")
        return 0.0

# The remainder of your code remains unchanged...
# (Including functions for transcription, SRT generation, translation, and the main UI)
# For brevity, only the download_audio_yt_dlp function was updated with the custom HTTP header.

st.title("🎬 YouTube Video Transcriber (SRT Output)")
st.markdown(
    """
Enter a YouTube URL below. The app will download the audio track, transcribe it using either Deepgram or OpenAI Whisper,
and generate the transcript in SRT (SubRip Subtitle) format with timestamps.
You can also download the SRT transcript as a Word (.docx) file.
*(Requires `ffmpeg` installed in the backend via packages.txt)*
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
selected_language_name = st.selectbox(
    "Audio Language (Language detection enabled)",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0,
    help="Select the expected audio language. (If Hindi is selected, the transcript will be translated to English.)"
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]
# (Include your transcription and SRT generation functions below as before.)
# For example, functions to split audio, transcribe using Deepgram/OpenAI, generate SRT, etc.

# ... (Rest of your code remains the same as your previous version)

# In your main UI, you would use download_audio_yt_dlp() to download the video,
# then transcribe and generate the SRT subtitles, and finally display and allow download.

