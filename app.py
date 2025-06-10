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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube & Drive Media Transcriber (SRT Output)",
    layout="wide",
    initial_sidebar_state="auto"
)

st.warning(
    """
**Dependency Alert:** This app relies on `ffmpeg` installed via `packages.txt` to extract audio from YouTube and Google Drive videos.
It uses ffmpeg to convert video to WAV for Whisper transcription. Please ensure ffmpeg is available and watch logs for any errors.
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


def sanitize_filename(filename: str) -> str:
    if not filename:
        return "transcript"
    base = os.path.splitext(filename)[0]
    san = re.sub(r'[<>:"/\\|?*\s]+', '_', base)
    return san.strip('_-') or "transcript"


def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio from a YouTube URL and converts it to a WAV file.
    Returns a tuple (file_path, video_title).
    """
    try:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_wav = temp.name
        temp.close()
    except Exception as e:
        st.error(f"Failed to create temporary file: {e}", icon="❌")
        return None, None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_wav,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
    }

    st.info("Downloading and converting YouTube audio to WAV... (requires ffmpeg)")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio_transcript')
    except Exception as e:
        st.error(f"yt-dlp error: {e}", icon="❌")
        return None, None

    return output_wav, title


def download_media(url: str) -> tuple[str | None, str | None]:
    """
    Downloads media from YouTube or Google Drive, returning a WAV audio file and title.
    """
    if "drive.google.com" in url:
        # Extract Drive file ID
        match = re.search(r"/(?:d/|open\?id=)([a-zA-Z0-9_-]+)", url)
        if not match:
            st.error("Could not parse Google Drive file ID. Please ensure it's a shareable URL.", icon="❌")
            return None, None
        file_id = match.group(1)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        try:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_path = temp_video.name
            temp_video.close()
            st.info("Downloading video from Google Drive...")
            gdown.download(download_url, video_path, quiet=False)
        except Exception as e:
            st.error(f"Drive download error: {e}", icon="❌")
            return None, None

        # Extract audio from video
        audio_path = video_path.rsplit('.', 1)[0] + ".wav"
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not os.path.exists(audio_path):
            st.error("Error extracting audio from video.", icon="❌")
            return None, None

        return audio_path, os.path.basename(video_path)
    else:
        return download_audio_yt_dlp(url)


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


def transcribe_with_whisper(file_path: str, language: str) -> str:
    """Transcribes audio using OpenAI Whisper model."""
    st.info("Transcribing audio with Whisper AI...")
    try:
        with open(file_path, "rb") as audio_file:
            resp = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language=language
            )
        return resp.get('text', '')
    except Exception as e:
        st.error(f"Whisper transcription error: {e}", icon="❌")
        return ""

# --- MAIN UI ---
st.title("🎬 Media Transcriber (SRT Output)")
st.markdown(
    """
Enter a YouTube or Google Drive URL below. The app will download the media, extract audio via ffmpeg,
transcribe it using OpenAI Whisper, and generate an SRT transcript with timestamps.
You can also download the SRT transcript as a Word (.docx) file.
*(Requires `ffmpeg` installed via packages.txt)*
    """
)

media_url = st.text_input("Enter YouTube or Drive URL:", placeholder="e.g., https://youtu.be/... or https://drive.google.com/...")
selected_language_name = st.selectbox(
    "Audio Language (for transcription)",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

if media_url:
    audio_path, title = download_media(media_url)
    if audio_path:
        duration = get_audio_duration(audio_path)
        transcript_text = transcribe_with_whisper(audio_path, selected_language_code)
        if transcript_text:
            srt_data = build_single_srt(transcript_text, duration)
            st.subheader("Generated SRT:")
            st.text_area("SRT Content", value=srt_data, height=200)

            # Download as .srt
            filename = sanitize_filename(title) + ".srt"
            st.download_button(
                label="Download SRT",
                data=srt_data,
                file_name=filename,
                mime="text/plain"
            )

            # Download as Word
            doc = Document()
            doc.add_paragraph(srt_data)
            doc_stream = io.BytesIO()
            doc.save(doc_stream)
            doc_stream.seek(0)
            st.download_button(
                label="Download as Word (.docx)",
                data=doc_stream,
                file_name=sanitize_filename(title) + ".docx"
            )

st.markdown("---")
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by OpenAI Whisper, yt-dlp, gdown, and Streamlit. | App loaded: {current_time_str}")
