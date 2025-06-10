# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
import yt_dlp      # For downloading YouTube audio
import re
import subprocess
import logging
import requests      # For downloading from Google Drive
import openai
import gdown         # For downloading from Google Drive

# ————— Logging —————
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ————— Page configuration —————
st.set_page_config(
    page_title="Media Transcriber with Whisper",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 Media Transcriber")
st.write("Transcribe from a YouTube/Google Drive URL or by uploading your own audio/video file.")

st.warning(
    """
**Dependency Alert:** This app requires `ffmpeg` to be installed on the system where it's running. 
It is used to extract audio from all media types. Please ensure `ffmpeg` is available.
""",
    icon="ℹ️"
)

# ————— Helper functions —————

def sanitize_filename(name: str) -> str:
    """Removes illegal characters from a string to make it a valid filename."""
    base = os.path.splitext(name or "transcript")[0]
    safe = re.sub(r'[<>:"/\\|?*\s]+', '_', base)
    return safe.strip('_-') or "transcript"

@st.cache_data
def load_api_key(key_name: str) -> str:
    """Loads an API key from Streamlit secrets and handles errors."""
    try:
        key = st.secrets[key_name]
        if not key or key.startswith("YOUR_"):
            st.error(f"⚠️ API key '{key_name}' is missing or invalid in your Streamlit secrets.")
            return ""
        return key
    except Exception:
        st.error(f"Error loading secret '{key_name}'. Make sure it is configured correctly.")
        return ""

def download_audio_yt(url: str) -> tuple[str | None, str | None]:
    """Downloads audio from a YouTube URL and converts it to WAV format."""
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = tmp.name
        tmp.close()

        opts = {
            'format': 'bestaudio/best',
            'outtmpl': wav_path,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'youtube_transcript')
        
        return wav_path, title
    except Exception as e:
        st.error(f"YouTube download error: {e}")
        logging.exception("YT download failed")
        return None, None

def download_drive_video(file_id: str, dest_path: str):
    """Downloads a file from Google Drive using the gdown library."""
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, dest_path, quiet=False)
    except Exception as e:
        st.error(f"Google Drive download failed. This can happen with private files. Error: {e}")
        logging.exception("gdown download failed")
        raise

def download_media(url: str) -> tuple[str | None, str | None]:
    """Detects the URL type (YouTube vs. Google Drive) and downloads the media."""
    if "drive.google.com" in url:
        m = re.search(r"/(?:d/|open\?id=)([\w-]+)", url)
        if not m:
            st.error("Could not extract a valid File ID from the Google Drive URL.")
            return None, None
            
        file_id = m.group(1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid_file:
            tmp_vid_path = tmp_vid_file.name
        
        try:
            download_drive_video(file_id, tmp_vid_path)
        except Exception:
            return None, None

        wav_path = process_media_file(tmp_vid_path)
        if not wav_path:
             os.remove(tmp_vid_path) # Clean up original download
             return None, None
        return wav_path, os.path.basename(tmp_vid_path)
    else:
        return download_audio_yt(url)

# --- NEW: Function to process any media file (downloaded or uploaded) with ffmpeg ---
def process_media_file(input_path: str) -> str | None:
    """
    Converts any media file (audio or video) into a 16kHz mono WAV file for Whisper.
    Returns the path to the converted WAV file.
    """
    st.info("Extracting audio with ffmpeg...")
    wav_path = os.path.splitext(input_path)[0] + ".wav"
    
    # This ffmpeg command works for both audio and video inputs.
    cmd = ["ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path, "-y"]
    
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        st.error(f"Audio extraction via ffmpeg failed.")
        logging.error(proc.stderr.decode())
        return None
        
    return wav_path

def transcribe_whisper(audio_path: str, lang_code: str | None) -> str | None:
    """Transcribes audio using the OpenAI Whisper API."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language=lang_code if lang_code != "auto" else None
            )
        return transcript['text']
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        logging.exception("Whisper transcription failed")
        return None

# ————— Main App Logic —————

# Load OpenAI API key
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.stop()
openai.api_key = OPENAI_API_KEY

SUPPORTED_LANGUAGES = {
    "Auto-Detect": "auto", "English": "en", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Portuguese": "pt", "Dutch": "nl",
    "Hindi": "hi", "Japanese": "ja", "Russian": "ru", "Chinese": "zh",
}

# --- UPDATED: UI to select between URL and File Upload ---
st.subheader("Step 1: Choose Your Input Method")
input_method = st.radio(
    "Select one:",
    ("Enter a URL", "Upload a File"),
    label_visibility="collapsed"
)

# --- Common UI elements for language and button ---
st.subheader("Step 2: Select Language and Transcribe")
selected_lang_name = st.selectbox(
    "Language of the media:",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0
)
lang_code = SUPPORTED_LANGUAGES[selected_lang_name]

# --- Main processing logic ---
if st.button("Start Transcription", type="primary"):
    audio_path, title = None, None
    temp_media_path = None # To keep track of temporary uploaded files

    if input_method == "Enter a URL":
        url = st.text_input("Enter a YouTube or Google Drive URL:", key="url_input")
        if not url:
            st.warning("Please enter a URL to start.")
            st.stop()
        
        with st.spinner("Step 1/2: Downloading and preparing audio... This might take a while."):
            audio_path, title = download_media(url)

    else: # "Upload a File"
        uploaded_file = st.file_uploader(
            "Upload an audio or video file",
            type=['mp3', 'mp4', 'm4a', 'wav', 'mov', 'avi', 'mkv'],
            label_visibility="collapsed"
        )
        if uploaded_file is None:
            st.warning("Please upload a file to start.")
            st.stop()
        
        with st.spinner("Step 1/2: Processing uploaded file..."):
            # Save uploaded file to a temporary path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_media_path = tmp_file.name # Keep track for cleanup
            
            title = uploaded_file.name
            audio_path = process_media_file(temp_media_path)
    
    # --- Common Transcription Logic ---
    if audio_path and os.path.exists(audio_path):
        st.success("✅ Audio ready for transcription.")
        
        with st.spinner("Step 2/2: Transcribing audio with Whisper..."):
            transcript_text = transcribe_whisper(audio_path, lang_code)
            
        if transcript_text:
            st.success("✅ Transcription complete!")
            
            st.subheader(f"Transcript from: {sanitize_filename(title)}")
            st.text_area("Full Transcript", transcript_text, height=300)

            st.download_button(
                label="Download Transcript (.txt)",
                data=transcript_text,
                file_name=f"{sanitize_filename(title)}.txt",
                mime="text/plain"
            )
        
        # Clean up all temporary files
        os.remove(audio_path)
        if temp_media_path and os.path.exists(temp_media_path):
            os.remove(temp_media_path)
            
    else:
        st.error("Could not prepare the audio for transcription. Please check the input and try again.")
