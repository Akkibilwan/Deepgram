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

# ————— Logging —————
# Configures basic logging to show informational messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ————— Page configuration —————
st.set_page_config(
    page_title="Media Transcriber with Whisper",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 Media Transcriber")
st.write("Enter a YouTube or Google Drive URL to extract its audio and generate a transcript using OpenAI Whisper.")

st.warning(
    """
**Dependency Alert:** This app requires `ffmpeg` to be installed on the system where it's running. 
It is used to extract audio from video files. Please ensure `ffmpeg` is available.
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
        tmp.close() # Close the file so yt-dlp can write to it

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
    """Downloads a file from Google Drive, handling large file confirmations."""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    params = {'id': file_id}
    resp = session.get(URL, params=params, stream=True)

    # Handle large file confirmation
    for key, value in resp.cookies.items():
        if key.startswith('download_warning'):
            params['confirm'] = value
            resp = session.get(URL, params=params, stream=True)
            break
            
    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(32768):
            if chunk:
                f.write(chunk)


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
        except Exception as e:
            st.error(f"Google Drive download error: {e}")
            logging.exception("Drive download failed")
            return None, None

        # Extract audio using ffmpeg
        wav_path = tmp_vid_path.rsplit('.', 1)[0] + ".wav"
        cmd = ["ffmpeg", "-i", tmp_vid_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path, "-y"]
        
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(wav_path):
            st.error("Audio extraction via ffmpeg failed.")
            logging.error(proc.stderr.decode())
            return None, None
            
        return wav_path, os.path.basename(tmp_vid_path)
    else:
        # Assumes YouTube or other yt-dlp compatible URL
        return download_audio_yt(url)


def transcribe_whisper(audio_path: str, lang_code: str | None) -> str | None:
    """Transcribes audio using the OpenAI Whisper API."""
    try:
        with open(audio_path, "rb") as audio_file:
            # Note: Using the older syntax compatible with openai < 1.0
            # If using openai > 1.0, the syntax would be:
            # client = openai.OpenAI(api_key=OPENAI_API_KEY)
            # transcript = client.audio.transcriptions.create(...)
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language=lang_code if lang_code != "auto" else None # Whisper API expects None for auto-detection
            )
        return transcript['text']
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        logging.exception("Whisper transcription failed")
        return None

# ————— Main App Logic —————

# Load OpenAI API key and stop if it's not available
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.stop()
openai.api_key = OPENAI_API_KEY

# Supported languages for Whisper, including an auto-detect option
SUPPORTED_LANGUAGES = {
    "Auto-Detect": "auto",
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
    "Chinese": "zh",
}

# --- UI elements for user input ---
url = st.text_input("Enter a YouTube or Google Drive URL:", key="url_input")
selected_lang_name = st.selectbox(
    "Choose the language of the media:",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0  # Default to "Auto-Detect"
)
lang_code = SUPPORTED_LANGUAGES[selected_lang_name]


# --- Button to trigger the analysis ---
if st.button("Start Transcription", type="primary"):
    if not url:
        st.warning("Please enter a URL to start.")
    else:
        audio_path, title = None, None
        with st.spinner("Step 1/2: Downloading and preparing audio... This might take a while."):
            audio_path, title = download_media(url)
            
        if audio_path and os.path.exists(audio_path):
            st.success("✅ Audio downloaded successfully.")
            
            transcript_text = None
            with st.spinner("Step 2/2: Transcribing audio with Whisper..."):
                transcript_text = transcribe_whisper(audio_path, lang_code)
                
            if transcript_text:
                st.success("✅ Transcription complete!")
                
                # Display the final transcript
                st.subheader(f"Transcript from: {sanitize_filename(title)}")
                st.text_area("Full Transcript", transcript_text, height=300)

                # Provide a download button for the transcript
                st.download_button(
                    label="Download Transcript (.txt)",
                    data=transcript_text,
                    file_name=f"{sanitize_filename(title)}.txt",
                    mime="text/plain"
                )
            # Clean up the temporary audio file
            os.remove(audio_path)
        else:
            st.error("Could not retrieve the audio file. Please check the URL and try again.")
