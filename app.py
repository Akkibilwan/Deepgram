# File: main/app.py

# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
import yt_dlp      # For YouTube
import gdown      # For Google Drive
import re
import subprocess
import openai
import logging

# ————— Logging —————
logging.basicConfig(level=logging.INFO)

# ————— Page config —————
st.set_page_config(
    page_title="Media Transcriber with Whisper",
    layout="wide",
)

st.write("🚀 **Media Transcriber** loaded")
st.warning(
    """
**Dependency Alert:** This app needs `ffmpeg` (via packages.txt) to extract audio from video.
Make sure `ffmpeg` is available and watch logs for any errors.
""",
    icon="ℹ️"
)

# ————— Helpers —————
def sanitize_filename(fn: str) -> str:
    base = os.path.splitext(fn or "transcript")[0]
    return re.sub(r'[<>:"/\\|?*\s]+', '_', base).strip('_-') or "transcript"

@st.cache_data
def load_api_key(name: str) -> str:
    try:
        key = st.secrets[name]
        if not key or key.startswith("YOUR_"):
            st.error(f"⚠️ {name} missing or invalid.")
            return ""
        return key
    except Exception as e:
        st.error(f"Secrets error: {e}")
        return ""

# Load OpenAI
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.stop()
openai.api_key = OPENAI_API_KEY

SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi",
    "Japanese": "ja", "Russian": "ru", "Chinese (Simplified)": "zh-CN"
}

def download_audio_yt(url: str):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = tmp.name; tmp.close()
        opts = {
            "format": "bestaudio/best",
            "outtmpl": wav_path,
            "postprocessors": [{"key":"FFmpegExtractAudio","preferredcodec":"wav"}],
            "quiet": True
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        return wav_path, info.get("title","transcript")
    except Exception as e:
        st.error(f"yt-dlp error: {e}")
        logging.exception("yt-dlp failed")
        return None, None

def download_media(url: str):
    if "drive.google.com" in url:
        m = re.search(r"/(?:d/|open\?id=)([\w-]+)", url)
        if not m:
            st.error("Invalid Drive URL")
            return None, None
        fid = m.group(1)
        dl = f"https://drive.google.com/uc?export=download&id={fid}"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        vid_path = tmp.name; tmp.close()
        try:
            gdown.download(dl, vid_path, quiet=False)
        except Exception as e:
            st.error(f"Drive download error: {e}")
            logging.exception("gdown failed")
            return None, None
        wav = vid_path.rsplit(".",1)[0]+".wav"
        cmd = ["ffmpeg","-i",vid_path,"-vn","-acodec","pcm_s16le","-ar","16000","-ac","1",wav]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode!=0 or not os.path.exists(wav):
            st.error("ffmpeg audio extraction failed")
            logging.error(proc.stderr.decode())
            return None, None
        return wav, os.path.basename(vid_path)
    else:
        return download_audio_yt(url)

def transcribe_whisper(path: str, lang: str):
    try:
        with open(path,"rb") as f:
            resp = openai.Audio.transcribe("whisper-1", file=f, language=lang)
        return resp.get("text","")
    except Exception as e:
        st.error(f"Whisper error: {e}")
        logging.exception("Whisper failed")
        return ""

# ————— UI —————
st.title("🎬 Media Transcriber")
url = st.text_input("Enter YouTube or Drive URL")
lang = st.selectbox("Pick language", list(SUPPORTED_LANGUAGES.keys()), index=0)

if st.button("Generate Transcript"):
    st.write("▶️ Button clicked!")
    st.write("URL:", url)
    logging.info("Button clicked with URL: %s", url)

    if not url:
        st.error("Please paste a valid URL.")
    else:
        audio_file, title = download_media(url)
        st.write("Downloaded audio to:", audio_file)
        logging.info("Downloaded audio: %s", audio_file)
        if audio_file:
            text = transcribe_whisper(audio_file, SUPPORTED_LANGUAGES[lang])
            if text:
                st.subheader("Transcript")
                st.text_area("", text, height=300)
                fname = sanitize_filename(title) + ".txt"
                st.download_button("Download .txt", text, file_name=fname, mime="text/plain")

st.markdown("---")
st.caption(f"Powered by Whisper, yt-dlp, gdown & ffmpeg — {datetime.now():%Y-%m-%d %H:%M:%S}")
