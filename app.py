# -*- coding: utf-8 -*-
import streamlit as st
import os
import io
import tempfile
import yt_dlp  # yt-dlp for downloading/converting audio
import re
from datetime import datetime
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Transcriber (yt-dlp)",
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

@st.cache_data
def load_api_key():
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"]
        if not api_key or api_key == "YOUR_DEEPGRAM_API_KEY_HERE" or len(api_key) < 20:
            st.error("Error: DEEPGRAM_API_KEY missing/invalid.", icon="🚨")
            return None
        return api_key
    except Exception as e:
        st.error(f"Secrets error: {e}", icon="🚨")
        return None

DEEPGRAM_API_KEY = load_api_key()
if not DEEPGRAM_API_KEY:
    st.stop()

@st.cache_resource
def get_deepgram_client(api_key):
    try:
        config = DeepgramClientOptions(verbose=False)
        deepgram = DeepgramClient(api_key, config)
        return deepgram
    except Exception as e:
        st.error(f"Deepgram client init error: {e}", icon="🚨")
        st.stop()

deepgram = get_deepgram_client(DEEPGRAM_API_KEY)

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
    san = san.strip('_-')
    return san if san else "transcript"

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Download audio using yt-dlp, converting it to a WAV file.
    Returns the file path (or None on error) and the video title.
    """
    video_title = "audio_transcript"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="") as temp_audio:
            temp_audio_path = temp_audio.name
    except Exception as e:
        st.error(f"Failed to create temporary file: {e}", icon="❌")
        return None, None

    # Set the output filename to be the temporary base plus '.wav'
    output_template = temp_audio_path + ".wav"
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

    # Sometimes the FFmpeg postprocessor appends an extra ".wav"
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

def transcribe_audio_data(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    """
    Sends the audio data to Deepgram and returns the transcript as text.
    """
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(
            model="base",
            smart_format=True,
            punctuate=True,
            numerals=True,
            detect_language=True,  # Enable language detection
        )
        st.info(f"Sending '{filename_hint}' (approx {len(audio_data)/1024:.1f} KB) to Deepgram...", icon="📤")
        # Synchronous Deepgram API call:
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = ""
        detected_lang = "unknown"
        if response and response.results and response.results.channels:
            first_channel = response.results.channels[0]
            detected_lang = getattr(first_channel, 'detected_language', detected_lang)
            if first_channel and first_channel.alternatives:
                first_alternative = first_channel.alternatives[0]
                if first_alternative and hasattr(first_alternative, 'transcript'):
                    transcript = first_alternative.transcript
        if transcript:
            st.success(f"Transcription received! (Detected Language: {detected_lang})", icon="✅")
            return transcript
        else:
            st.warning("Transcription completed but no text was detected.", icon="⚠️")
            return "[Transcription empty or failed]"
    except Exception as e:
        st.error("Deepgram transcription failed.", icon="❌")
        st.exception(e)
        return ""

def create_word_document(text: str) -> io.BytesIO | None:
    if not text or text == "[Transcription empty or failed]":
        return None
    try:
        doc = Document()
        doc.add_paragraph(text)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error creating Word document: {e}", icon="❌")
        return None

# --- Main App ---
st.title("🎬 YouTube Video Transcriber")
st.markdown(
    """
Enter a YouTube URL below. The app will download the audio track, transcribe it using Deepgram,
and display the transcript as text along with an option to download it as a Word (.docx) file.
*(Requires `ffmpeg` installed in the backend via packages.txt)*
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
selected_language_name = st.selectbox(
    "Audio Language (Note: Language detection enabled)",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0,
    help="Select the primary expected language. Deepgram will attempt auto-detection."
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

# Process only if URL is entered and the button is clicked.
if st.button("Transcribe"):
    if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        st.warning("Please enter a valid URL starting with http:// or https://", icon="⚠️")
    else:
        st.info(f"Processing URL: {youtube_url}", icon="⏳")
        audio_filepath, video_title = download_audio_yt_dlp(youtube_url)
        if audio_filepath is None:
            st.error("Download or conversion failed. Cannot proceed with transcription.", icon="❌")
        else:
            try:
                st.info("Reading downloaded WAV data...", icon="🎧")
                with open(audio_filepath, "rb") as audio_file:
                    audio_data = audio_file.read()
                if not audio_data:
                    st.error("Failed to read downloaded audio data.", icon="⚠️")
                    transcript_text = "[File Read Error]"
                else:
                    filename_hint = sanitize_filename(video_title)
                    transcript_text = transcribe_audio_data(audio_data, selected_language_code, filename_hint)
            except Exception as e:
                st.error(f"Transcription error: {e}", icon="❌")
                transcript_text = "[Transcription Error]"
            finally:
                try:
                    os.remove(audio_filepath)
                    st.info("Temporary WAV file cleaned up.", icon="🧹")
                except Exception as e:
                    st.warning(f"Could not remove temp file: {e}", icon="⚠️")

            # --- Display Transcript & Download Option ---
            st.subheader(f"📄 Transcription Result for '{video_title}'")
            if transcript_text and transcript_text not in [
                "[Transcription empty or failed]",
                "[File Read Error]",
                "[Transcription Error]",
            ]:
                st.text_area("Transcript Text:", transcript_text, height=350)
                word_buffer = create_word_document(transcript_text)
                if word_buffer:
                    base_filename = sanitize_filename(video_title)
                    file_name = f"{base_filename}_transcript.docx"
                    st.download_button(
                        label="Download as Word (.docx)",
                        data=word_buffer,
                        file_name=file_name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        help="Download your transcript as a Word document."
                    )
            else:
                st.warning("No valid transcript was generated.", icon="⚠️")

st.markdown("---")
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by Deepgram, yt-dlp, and Streamlit. | App loaded: {current_time_str}")
