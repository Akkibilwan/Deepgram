# -*- coding: utf-8 -*-
import streamlit as st
import os
import io
import tempfile
import yt_dlp  # yt-dlp for downloading and converting audio
import re
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
def load_api_key(key_name: str) -> str:
    try:
        api_key = st.secrets[key_name]
        if not api_key or api_key == f"YOUR_{key_name}_HERE" or len(api_key) < 20:
            st.error(f"Error: {key_name} missing/invalid.", icon="🚨")
            return ""
        return api_key
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
openai.api_key = OPENAI_API_KEY  # Set the OpenAI API key

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
    Downloads the audio from the provided YouTube URL and converts it to WAV using yt-dlp.
    Returns a tuple of the file path and the video title.
    """
    video_title = "audio_transcript"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="") as temp_audio:
            temp_audio_path = temp_audio.name
    except Exception as e:
        st.error(f"Failed to create temporary file: {e}", icon="❌")
        return None, None

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

def transcribe_audio_deepgram(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    """
    Uses Deepgram to transcribe the given audio data.
    """
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(
            model="base",
            smart_format=True,
            punctuate=True,
            numerals=True,
            detect_language=True,
        )
        st.info(f"Sending '{filename_hint}' to Deepgram...", icon="📤")
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
            st.success(f"Deepgram transcription received! (Detected Language: {detected_lang})", icon="✅")
            return transcript
        else:
            st.warning("Deepgram transcription completed but no text was detected.", icon="⚠️")
            return "[Transcription empty or failed]"
    except Exception as e:
        st.error("Deepgram transcription failed.", icon="❌")
        st.exception(e)
        return ""

def transcribe_audio_openai(audio_file_obj, language_code: str, filename_hint: str = "audio") -> str:
    """
    Uses OpenAI Whisper (via the new API) to transcribe the audio.
    The audio_file_obj is a file-like object opened in binary mode.
    """
    try:
        st.info(f"Sending '{filename_hint}' to OpenAI Whisper...", icon="📤")
        response = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_obj,
            language=language_code  # Pass the language code if desired
        )
        transcript = response.get("text", "")
        if transcript:
            st.success("OpenAI Whisper transcription received!", icon="✅")
            return transcript
        else:
            st.warning("OpenAI Whisper transcription completed but no text was detected.", icon="⚠️")
            return "[Transcription empty or failed]"
    except Exception as e:
        st.error("OpenAI Whisper transcription failed.", icon="❌")
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

# --- Main App UI ---
st.title("🎬 YouTube Video Transcriber")
st.markdown(
    """
Enter a YouTube URL below. The app will download the audio track, transcribe it using either Deepgram or OpenAI Whisper,
and display the transcript as text along with an option to download it as a Word (.docx) file.
*(Requires `ffmpeg` installed in the backend via packages.txt)*
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
selected_language_name = st.selectbox(
    "Audio Language (Note: Language detection enabled)",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0,
    help="Select the expected audio language. The transcription engines may auto-detect the language."
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

transcription_engine = st.selectbox(
    "Transcription Engine",
    options=["Deepgram", "OpenAI Whisper"],
    index=0,
    help="Choose whether to use Deepgram or OpenAI Whisper for transcription."
)

if st.button("Transcribe"):
    if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        st.warning("Please enter a valid URL starting with http:// or https://", icon="⚠️")
    else:
        st.info(f"Processing URL: {youtube_url}", icon="⏳")
        audio_filepath, video_title = download_audio_yt_dlp(youtube_url)
        if audio_filepath is None:
            st.error("Download or conversion failed. Cannot proceed with transcription.", icon="❌")
        else:
            filename_hint = sanitize_filename(video_title)
            transcript_text = ""
            try:
                if transcription_engine == "Deepgram":
                    st.info("Reading downloaded WAV data for Deepgram...", icon="🎧")
                    with open(audio_filepath, "rb") as audio_file:
                        audio_data = audio_file.read()
                    if not audio_data:
                        st.error("Failed to read downloaded audio data.", icon="⚠️")
                        transcript_text = "[File Read Error]"
                    else:
                        transcript_text = transcribe_audio_deepgram(audio_data, selected_language_code, filename_hint)
                elif transcription_engine == "OpenAI Whisper":
                    st.info("Reading downloaded WAV data for OpenAI Whisper...", icon="🎧")
                    with open(audio_filepath, "rb") as audio_file:
                        transcript_text = transcribe_audio_openai(audio_file, selected_language_code, filename_hint)
            except Exception as e:
                st.error(f"Transcription error: {e}", icon="❌")
                transcript_text = "[Transcription Error]"
            finally:
                try:
                    os.remove(audio_filepath)
                    st.info("Temporary WAV file cleaned up.", icon="🧹")
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {e}", icon="⚠️")

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
st.caption(f"Powered by Deepgram, OpenAI Whisper, yt-dlp, and Streamlit. | App loaded: {current_time_str}")
