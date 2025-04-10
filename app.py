import streamlit as st
import os
import asyncio
import io
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document
from docx.shared import Inches
import re # For basic filename sanitization

# --- Configuration ---

# Load Deepgram API Key from secrets
try:
    DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
    if not DEEPGRAM_API_KEY:
        st.error("Error: DEEPGRAM_API_KEY not found in secrets.toml. Please add it.")
        st.stop()
except KeyError:
    st.error("Error: DEEPGRAM_API_KEY not found in secrets.toml. Please add it.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred reading secrets: {e}")
    st.stop()

# Deepgram client configuration
config: DeepgramClientOptions = DeepgramClientOptions(
    verbose=False,
)
deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)

# Supported Languages for Transcription
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
    "Chinese (Mandarin, Simplified)": "zh-CN",
}

# --- Helper Functions ---

async def transcribe_uploaded_audio(audio_data: bytes, language_code: str, mimetype: str | None = None) -> str:
    """Transcribes audio data (bytes) using Deepgram asynchronously."""
    try:
        payload: FileSource = {
            "buffer": audio_data,
        }
        # Optionally include mimetype if available and reliable
        # if mimetype:
        #     payload["mimetype"] = mimetype

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language=language_code,
        )

        st.info(f"Sending audio to Deepgram for transcription in {language_code}...")

        # Make the async API call
        # Note: Using transcribe_file method even for buffer source in v3 SDK
        response = await deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Extract transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        st.success("Transcription complete!")
        return transcript

    except Exception as e:
        st.error(f"Deepgram transcription failed: {e}")
        # Consider logging the full error for debugging
        # print(f"Deepgram Error: {e}")
        return "" # Return empty string on failure


def create_word_document(text: str) -> io.BytesIO:
    """Creates a Word document (.docx) in memory containing the text."""
    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer

def sanitize_filename(filename: str) -> str:
    """Removes potentially problematic characters for filenames."""
    # Remove file extension first
    base_name = os.path.splitext(filename)[0]
    # Replace spaces and invalid chars with underscore
    sanitized = re.sub(r'[\\/*?:"<>|\s]+', '_', base_name)
    # Limit length if necessary
    return sanitized[:100] # Limit length to avoid issues

# --- Streamlit App UI ---

st.set_page_config(page_title="Audio File Transcriber", layout="wide")
st.title("🎙️ Audio File Transcriber using Deepgram")
st.markdown("""
**Instructions:**
1.  Download the audio from your source (e.g., YouTube) using a tool like `yt-dlp`*, an online downloader, or `ffmpeg`.
2.  Save the audio as an MP3, MP4 (audio track), M4A, WAV, or other format Deepgram supports.
3.  Upload the downloaded audio file below.
4.  Choose the language spoken in the audio.
5.  Click 'Transcribe Audio File'.

*Example using `yt-dlp` (command-line tool): `yt-dlp -x --audio-format mp3 YOUR_YOUTUBE_URL`*
""")

# --- Input Fields ---
uploaded_file = st.file_uploader(
    "Upload Audio File:",
    type=['mp3', 'wav', 'm4a', 'mp4', 'aac', 'ogg', 'flac', 'amr'], # Common audio types Deepgram might support
    accept_multiple_files=False
)

# Language selection
selected_language_name = st.selectbox(
    "Choose Transcription Language:",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0 # Default to English
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

# --- Transcription Button and Logic ---
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = "transcript" # Default filename base

transcribe_button = st.button("Transcribe Audio File", type="primary")

if transcribe_button and uploaded_file is not None:
    st.session_state.transcript = "" # Clear previous transcript
    st.session_state.original_filename = uploaded_file.name # Store filename for download

    with st.spinner("Processing... Please wait."):
        # 1. Read uploaded audio data
        audio_data = uploaded_file.getvalue()
        file_mimetype = uploaded_file.type # Get mimetype if provided by browser

        # 2. Transcribe Audio (run async function)
        try:
            transcript_text = asyncio.run(transcribe_uploaded_audio(audio_data, selected_language_code, file_mimetype))
            st.session_state.transcript = transcript_text
        except Exception as e:
            st.error(f"An error occurred during transcription processing: {e}")

elif transcribe_button and uploaded_file is None:
    st.warning("Please upload an audio file first.")

# --- Display Transcript ---
if st.session_state.transcript:
    st.subheader("Transcription Result:")
    st.text_area("Transcript", st.session_state.transcript, height=300)

    # --- Download Button ---
    st.subheader("Download Transcript:")
    try:
        word_buffer = create_word_document(st.session_state.transcript)
        base_filename = sanitize_filename(st.session_state.original_filename)
        file_name = f"{base_filename}_{selected_language_code}.docx"

        st.download_button(
            label="Download as Word (.docx)",
            data=word_buffer,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        st.error(f"Error creating download file: {e}")

st.markdown("---")
st.caption("Powered by Deepgram and Streamlit")
