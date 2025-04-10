import streamlit as st
import os
import asyncio
import tempfile
import io
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from pytube import YouTube
from docx import Document
from docx.shared import Inches

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
    verbose=False, # Set to True for more detailed logs if needed
    # You can add other config options here if necessary
)
deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)

# Supported Languages for Transcription (Add more as needed based on Deepgram support)
# Format: {Display Name: Deepgram Language Code}
# See Deepgram docs for full list: https://developers.deepgram.com/docs/languages
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

async def transcribe_audio(audio_filepath: str, language_code: str) -> str:
    """Transcribes audio file using Deepgram asynchronously."""
    try:
        with open(audio_filepath, "rb") as audio_file:
            buffer_data = audio_file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2", # Or choose another model like "base"
            smart_format=True,
            language=language_code,
            # Add other options like diarize=True if needed
        )

        st.info(f"Sending audio to Deepgram for transcription in {language_code}...")

        # Make the async API call
        response = await deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Extract transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        st.success("Transcription complete!")
        return transcript

    except Exception as e:
        st.error(f"Deepgram transcription failed: {e}")
        return "" # Return empty string on failure

def download_youtube_audio(url: str) -> str | None:
    """Downloads the best audio-only stream from a YouTube URL to a temporary file."""
    try:
        yt = YouTube(url)
        # Filter for audio-only streams and get the best quality (highest abr)
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

        if not audio_stream:
            st.error("Could not find an audio-only stream for this video.")
            return None

        st.info(f"Downloading audio for '{yt.title}'...")
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_audio: # Use .mp4 or .webm common formats
            audio_stream.download(output_path=os.path.dirname(temp_audio.name), filename=os.path.basename(temp_audio.name))
            st.success("Audio downloaded successfully.")
            return temp_audio.name # Return the path to the temporary file

    except Exception as e:
        st.error(f"Error downloading YouTube audio: {e}")
        return None

def create_word_document(text: str) -> io.BytesIO:
    """Creates a Word document (.docx) in memory containing the text."""
    document = Document()
    document.add_paragraph(text)
    # You can add more formatting here if needed
    # e.g., document.add_heading('Transcript', level=1)

    # Save document to a byte stream
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0) # Rewind buffer to the beginning
    return buffer

# --- Streamlit App UI ---

st.set_page_config(page_title="YouTube Transcriber", layout="wide")
st.title("🎙️ YouTube Video Transcriber using Deepgram")
st.markdown("Enter a YouTube URL, choose the language, and get the transcript.")

# --- Input Fields ---
youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

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
if 'video_title' not in st.session_state:
    st.session_state.video_title = "transcript" # Default filename base

transcribe_button = st.button("Transcribe Video", type="primary")

if transcribe_button and youtube_url:
    st.session_state.transcript = "" # Clear previous transcript
    with st.spinner("Processing... Please wait."):
        # 1. Download Audio
        audio_filepath = download_youtube_audio(youtube_url)

        if audio_filepath:
            # 2. Transcribe Audio (run async function)
            try:
                # Get video title for potential filename use
                try:
                    yt = YouTube(youtube_url)
                    st.session_state.video_title = yt.title.replace(" ", "_").replace("/","-") # Basic sanitization
                except Exception:
                    st.session_state.video_title = "transcript" # Fallback title

                # Run the asynchronous transcription function
                # asyncio.run() is suitable for simple cases in scripts/Streamlit
                transcript_text = asyncio.run(transcribe_audio(audio_filepath, selected_language_code))
                st.session_state.transcript = transcript_text

            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
            finally:
                # 3. Clean up temporary audio file
                if os.path.exists(audio_filepath):
                    try:
                        os.remove(audio_filepath)
                        st.info("Temporary audio file cleaned up.")
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {audio_filepath}: {e}")
        else:
            st.warning("Could not proceed without downloaded audio.")

# --- Display Transcript ---
if st.session_state.transcript:
    st.subheader("Transcription Result:")
    st.text_area("Transcript", st.session_state.transcript, height=300)

    # --- Download Button ---
    st.subheader("Download Transcript:")
    try:
        word_buffer = create_word_document(st.session_state.transcript)
        file_name = f"{st.session_state.video_title}_{selected_language_code}.docx"

        st.download_button(
            label="Download as Word (.docx)",
            data=word_buffer,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        st.error(f"Error creating download file: {e}")

elif transcribe_button and not youtube_url:
    st.warning("Please enter a YouTube URL.")

st.markdown("---")
st.caption("Powered by Deepgram and Streamlit")
