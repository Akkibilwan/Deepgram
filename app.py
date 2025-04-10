import streamlit as st
import os
import asyncio
import io
import tempfile
import yt_dlp # Use yt-dlp library
import re
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document
from docx.shared import Inches

# --- FFmpeg Warning ---
st.warning("""
**Dependency Alert:** This app uses `yt-dlp` which requires **`ffmpeg`** to be installed on the system where Streamlit is running.
- **Local:** Ensure `ffmpeg` is installed and accessible in your system's PATH.
- **Deployment:** Standard deployment platforms (like Streamlit Community Cloud) may **not** have `ffmpeg`. Container-based deployment (e.g., Docker) might be necessary.
""", icon="⚠️")


# --- Configuration ---

# Load Deepgram API Key
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
config: DeepgramClientOptions = DeepgramClientOptions(verbose=False)
deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)

# Supported Languages
SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi",
    "Japanese": "ja", "Russian": "ru", "Chinese (Mandarin, Simplified)": "zh-CN",
}

# --- Helper Functions ---

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio from URL using yt-dlp to a temporary file.
    Returns a tuple: (path_to_temp_audio_file, video_title) or (None, None) on failure.
    """
    temp_audio_path = None
    video_title = "audio_transcript" # Default title

    try:
        # Create a temporary file *descriptor* first, yt-dlp will write to its path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name

        # yt-dlp options
        # Reference: https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best', # Prioritize best audio-only, fallback to best overall
            'outtmpl': temp_audio_path, # Set output template to our temp file path
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3', # Extract to mp3
                'preferredquality': '192', # Audio quality
            }],
            'noplaylist': True, # Only download single video if URL is playlist
            'quiet': True, # Suppress console output from yt-dlp
            'no_warnings': True,
            # 'verbose': True, # Uncomment for debugging yt-dlp issues
        }

        st.info(f"Attempting to download audio from URL...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(url, download=True) # Download the audio
                video_title = info_dict.get('title', video_title) # Get video title if available
                st.success(f"Audio downloaded successfully for '{video_title}'.")
                return temp_audio_path, video_title
            except yt_dlp.utils.DownloadError as e:
                st.error(f"yt-dlp download failed: {e}")
                # Clean up temp file if download fails partway
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return None, None
            except Exception as e: # Catch other potential errors during extraction
                 st.error(f"An unexpected error occurred during download/extraction: {e}")
                 if temp_audio_path and os.path.exists(temp_audio_path):
                     os.remove(temp_audio_path)
                 return None, None

    except Exception as e:
        st.error(f"Failed to initialize download: {e}")
        # Clean up if temp file was created but ydl failed before download
        if temp_audio_path and os.path.exists(temp_audio_path):
             try:
                 os.remove(temp_audio_path)
             except OSError:
                 pass # Ignore error if file couldn't be removed
        return None, None


async def transcribe_audio_data(audio_data: bytes, language_code: str) -> str:
    """Transcribes audio data (bytes) using Deepgram asynchronously."""
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2", smart_format=True, language=language_code
        )

        st.info(f"Sending audio to Deepgram for transcription in {language_code}...")
        response = await deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript
        st.success("Transcription complete!")
        return transcript

    except Exception as e:
        st.error(f"Deepgram transcription failed: {e}")
        return ""

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
    base_name = os.path.splitext(filename)[0]
    sanitized = re.sub(r'[\\/*?:"<>|\s]+', '_', base_name)
    return sanitized[:100]

# --- Streamlit App UI ---

st.set_page_config(page_title="YouTube Transcriber (yt-dlp)", layout="wide")
st.title("🎙️ YouTube Video Transcriber (via yt-dlp)")
st.markdown("""
Enter a YouTube URL. The app will attempt to download the audio using `yt-dlp`
(requires `ffmpeg` installed system-wide) and then transcribe it using Deepgram.
""")

# --- Input Fields ---
youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
selected_language_name = st.selectbox(
    "Choose Transcription Language:",
    options=list(SUPPORTED_LANGUAGES.keys()), index=0
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

# --- Session State Initialization ---
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'video_title' not in st.session_state:
    st.session_state.video_title = "transcript"

# --- Transcription Button and Logic ---
transcribe_button = st.button("Transcribe Video", type="primary")

if transcribe_button and youtube_url:
    st.session_state.transcript = "" # Clear previous transcript
    audio_filepath = None # Ensure filepath is defined for finally block

    with st.spinner("Processing... This may take a while depending on video length."):
        try:
            # 1. Download Audio using yt-dlp
            audio_filepath, video_title = download_audio_yt_dlp(youtube_url)
            st.session_state.video_title = video_title or "transcript" # Update title

            if audio_filepath and os.path.exists(audio_filepath):
                # 2. Read Audio Data from temp file
                st.info("Reading downloaded audio file...")
                with open(audio_filepath, "rb") as audio_file:
                    audio_data = audio_file.read()

                if not audio_data:
                     st.error("Failed to read audio data from downloaded file.")
                else:
                    # 3. Transcribe Audio Data
                    transcript_text = asyncio.run(transcribe_audio_data(audio_data, selected_language_code))
                    st.session_state.transcript = transcript_text
            else:
                 # Error message handled within download_audio_yt_dlp
                 st.warning("Could not proceed without successfully downloaded audio.")

        except Exception as e:
            st.error(f"An unexpected error occurred in the main process: {e}")
        finally:
            # 4. Clean up temporary audio file
            if audio_filepath and os.path.exists(audio_filepath):
                try:
                    os.remove(audio_filepath)
                    st.info("Temporary audio file cleaned up.")
                except Exception as e:
                    st.warning(f"Could not remove temporary file {audio_filepath}: {e}")

elif transcribe_button and not youtube_url:
    st.warning("Please enter a YouTube URL.")

# --- Display Transcript & Download ---
if st.session_state.transcript:
    st.subheader("Transcription Result:")
    st.text_area("Transcript", st.session_state.transcript, height=300)

    st.subheader("Download Transcript:")
    try:
        word_buffer = create_word_document(st.session_state.transcript)
        base_filename = sanitize_filename(st.session_state.video_title)
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
st.caption("Powered by Deepgram, yt-dlp, and Streamlit")
