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

# --- SET PAGE CONFIG FIRST ---
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="YouTube Transcriber (yt-dlp)",
    layout="wide",
    initial_sidebar_state="auto" # Or "expanded" or "collapsed"
)

# --- FFmpeg Warning ---
# Now it's safe to call other Streamlit commands
st.warning("""
**Dependency Alert:** This app uses `yt-dlp` which requires **`ffmpeg`** to be installed on the system where Streamlit is running.
- **Local:** Ensure `ffmpeg` is installed and accessible in your system's PATH. (e.g., `brew install ffmpeg`, `sudo apt install ffmpeg`, or download from ffmpeg.org)
- **Deployment:** Standard deployment platforms (like Streamlit Community Cloud) may **not** have `ffmpeg`. Container-based deployment (e.g., Docker) might be necessary.
""", icon="⚠️")


# --- Configuration ---

# Load Deepgram API Key
# Use a function to keep the main script cleaner and handle errors gracefully
@st.cache_data # Cache the key loading to avoid re-reading on every interaction
def load_api_key():
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"]
        if not api_key:
            st.error("Error: DEEPGRAM_API_KEY is empty in secrets.toml. Please add it.", icon="🚨")
            return None
        return api_key
    except KeyError:
        st.error("Error: DEEPGRAM_API_KEY not found in secrets.toml. Please create it.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred reading secrets: {e}", icon="🚨")
        return None

DEEPGRAM_API_KEY = load_api_key()

# Stop execution if API key is missing
if not DEEPGRAM_API_KEY:
    st.stop()

# Deepgram client configuration (consider caching the client too)
@st.cache_resource # Cache the Deepgram client resource
def get_deepgram_client(api_key):
    try:
        config: DeepgramClientOptions = DeepgramClientOptions(verbose=False)
        deepgram: DeepgramClient = DeepgramClient(api_key, config)
        return deepgram
    except Exception as e:
        st.error(f"Failed to initialize Deepgram client: {e}", icon="🚨")
        st.stop() # Stop if client can't be initialized

deepgram = get_deepgram_client(DEEPGRAM_API_KEY)


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
    video_title = "audio_transcript"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_path,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30, # Add a timeout
            'retries': 3, # Add retries
            # 'verbose': True, # Uncomment for debugging
        }

        st.info(f"Attempting to download audio from URL...")
        progress_bar = st.progress(0, text="Download starting...") # Add progress bar

        def progress_hook(d):
            if d['status'] == 'downloading':
                # Estimate progress (may not be perfectly accurate for audio extraction)
                percent = d.get('_percent_str', '0%')
                # Remove ANSI codes if any
                percent_clean = re.sub(r'\x1b\[[0-9;]*m', '', percent).replace('%','').strip()
                try:
                   progress_val = float(percent_clean) / 100.0
                   progress_bar.progress(progress_val, text=f"Downloading: {percent}")
                except ValueError:
                   progress_bar.progress(0.5, text=f"Downloading: Status - {d.get('speed_str', 'Processing...')}") # Fallback progress text
            elif d['status'] == 'finished':
                progress_bar.progress(1.0, text="Download complete, post-processing...")
            elif d['status'] == 'error':
                 st.error("Download hook reported an error.")


        ydl_opts['progress_hooks'] = [progress_hook]


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(url, download=True)
                video_title = info_dict.get('title', video_title)
                st.success(f"Audio downloaded successfully for '{video_title}'.")
                progress_bar.progress(1.0, text="Audio ready.")
                return temp_audio_path, video_title
            except yt_dlp.utils.DownloadError as e:
                st.error(f"yt-dlp download failed: {e}", icon="❌")
                progress_bar.empty() # Remove progress bar on error
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return None, None
            except Exception as e:
                 st.error(f"An unexpected error occurred during download/extraction: {e}", icon="❌")
                 progress_bar.empty()
                 if temp_audio_path and os.path.exists(temp_audio_path):
                     os.remove(temp_audio_path)
                 return None, None

    except Exception as e:
        st.error(f"Failed to initialize download: {e}", icon="❌")
        if temp_audio_path and os.path.exists(temp_audio_path):
             try:
                 os.remove(temp_audio_path)
             except OSError:
                 pass
        return None, None


async def transcribe_audio_data(audio_data: bytes, language_code: str) -> str:
    """Transcribes audio data (bytes) using Deepgram asynchronously."""
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2", # Consider making model selectable if needed
            smart_format=True,
            language=language_code,
            # Add other options if desired, e.g., diarize=True
        )

        st.info(f"Sending audio to Deepgram for transcription in {language_code}...")
        response = await deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Basic check on response structure
        if response and response.results and response.results.channels and \
           response.results.channels[0].alternatives:
            transcript = response.results.channels[0].alternatives[0].transcript
            st.success("Transcription complete!", icon="✅")
            return transcript
        else:
            st.error("Transcription completed, but the response structure was unexpected.", icon="⚠️")
            # Log the response for debugging if needed
            # st.write(response)
            return "[Error retrieving transcript]"


    except Exception as e:
        st.error(f"Deepgram transcription failed: {e}", icon="❌")
        return ""

def create_word_document(text: str) -> io.BytesIO:
    """Creates a Word document (.docx) in memory containing the text."""
    try:
        document = Document()
        document.add_paragraph(text)
        buffer = io.BytesIO()
        document.save(buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Failed to create Word document: {e}", icon="📄")
        return None # Return None to indicate failure

def sanitize_filename(filename: str) -> str:
    """Removes potentially problematic characters for filenames."""
    if not filename: return "transcript"
    base_name = os.path.splitext(filename)[0]
    # Replace spaces and invalid chars (Windows/Linux/Mac) with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\s\.]+', '_', base_name)
    # Remove leading/trailing underscores/hyphens
    sanitized = sanitized.strip('_-')
    # Limit length
    return sanitized[:100] if sanitized else "transcript"

# --- Streamlit App UI ---

st.title("🎙️ YouTube Video Transcriber (via yt-dlp)")
st.markdown("""
Enter a YouTube URL. The app will attempt to download the audio using `yt-dlp`
(requires `ffmpeg` installed system-wide) and then transcribe it using Deepgram.
""")

# --- Input Fields ---
youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
selected_language_name = st.selectbox(
    "Choose Transcription Language:",
    options=list(SUPPORTED_LANGUAGES.keys()), index=0
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

# --- Session State Initialization ---
# Use functions to initialize state if needed, helps organization
def init_session_state():
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'video_title' not in st.session_state:
        st.session_state.video_title = "transcript"
    if 'processing' not in st.session_state:
        st.session_state.processing = False # To disable button during run

init_session_state()


# --- Transcription Button and Logic ---
transcribe_button = st.button(
    "Transcribe Video",
    type="primary",
    disabled=st.session_state.processing # Disable button while running
)

if transcribe_button and youtube_url:
    st.session_state.processing = True # Set flag
    st.session_state.transcript = "" # Clear previous transcript
    audio_filepath = None

    # Rerun to update the button state immediately
    st.rerun()

# Separate block to run processing only if flag is set
if st.session_state.processing:
    if not youtube_url: # Check URL again in case state is inconsistent
        st.warning("Please enter a YouTube URL.")
        st.session_state.processing = False
        st.rerun()
    else:
        with st.spinner("Processing... This may take a while depending on video length."):
            try:
                # 1. Download Audio using yt-dlp
                audio_filepath, video_title = download_audio_yt_dlp(youtube_url)
                st.session_state.video_title = video_title or "transcript"

                if audio_filepath and os.path.exists(audio_filepath):
                    # 2. Read Audio Data from temp file
                    st.info("Reading downloaded audio file...")
                    with open(audio_filepath, "rb") as audio_file:
                        audio_data = audio_file.read()

                    if not audio_data:
                        st.error("Failed to read audio data from downloaded file.", icon="⚠️")
                        transcript_text = "" # Ensure transcript is empty
                    else:
                        # 3. Transcribe Audio Data
                        # Run the async function within the spinner context
                        transcript_text = asyncio.run(transcribe_audio_data(audio_data, selected_language_code))

                    st.session_state.transcript = transcript_text
                else:
                    st.warning("Could not proceed without successfully downloaded audio.", icon="⚠️")
                    st.session_state.transcript = "" # Ensure transcript is empty if download failed


            except Exception as e:
                st.error(f"An unexpected error occurred in the main process: {e}", icon="❌")
                st.session_state.transcript = "" # Ensure transcript is empty on error
            finally:
                # 4. Clean up temporary audio file
                if audio_filepath and os.path.exists(audio_filepath):
                    try:
                        os.remove(audio_filepath)
                        st.info("Temporary audio file cleaned up.")
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {audio_filepath}: {e}", icon="⚠️")

                # Reset processing flag regardless of outcome
                st.session_state.processing = False
                # Rerun to show results and re-enable button
                st.rerun()


# --- Display Transcript & Download ---
# This part runs after processing is complete (due to rerun)
if st.session_state.transcript:
    st.subheader("Transcription Result:")
    st.text_area("Transcript", st.session_state.transcript, height=300, key="transcript_display")

    st.subheader("Download Transcript:")
    word_buffer = create_word_document(st.session_state.transcript)

    if word_buffer: # Only show button if doc creation succeeded
        base_filename = sanitize_filename(st.session_state.video_title)
        file_name = f"{base_filename}_{selected_language_code}.docx"

        st.download_button(
            label="Download as Word (.docx)",
            data=word_buffer,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_button"
        )
    else:
        st.error("Could not generate the Word document for download.", icon="📄")

elif transcribe_button and not youtube_url: # Handle case where button clicked with no URL
    st.warning("Please enter a YouTube URL.")
    st.session_state.processing = False # Reset flag if set erroneously


st.markdown("---")
st.caption(f"Powered by Deepgram, yt-dlp, and Streamlit | Current Time (Server): {st.experimental_get_query_params().get('ts', ['N/A'])[0]}")
# Note: Added a placeholder for timestamp, actual update needs more work if live time is needed
