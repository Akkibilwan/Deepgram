import streamlit as st
import os
import asyncio
import io
import tempfile
import yt_dlp # Use yt-dlp library
import re
from datetime import datetime # Import datetime for timestamp
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document
from docx.shared import Inches

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(
    page_title="YouTube Transcriber (yt-dlp)",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- FFmpeg Warning ---
# This warning is crucial due to the "ffmpeg not found" error you encountered.
st.warning("""
**ACTION REQUIRED: `ffmpeg` Dependency**
The error `ffmpeg not found` means you **must** install `ffmpeg` on this system for the audio download to work.
- **Local:** Ensure `ffmpeg` is installed and accessible in your system's PATH. (e.g., `brew install ffmpeg`, `sudo apt install ffmpeg`, or download from ffmpeg.org & add to PATH)
- **Deployment:** If deploying, ensure the deployment environment includes `ffmpeg`. Standard platforms often don't; container-based deployment (e.g., Docker) is usually required.
""", icon="⚠️")


# --- Configuration ---

@st.cache_data
def load_api_key():
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"]
        if not api_key or api_key == "YOUR_DEEPGRAM_API_KEY_HERE":
            st.error("Error: DEEPGRAM_API_KEY is missing or not set in secrets.toml. Please add your key.", icon="🚨")
            return None
        return api_key
    except KeyError:
        st.error("Error: DEEPGRAM_API_KEY not found in secrets.toml. Please create it.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred reading secrets: {e}", icon="🚨")
        return None

DEEPGRAM_API_KEY = load_api_key()

if not DEEPGRAM_API_KEY:
    st.stop()

@st.cache_resource
def get_deepgram_client(api_key):
    try:
        config: DeepgramClientOptions = DeepgramClientOptions(verbose=False)
        deepgram: DeepgramClient = DeepgramClient(api_key, config)
        return deepgram
    except Exception as e:
        st.error(f"Failed to initialize Deepgram client: {e}", icon="🚨")
        st.stop()

deepgram = get_deepgram_client(DEEPGRAM_API_KEY)

SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi",
    "Japanese": "ja", "Russian": "ru", "Chinese (Mandarin, Simplified)": "zh-CN",
}

# --- Helper Functions ---

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """Downloads audio using yt-dlp. Requires ffmpeg."""
    temp_audio_path = None
    video_title = "audio_transcript"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_path,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'noplaylist': True, 'quiet': True, 'no_warnings': True,
            'socket_timeout': 30, 'retries': 3,
            # 'ffmpeg_location': '/path/to/your/ffmpeg' # Optional: uncomment and set if ffmpeg isn't in PATH
        }

        st.info(f"Attempting to download audio (requires ffmpeg)...")
        progress_bar = st.progress(0, text="Download starting...")

        def progress_hook(d):
            if d['status'] == 'downloading':
                percent_str = d.get('_percent_str', '0%')
                percent_clean = re.sub(r'\x1b\[[0-9;]*m', '', percent_str).replace('%','').strip()
                try:
                   progress_val = float(percent_clean) / 100.0
                   # Clamp value between 0 and 1
                   progress_val = max(0.0, min(1.0, progress_val))
                   progress_bar.progress(progress_val, text=f"Downloading: {percent_str.strip()}")
                except ValueError:
                   # Estimate progress if percentage is not reliable
                   total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
                   downloaded_bytes = d.get('downloaded_bytes')
                   if total_bytes and downloaded_bytes:
                        progress_val = max(0.0, min(1.0, downloaded_bytes / total_bytes))
                        progress_bar.progress(progress_val, text=f"Downloading: {d.get('_speed_str', 'Processing...')}")
                   else:
                        progress_bar.progress(0.5, text=f"Downloading: Status - {d.get('speed_str', 'Processing...')}") # Fallback
            elif d['status'] == 'finished':
                # Keep progress at 1 until post-processing potentially updates
                progress_bar.progress(1.0, text="Download complete, post-processing...")
            elif d['status'] == 'error':
                 st.error("Download hook reported an error.", icon="❌")


        ydl_opts['progress_hooks'] = [progress_hook]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(url, download=True)
                video_title = info_dict.get('title', video_title)
                st.success(f"Audio downloaded successfully for '{video_title}'.")
                progress_bar.progress(1.0, text="Audio ready.")
                return temp_audio_path, video_title
            except yt_dlp.utils.DownloadError as e:
                # Check specifically for ffmpeg error if possible (difficult robustly)
                if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
                     st.error(f"yt-dlp failed: {e}. This often means ffmpeg is missing or not found. Please install ffmpeg and ensure it's in your system PATH.", icon="❌")
                else:
                     st.error(f"yt-dlp download failed: {e}", icon="❌")
                progress_bar.empty()
                if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                return None, None
            except Exception as e:
                 st.error(f"An unexpected error occurred during download/extraction: {e}", icon="❌")
                 progress_bar.empty()
                 if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                 return None, None

    except Exception as e:
        st.error(f"Failed to initialize download: {e}", icon="❌")
        if temp_audio_path and os.path.exists(temp_audio_path):
             try: os.remove(temp_audio_path)
             except OSError: pass
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
        if response and response.results and response.results.channels and \
           response.results.channels[0].alternatives:
            transcript = response.results.channels[0].alternatives[0].transcript
            st.success("Transcription complete!", icon="✅")
            return transcript
        else:
            st.error("Transcription completed, but the response structure was unexpected.", icon="⚠️")
            return "[Error retrieving transcript]"
    except Exception as e:
        st.error(f"Deepgram transcription failed: {e}", icon="❌")
        return ""

def create_word_document(text: str) -> io.BytesIO | None:
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
        return None

def sanitize_filename(filename: str) -> str:
    """Removes potentially problematic characters for filenames."""
    if not filename: return "transcript"
    base_name = os.path.splitext(filename)[0]
    sanitized = re.sub(r'[<>:"/\\|?*\s\.]+', '_', base_name)
    sanitized = sanitized.strip('_-')
    return sanitized[:100] if sanitized else "transcript"

# --- Streamlit App UI ---

st.title("🎙️ YouTube Video Transcriber (via yt-dlp)")
st.markdown("""
Enter a YouTube URL. The app will attempt to download the audio using `yt-dlp`
(**requires `ffmpeg` installed system-wide**) and then transcribe it using Deepgram.
""")

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
selected_language_name = st.selectbox(
    "Choose Transcription Language:", options=list(SUPPORTED_LANGUAGES.keys()), index=0
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

def init_session_state():
    if 'transcript' not in st.session_state: st.session_state.transcript = ""
    if 'video_title' not in st.session_state: st.session_state.video_title = "transcript"
    if 'processing' not in st.session_state: st.session_state.processing = False
init_session_state()

transcribe_button = st.button(
    "Transcribe Video", type="primary", disabled=st.session_state.processing
)

if transcribe_button and youtube_url:
    st.session_state.processing = True
    st.session_state.transcript = ""
    st.rerun() # Update UI to disable button

if st.session_state.processing:
    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
        st.session_state.processing = False
        st.rerun()
    else:
        audio_filepath = None # Define for finally block
        with st.spinner("Processing... This may take a while depending on video length."):
            try:
                # 1. Download Audio (Requires ffmpeg)
                audio_filepath, video_title = download_audio_yt_dlp(youtube_url)
                st.session_state.video_title = video_title or "transcript"

                if audio_filepath and os.path.exists(audio_filepath):
                    st.info("Reading downloaded audio file...")
                    with open(audio_filepath, "rb") as audio_file: audio_data = audio_file.read()

                    if not audio_data:
                         st.error("Failed to read audio data from downloaded file.", icon="⚠️")
                         transcript_text = ""
                    else:
                        # 2. Transcribe Audio Data
                        transcript_text = asyncio.run(transcribe_audio_data(audio_data, selected_language_code))
                    st.session_state.transcript = transcript_text
                else:
                    # Error displayed in download function if it failed
                    st.warning("Could not proceed without successfully downloaded audio.", icon="⚠️")
                    st.session_state.transcript = ""
            except Exception as e:
                st.error(f"An unexpected error occurred in the main process: {e}", icon="❌")
                st.session_state.transcript = ""
            finally:
                # 3. Clean up temporary audio file
                if audio_filepath and os.path.exists(audio_filepath):
                    try:
                        os.remove(audio_filepath)
                        st.info("Temporary audio file cleaned up.")
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {audio_filepath}: {e}", icon="⚠️")
                # Reset processing flag and rerun to show results/enable button
                st.session_state.processing = False
                st.rerun()


# --- Display Transcript & Download ---
if st.session_state.transcript:
    st.subheader("Transcription Result:")
    st.text_area("Transcript", st.session_state.transcript, height=300, key="transcript_display")
    st.subheader("Download Transcript:")
    word_buffer = create_word_document(st.session_state.transcript)
    if word_buffer:
        base_filename = sanitize_filename(st.session_state.video_title)
        file_name = f"{base_filename}_{selected_language_code}.docx"
        st.download_button(
            label="Download as Word (.docx)", data=word_buffer, file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_button"
        )
    else: st.error("Could not generate the Word document for download.", icon="📄")
elif transcribe_button and not youtube_url: # Handle case where button clicked with no URL
    st.warning("Please enter a YouTube URL.")
    st.session_state.processing = False


st.markdown("---")
# Use st.query_params - Updated! Added a simple timestamp from Python.
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by Deepgram, yt-dlp, and Streamlit | App loaded: {current_time_str}")
