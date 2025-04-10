# -*- coding: utf-8 -*-
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
st.warning("""
**Dependency Alert:** This app relies on `ffmpeg` being installed via `packages.txt` in the deployment environment. While the current approach tries to bypass direct conversion, `yt-dlp` might still need `ffmpeg` internally. If errors persist, ensure `packages.txt` contains `ffmpeg` and the app was rebooted.
""", icon="ℹ️")

# --- Configuration ---

@st.cache_data
def load_api_key():
    """Loads Deepgram API key from Streamlit secrets."""
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"]
        if not api_key or api_key == "YOUR_DEEPGRAM_API_KEY_HERE" or len(api_key) < 20:
            st.error("Error: DEEPGRAM_API_KEY is missing, empty, or seems like a placeholder in Streamlit secrets. Please add your actual key.", icon="🚨")
            return None
        return api_key
    except KeyError:
        st.error("Error: DEEPGRAM_API_KEY not found in Streamlit secrets. Please add it via the app settings.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred reading secrets: {e}", icon="🚨")
        return None

DEEPGRAM_API_KEY = load_api_key()

if not DEEPGRAM_API_KEY:
    st.stop()

@st.cache_resource
def get_deepgram_client(api_key):
    """Initializes and returns a Deepgram client."""
    try:
        config: DeepgramClientOptions = DeepgramClientOptions(verbose=False)
        deepgram: DeepgramClient = DeepgramClient(api_key, config)
        return deepgram
    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize Deepgram client: {e}", icon="🚨")
        st.stop()

deepgram = get_deepgram_client(DEEPGRAM_API_KEY)

SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi",
    "Japanese": "ja", "Russian": "ru", "Chinese (Mandarin, Simplified)": "zh-CN",
}

# --- Helper Functions ---

# download_audio_yt_dlp remains unchanged from the previous version (with overwrites=True)
def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads the best audio from URL using yt-dlp to a temporary file
    in its NATIVE format (no ffmpeg post-processing).
    Forces overwrite and uses .webm suffix hint. Enables verbose logging.
    Returns a tuple: (path_to_temp_audio_file, video_title) or (None, None) on failure.
    """
    temp_audio_path = None
    video_title = "audio_transcript"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio_path = temp_audio.name
    except Exception as e:
        st.error(f"Failed to create temporary file: {e}", icon="❌")
        return None, None
    ydl_opts = {
        'format': 'bestaudio/best', 'outtmpl': temp_audio_path, 'noplaylist': True,
        'quiet': False, 'no_warnings': False, 'verbose': True,
        'socket_timeout': 45, 'retries': 2, 'overwrites': True,
    }
    st.info(f"Attempting direct download of best audio stream (forcing overwrite)...")
    progress_placeholder = st.empty()
    progress_placeholder.info("Download in progress... See console/app logs for details.")
    def progress_hook(d):
        hook_status = d.get('status')
        filename = d.get('filename', '')
        info_dict = d.get('info_dict')
        if hook_status == 'downloading':
            percent_str = d.get('_percent_str', '?%')
            speed_str = d.get('_speed_str', '?/s')
            eta_str = d.get('_eta_str', '?s')
            progress_placeholder.info(f"Downloading: {percent_str} ({speed_str} ETA: {eta_str})")
        elif hook_status == 'error':
            progress_placeholder.error("Download hook reported an error. Check logs.", icon="⚠️")
        elif hook_status == 'finished':
             progress_placeholder.info("Download finished.")
             if info_dict: nonlocal video_title; video_title = info_dict.get('title', video_title)
    ydl_opts['progress_hooks'] = [progress_hook]
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                st.info("Running yt-dlp direct download (verbose mode, overwrite enabled)...")
                info_dict = ydl.extract_info(url, download=True)
                video_title = info_dict.get('title', video_title)
                actual_filepath = temp_audio_path
                if not os.path.exists(actual_filepath) or os.path.getsize(actual_filepath) == 0:
                    st.error(f"Download process finished, but the output file '{os.path.basename(actual_filepath)}' is missing or empty (0 bytes). Check logs.", icon="❌")
                    progress_placeholder.empty()
                    if os.path.exists(actual_filepath): os.remove(actual_filepath)
                    return None, None
                st.success(f"Audio direct download completed for '{video_title}'. File size: {os.path.getsize(actual_filepath)} bytes.")
                progress_placeholder.empty()
                return actual_filepath, video_title
            except yt_dlp.utils.DownloadError as e:
                st.error(f"yt-dlp direct download failed: {e}. Check verbose logs.", icon="❌")
                progress_placeholder.empty();
                if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                return None, None
            except Exception as e:
                 st.error(f"An unexpected error occurred during yt-dlp execution: {e}", icon="❌")
                 progress_placeholder.empty()
                 if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                 return None, None
    except Exception as e:
        st.error(f"Failed to initialize yt-dlp downloader: {e}", icon="❌")
        if temp_audio_path and os.path.exists(temp_audio_path):
             try: os.remove(temp_audio_path)
             except OSError: pass
        return None, None

# transcribe_audio_data remains unchanged from previous version (using listen.rest)
async def transcribe_audio_data(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    """Transcribes audio data (bytes) using Deepgram asynchronously."""
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2", smart_format=True, language=language_code,
        )
        st.info(f"Sending '{filename_hint}' (approx {len(audio_data)/1024:.1f} KB) to Deepgram ({language_code})...", icon="📤")
        # Use listen.rest
        response = await deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = ""
        if response and response.results and response.results.channels:
            first_channel = response.results.channels[0]
            if first_channel and first_channel.alternatives:
                first_alternative = first_channel.alternatives[0]
                if first_alternative and hasattr(first_alternative, 'transcript'):
                     transcript = first_alternative.transcript
        if transcript:
             st.success("Transcription received from Deepgram!", icon="✅")
             # Add a debug print here if needed locally: print(f"Transcript received: {transcript[:100]}...")
             return transcript
        else:
             st.warning("Transcription completed, but no transcript text found in the response.", icon="⚠️")
             # Add a debug print here: print("Transcript appears empty or failed.")
             return "[Transcription empty or failed]" # Return specific string
    except Exception as e:
        st.error(f"Deepgram transcription failed:", icon="❌")
        st.exception(e) # Show exception in Streamlit UI
        # Add a debug print here: print(f"Deepgram transcription exception: {e}")
        return "" # Return empty string on exception

# create_word_document remains unchanged
def create_word_document(text: str) -> io.BytesIO | None:
    """Creates a Word document (.docx) in memory containing the text."""
    if not text or text == "[Transcription empty or failed]":
        st.warning("Cannot create Word document from empty or failed transcript.", icon="📄")
        return None
    try:
        document = Document(); document.add_paragraph(text); buffer = io.BytesIO()
        document.save(buffer); buffer.seek(0); return buffer
    except Exception as e:
        st.error(f"Failed to create Word document: {e}", icon="📄"); return None

# sanitize_filename remains unchanged
def sanitize_filename(filename: str) -> str:
    """Removes potentially problematic characters for filenames."""
    if not filename: return "transcript"
    base_name = os.path.splitext(filename)[0]
    sanitized = re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+', '_', base_name)
    sanitized = sanitized.strip('_-'); sanitized = sanitized if sanitized else "transcript"
    return sanitized[:100]

# --- Streamlit App UI ---

st.title("🎬 YouTube Video Transcriber")
st.markdown("""
Enter a YouTube URL below. The app will attempt to download the audio track,
transcribe it using Deepgram, and provide the text for download.
*(Requires `ffmpeg` installed in the backend via `packages.txt`)*
""")

col1, col2 = st.columns([3, 2])
with col1:
    youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
with col2:
    selected_language_name = st.selectbox("Audio Language:", options=list(SUPPORTED_LANGUAGES.keys()), index=0, help="Select the primary language spoken in the video.")
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

# --- Session State Initialization ---
if 'transcript' not in st.session_state: st.session_state.transcript = None # Initialize as None
if 'video_title' not in st.session_state: st.session_state.video_title = "transcript"
if 'processing' not in st.session_state: st.session_state.processing = False
if 'current_url' not in st.session_state: st.session_state.current_url = ""
if 'show_result' not in st.session_state: st.session_state.show_result = False # <<< New state variable

# --- Transcription Button and Logic ---
if youtube_url:
    transcribe_button = st.button(
        f"Transcribe '{youtube_url[:50]}...' in {selected_language_name}",
        type="primary", disabled=st.session_state.processing, key="transcribe_button"
    )
    if transcribe_button and not st.session_state.processing:
        if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
             st.warning("Please enter a valid starting with http:// or https://", icon="⚠️")
        else:
            st.session_state.processing = True
            st.session_state.transcript = None # Reset transcript to None
            st.session_state.show_result = False # Hide previous result
            st.session_state.current_url = youtube_url
            st.session_state.video_title = "transcript"
            st.rerun()

# --- Processing Block ---
if st.session_state.processing:
    url_to_process = st.session_state.current_url
    lang_code_to_process = selected_language_code
    if not url_to_process:
        st.error("Internal state error: URL missing during processing.")
        st.session_state.processing = False; st.session_state.show_result = False
        st.rerun()
    else:
        st.info(f"Processing URL: {url_to_process}", icon="⏳")
        audio_filepath = None; transcript_text = "" # Default to empty string
        with st.spinner(f"Step 1/2: Downloading audio for '{url_to_process[:50]}...'"):
            try:
                audio_filepath, video_title = download_audio_yt_dlp(url_to_process)
                st.session_state.video_title = video_title if video_title else "downloaded_audio"
            except Exception as e:
                st.error(f"Error during audio download phase: {e}", icon="❌")
                st.session_state.processing = False; st.session_state.show_result = True # Show failure
                st.session_state.transcript = "[Download Error]" # Indicate specific error type
                st.rerun()

        # Proceed only if download seemed successful
        if audio_filepath and os.path.exists(audio_filepath):
            with st.spinner(f"Step 2/2: Transcribing audio using Deepgram ({lang_code_to_process})..."):
                try:
                    st.info("Reading downloaded audio data...", icon="🎧")
                    with open(audio_filepath, "rb") as audio_file: audio_data = audio_file.read()
                    if not audio_data:
                        st.error("Failed to read audio data from downloaded file.", icon="⚠️")
                        transcript_text = "[File Read Error]" # Indicate error
                    else:
                        filename_hint = sanitize_filename(st.session_state.video_title)
                        transcript_text = asyncio.run(
                            transcribe_audio_data(audio_data, lang_code_to_process, filename_hint)
                        )
                except Exception as e:
                    st.error(f"Error during transcription phase: {e}", icon="❌")
                    # Exception might be shown by transcribe_audio_data already via st.exception
                    transcript_text = "[Transcription Error]" # Indicate error
                finally:
                    if os.path.exists(audio_filepath):
                        try: os.remove(audio_filepath); st.info("Temporary audio file cleaned up.", icon="🧹")
                        except Exception as e: st.warning(f"Could not remove temporary file {audio_filepath}: {e}", icon="⚠️")
        else:
            if audio_filepath is None :
                 st.warning("Transcription step skipped because audio download failed or was interrupted.", icon="⚠️")
            else:
                 st.warning("Transcription step skipped because audio download failed (file missing/empty post-download). Check logs.", icon="⚠️")
            transcript_text = "[Download Failed]" # Indicate download failure clearly

        # Store result and set flag to show result area
        st.session_state.transcript = transcript_text
        st.session_state.processing = False
        st.session_state.show_result = True # <<< Set flag to show results area
        st.rerun() # Rerun to display results

# --- Display Transcript & Download ---
# <<< Modified logic: Show this section if a result attempt was made >>>
if st.session_state.show_result:
    st.subheader(f"📄 Transcription Result for '{st.session_state.video_title}'")

    # Display transcript if it's not None and not an empty string or specific error marker
    if st.session_state.transcript and \
       st.session_state.transcript not in ["[Transcription empty or failed]", "[Download Error]", "[File Read Error]", "[Transcription Error]", "[Download Failed]"]:
        st.text_area(
            "Transcript Text:",
            st.session_state.transcript,
            height=350,
            key="transcript_display_area"
        )
        st.subheader("⬇️ Download Transcript")
        word_buffer = create_word_document(st.session_state.transcript)
        if word_buffer:
            base_filename = sanitize_filename(st.session_state.video_title)
            file_name = f"{base_filename}_{selected_language_code}.docx"
            st.download_button(
                label="Download as Word (.docx)", data=word_buffer, file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_word_button", help="Click to download the transcript as a Microsoft Word file."
            )
        else:
             # Error handled in create_word_document
             pass
    else:
        # Display specific messages for empty or failed states
        result_message = st.session_state.transcript # Get the specific marker or empty string
        if not result_message: # Handle empty string case explicitly
            result_message = "Transcription result is empty. No speech detected or wrong language selected?"
        elif result_message == "[Transcription empty or failed]":
            result_message = "Transcription result is empty or failed. No speech detected or wrong language selected?"
        elif result_message == "[Download Error]" or result_message == "[Download Failed]":
             result_message = "Could not generate transcript because the audio download failed."
        elif result_message == "[File Read Error]":
             result_message = "Could not generate transcript because the downloaded audio file could not be read."
        elif result_message == "[Transcription Error]":
            result_message = "Could not generate transcript due to an error during the Deepgram API call. Check logs or error details above."

        st.warning(result_message, icon="⚠️")


# --- Footer ---
st.markdown("---")
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by Deepgram, yt-dlp, and Streamlit. | App loaded: {current_time_str}")
