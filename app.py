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
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="YouTube Transcriber (yt-dlp)",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- FFmpeg Warning ---
st.warning("""
**Dependency Alert:** This app relies on `ffmpeg` being installed via `packages.txt` in the deployment environment.
If you still encounter `ffmpeg not found` errors after deployment, ensure `packages.txt` is in your GitHub repo root and contains the line `ffmpeg`, then reboot the app via the Streamlit Cloud dashboard.
""", icon="ℹ️") # Changed icon for less alarm

# --- Configuration ---

@st.cache_data
def load_api_key():
    """Loads Deepgram API key from Streamlit secrets."""
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"]
        # Basic check if the key looks like a placeholder
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

# Stop execution if API key is missing or invalid
if not DEEPGRAM_API_KEY:
    st.stop()

@st.cache_resource # Cache the Deepgram client resource
def get_deepgram_client(api_key):
    """Initializes and returns a Deepgram client."""
    try:
        config: DeepgramClientOptions = DeepgramClientOptions(verbose=False)
        deepgram: DeepgramClient = DeepgramClient(api_key, config)
        st.info("Deepgram client initialized.", icon="🎙️") # Indicate client is ready
        return deepgram
    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize Deepgram client: {e}", icon="🚨")
        st.stop() # Stop if client can't be initialized

deepgram = get_deepgram_client(DEEPGRAM_API_KEY)

# Supported Languages - Consider making this dynamic or configurable if needed
SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi",
    "Japanese": "ja", "Russian": "ru", "Chinese (Mandarin, Simplified)": "zh-CN",
}

# --- Helper Functions ---

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio from URL using yt-dlp to a temporary file using AAC codec.
    Requires ffmpeg installed in the environment. Enables verbose logging.
    Returns a tuple: (path_to_temp_audio_file, video_title) or (None, None) on failure.
    """
    temp_audio_path = None
    video_title = "audio_transcript" # Default title

    # Create a temporary file path first
    try:
        # Using .m4a suffix as we prefer AAC codec
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
            temp_audio_path = temp_audio.name
    except Exception as e:
        st.error(f"Failed to create temporary file: {e}", icon="❌")
        return None, None

    # yt-dlp options - Using AAC codec and Verbose logging
    ydl_opts = {
        'format': 'bestaudio/best', # Prioritize best audio-only
        'outtmpl': temp_audio_path, # Output to the temp file path
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'aac', # <<< Using AAC codec
            'preferredquality': '192', # Standard quality for AAC
        }],
        'noplaylist': True, # Download only single video
        'quiet': False, # Set quiet=False to ensure verbose output is not suppressed internally
        'no_warnings': False, # Show warnings from yt-dlp
        'verbose': True, # <<< Enable verbose logging for detailed output
        'socket_timeout': 45, # Increased timeout slightly
        'retries': 2, # Number of retries on download errors
        # 'ffmpeg_location': '/path/to/your/ffmpeg' # Optional: uncomment and set if ffmpeg isn't in system PATH
    }

    st.info(f"Attempting download & audio extraction (codec: aac)... Check logs for details.")
    # Placeholder for progress, verbose logs will show detailed progress in the console/log file
    progress_placeholder = st.empty()
    progress_placeholder.info("Download/Extraction in progress... See console/app logs for details.")

    # Progress Hook (Less critical with verbose logs, but can provide basic status)
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
             # Post-processing might happen after this 'finished' status for the download part
             progress_placeholder.info("Download finished. Starting audio extraction/conversion...")
             # If info_dict available, update title early (though may not be final title)
             if info_dict:
                 nonlocal video_title # Allow modifying the outer scope variable
                 video_title = info_dict.get('title', video_title)


    ydl_opts['progress_hooks'] = [progress_hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Verbose logs will now print to the standard output/error streams (visible in Streamlit logs)
                st.info("Running yt-dlp (verbose mode)...")
                info_dict = ydl.extract_info(url, download=True)
                # Update title with the final extracted info
                video_title = info_dict.get('title', video_title)

                # Check if the output file was actually created and has size
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                    st.error(f"Download/Extraction process finished, but the output file '{os.path.basename(temp_audio_path)}' is missing or empty. Check the application logs for detailed errors from yt-dlp/ffmpeg.", icon="❌")
                    progress_placeholder.empty()
                    # Attempt cleanup just in case
                    if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                    return None, None

                st.success(f"Audio download & extraction completed for '{video_title}'.")
                progress_placeholder.empty() # Remove progress message on success
                return temp_audio_path, video_title

            except yt_dlp.utils.DownloadError as e:
                st.error(f"yt-dlp download/extraction failed: {e}. Check application logs for verbose details.", icon="❌")
                progress_placeholder.empty()
                if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                return None, None
            except Exception as e: # Catch other potential errors during extraction
                 st.error(f"An unexpected error occurred during yt-dlp execution: {e}", icon="❌")
                 progress_placeholder.empty()
                 if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                 return None, None

    except Exception as e: # Catch errors during YoutubeDL initialization
        st.error(f"Failed to initialize yt-dlp downloader: {e}", icon="❌")
        if temp_audio_path and os.path.exists(temp_audio_path):
             try: os.remove(temp_audio_path)
             except OSError: pass # Ignore error if file couldn't be removed
        return None, None


async def transcribe_audio_data(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    """Transcribes audio data (bytes) using Deepgram asynchronously."""
    try:
        # Prepare source. Providing a filename hint *might* help Deepgram in some cases, but buffer is key.
        payload: FileSource = {"buffer": audio_data}
        # Mimetype might be inferred, but common ones for AAC are audio/aac, audio/mp4, audio/x-m4a
        # Let's try without mimetype first, Deepgram usually detects well.
        # payload["mimetype"] = "audio/aac" # Or "audio/mp4"

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2", # Or choose another, like "base"
            smart_format=True,
            language=language_code,
            # Consider adding other options like:
            # numerals=True,
            # punctuate=True,
            # diarize=True, # If speaker separation is needed
        )

        st.info(f"Sending '{filename_hint}' to Deepgram for transcription ({language_code})...", icon="📤")
        response = await deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Robust check for transcript in response
        transcript = ""
        if response and response.results and response.results.channels:
            first_channel = response.results.channels[0]
            if first_channel and first_channel.alternatives:
                first_alternative = first_channel.alternatives[0]
                if first_alternative and hasattr(first_alternative, 'transcript'):
                     transcript = first_alternative.transcript

        if transcript:
             st.success("Transcription received from Deepgram!", icon="✅")
             return transcript
        else:
             st.warning("Transcription completed, but no transcript text found in the response.", icon="⚠️")
             # Log the structure for debugging if needed
             # st.json(response.to_json(indent=2))
             return "[Transcription empty or failed]"

    except Exception as e:
        st.error(f"Deepgram transcription failed: {e}", icon="❌")
        # Log the exception details for debugging
        # print(f"Deepgram Error: {e}")
        return ""


def create_word_document(text: str) -> io.BytesIO | None:
    """Creates a Word document (.docx) in memory containing the text."""
    if not text:
        st.warning("Cannot create Word document from empty transcript.", icon="📄")
        return None
    try:
        document = Document()
        document.add_paragraph(text)
        # Add more formatting here if desired (e.g., headings, font styles)
        buffer = io.BytesIO()
        document.save(buffer)
        buffer.seek(0) # Rewind buffer to the beginning for reading
        return buffer
    except Exception as e:
        st.error(f"Failed to create Word document: {e}", icon="📄")
        return None


def sanitize_filename(filename: str) -> str:
    """Removes potentially problematic characters for filenames."""
    if not filename: return "transcript"
    # Remove file extension before sanitizing
    base_name = os.path.splitext(filename)[0]
    # Replace common invalid chars and excessive whitespace with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+', '_', base_name)
    # Remove leading/trailing underscores/hyphens that might have resulted
    sanitized = sanitized.strip('_-')
    # Ensure filename isn't empty after sanitization
    sanitized = sanitized if sanitized else "transcript"
    # Limit length to avoid filesystem issues
    return sanitized[:100]


# --- Streamlit App UI ---

st.title("🎬 YouTube Video Transcriber")
st.markdown("""
Enter a YouTube URL below. The app will attempt to download the audio track,
transcribe it using Deepgram, and provide the text for download.
*(Requires `ffmpeg` installed in the backend via `packages.txt`)*
""")

# Input section using columns for better layout
col1, col2 = st.columns([3, 2]) # URL input takes more space

with col1:
    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="e.g., https://www.youtube.com/watch?v=..."
    )

with col2:
    selected_language_name = st.selectbox(
        "Audio Language:",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=0, # Default to English
        help="Select the primary language spoken in the video."
    )
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]


# --- Session State Initialization ---
# Ensures state variables exist across reruns within a session
if 'transcript' not in st.session_state: st.session_state.transcript = ""
if 'video_title' not in st.session_state: st.session_state.video_title = "transcript"
if 'processing' not in st.session_state: st.session_state.processing = False
if 'current_url' not in st.session_state: st.session_state.current_url = ""


# --- Transcription Button and Logic ---

# Only show button if URL is entered
if youtube_url:
    transcribe_button = st.button(
        f"Transcribe '{youtube_url[:50]}...' in {selected_language_name}", # Show partial URL and lang
        type="primary",
        disabled=st.session_state.processing, # Disable if already processing
        key="transcribe_button"
    )

    if transcribe_button and not st.session_state.processing:
        # Check if URL is reasonably valid (basic check)
        if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
             st.warning("Please enter a valid starting with http:// or https://", icon="⚠️")
        else:
            # Start processing
            st.session_state.processing = True
            st.session_state.transcript = "" # Clear previous results
            st.session_state.current_url = youtube_url # Store URL being processed
            st.session_state.video_title = "transcript" # Reset title
            # Rerun immediately to disable button and show spinner logic
            st.rerun()


# --- Processing Block ---
# This block runs only when the 'processing' flag is True
if st.session_state.processing:
    # Retrieve the URL that triggered the processing
    url_to_process = st.session_state.current_url
    lang_code_to_process = selected_language_code # Use language selected when button was clicked

    if not url_to_process: # Safety check
        st.error("Internal state error: URL missing during processing.")
        st.session_state.processing = False
        st.rerun()
    else:
        st.info(f"Processing URL: {url_to_process}", icon="⏳")
        audio_filepath = None # Ensure variable exists for finally block
        transcript_text = "" # Initialize transcript text

        # Use a status indicator during the long operations
        with st.spinner(f"Step 1/2: Downloading audio for '{url_to_process[:50]}...'"):
            try:
                # 1. Download Audio using yt-dlp (with AAC and verbose logs)
                audio_filepath, video_title = download_audio_yt_dlp(url_to_process)
                # Store the extracted video title
                st.session_state.video_title = video_title if video_title else "downloaded_audio"

            except Exception as e: # Catch unexpected errors during download call itself
                st.error(f"Error during audio download phase: {e}", icon="❌")
                # Ensure processing stops if download fails critically here
                st.session_state.processing = False
                st.rerun() # Rerun to show error and enable button


        # Proceed only if download seemed successful (filepath exists)
        if audio_filepath and os.path.exists(audio_filepath):
            with st.spinner(f"Step 2/2: Transcribing audio using Deepgram ({lang_code_to_process})..."):
                try:
                    # 2. Read Audio Data from temp file
                    st.info("Reading downloaded audio data...", icon="🎧")
                    with open(audio_filepath, "rb") as audio_file:
                        audio_data = audio_file.read()

                    if not audio_data:
                        st.error("Failed to read audio data from downloaded file (file might be empty).", icon="⚠️")
                    else:
                        # 3. Transcribe Audio Data
                        # Pass filename hint from video title if available
                        filename_hint = sanitize_filename(st.session_state.video_title)
                        transcript_text = asyncio.run(
                            transcribe_audio_data(audio_data, lang_code_to_process, filename_hint)
                        )
                        # Store the final transcript
                        st.session_state.transcript = transcript_text

                except Exception as e:
                    st.error(f"Error during transcription phase: {e}", icon="❌")
                    st.session_state.transcript = "" # Clear transcript on error
                finally:
                    # 4. Clean up temporary audio file in all cases after processing it
                    if os.path.exists(audio_filepath):
                        try:
                            os.remove(audio_filepath)
                            st.info("Temporary audio file cleaned up.", icon="🧹")
                        except Exception as e:
                            st.warning(f"Could not remove temporary file {audio_filepath}: {e}", icon="⚠️")
        else:
            # Download failed or file was missing - message shown in download function
            st.warning("Transcription step skipped because audio download/extraction failed.", icon="⚠️")
            st.session_state.transcript = "" # Ensure transcript is empty

        # Processing finished, reset flag and rerun to display results/enable button
        st.session_state.processing = False
        st.rerun()


# --- Display Transcript & Download ---
# This runs after processing is complete (due to the rerun)
if st.session_state.transcript:
    st.subheader(f"📄 Transcription Result for '{st.session_state.video_title}'")
    st.text_area(
        "Transcript Text:",
        st.session_state.transcript,
        height=350, # Increased height
        key="transcript_display_area"
    )

    st.subheader("⬇️ Download Transcript")
    # Attempt to create Word document
    word_buffer = create_word_document(st.session_state.transcript)

    if word_buffer: # Only show button if doc creation succeeded
        # Use sanitized video title for filename
        base_filename = sanitize_filename(st.session_state.video_title)
        file_name = f"{base_filename}_{selected_language_code}.docx"

        st.download_button(
            label="Download as Word (.docx)",
            data=word_buffer,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_word_button",
            help="Click to download the transcript as a Microsoft Word file."
        )
    else:
        st.error("Could not generate the Word document for download.", icon="📄")

# Footer or Credits Section
st.markdown("---")
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Simple timestamp
st.caption(f"Powered by Deepgram, yt-dlp, and Streamlit. | App loaded: {current_time_str}")
