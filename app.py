# -*- coding: utf-8 -*-
import streamlit as st
import os
import io
import tempfile
import yt_dlp  # yt-dlp for downloading and converting audio
import re
import asyncio # Make sure asyncio is imported
from datetime import datetime
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document
# Ensure OpenAI import is correct
try:
    import openai
    from openai import OpenAI as OpenAIClient # Use specific import for clarity if needed
except ImportError:
    st.error("OpenAI library not installed. Please add 'openai>=1.0.0' to requirements.txt")
    openai = None
    OpenAIClient = None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Transcriber",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- FFmpeg Warning ---
st.warning(
    """
**Dependency Alert:** This app relies on `ffmpeg` being installed via `packages.txt`.
It uses ffmpeg to convert the downloaded audio to WAV. Please monitor logs for any ffmpeg errors.
""",
    icon="ℹ️"
)

# --- Helper Functions ---

@st.cache_data
def load_api_key(key_name: str) -> str | None:
    """Loads API key safely from Streamlit secrets."""
    try:
        api_key = st.secrets[key_name]
        if not api_key or api_key == f"YOUR_{key_name}_HERE" or len(api_key) < 20:
            return None
        return api_key
    except KeyError:
        # Key not found is not necessarily an error to display, just return None
        return None
    except Exception as e:
        st.error(f"Secrets error reading {key_name}: {e}", icon="🚨")
        return None

# --- API Key Loading and Client Initialization ---

# Load Deepgram Key and Init Client
DEEPGRAM_API_KEY = load_api_key("DEEPGRAM_API_KEY")
deepgram_client = None # Initialize to None
if not DEEPGRAM_API_KEY:
    st.warning("Deepgram API key missing. Deepgram transcription disabled.", icon="⚠️")
else:
    @st.cache_resource
    def get_deepgram_client(api_key):
        """Initializes Deepgram Client safely."""
        try:
            config = DeepgramClientOptions(verbose=False)
            client = DeepgramClient(api_key, config)
            st.info("Deepgram client ready.", icon="✔️")
            return client
        except Exception as e:
            st.error(f"Deepgram client init error: {e}", icon="🚨")
            return None
    deepgram_client = get_deepgram_client(DEEPGRAM_API_KEY)

# Load OpenAI Key and Init Client
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
openai_client = None # Initialize to None
if openai is None: # Check if import failed
    st.warning("OpenAI library failed to import. OpenAI transcription disabled.", icon="⚠️")
elif not OPENAI_API_KEY:
    st.warning("OpenAI API key missing. OpenAI Whisper transcription disabled.", icon="⚠️")
else:
    @st.cache_resource
    def get_openai_client(api_key):
        """Initializes OpenAI Client safely."""
        try:
            # Use the specific client class if imported
            client = OpenAIClient(api_key=api_key) if OpenAIClient else openai.OpenAI(api_key=api_key)
            # Optional: Test connection (can add latency)
            # client.models.list()
            st.info("OpenAI client ready.", icon="✔️")
            return client
        except Exception as e:
            st.error(f"OpenAI client init error: {e}", icon="🚨")
            st.exception(e) # Show traceback
            return None
    openai_client = get_openai_client(OPENAI_API_KEY)

# Check if at least one client is available
if not deepgram_client and not openai_client:
     st.error("Neither Deepgram nor OpenAI API keys/clients are configured. Cannot transcribe.", icon="🚨")
     st.stop()


SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi",
    "Japanese": "ja", "Russian": "ru", "Chinese": "zh"
}

def sanitize_filename(filename: str) -> str:
    """Removes potentially problematic characters for filenames."""
    if not filename:
        return "transcript"
    base = os.path.splitext(filename)[0]
    # Replace common invalid chars and whitespace with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+', '_', base)
    # Remove leading/trailing underscores/hyphens
    sanitized = sanitized.strip('_-')
    # Ensure not empty
    return sanitized if sanitized else "transcript"

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio using yt-dlp and uses ffmpeg to convert it to WAV.
    Returns tuple: (path_to_final_wav_file, video_title) or (None, None) on failure.
    """
    temp_audio_intermediate_path = None
    final_audio_path = None
    video_title = "audio_transcript" # Default title

    try:
        # Create a temp file path, suffix will be added by yt-dlp/ffmpeg
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_intermediate_path = temp_audio.name
    except Exception as e:
        st.error(f"Failed to create temp file: {e}", icon="❌")
        return None, None

    # Expecting WAV output from postprocessor
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_audio_intermediate_path, # Base path for output/intermediate
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'noplaylist': True, 'quiet': False, 'no_warnings': False, 'verbose': False, # Keep verbose False for cleaner UI
        'socket_timeout': 45, 'retries': 2, 'overwrites': True,
    }

    st.info("Attempting download & conversion to WAV...")
    progress_placeholder = st.empty()
    progress_placeholder.info("Download/Conversion in progress...")

    # --- Carefully Indented progress_hook ---
    def progress_hook(d):
        # This function needs access to video_title in the outer scope
        nonlocal video_title
        # Correct Indentation Level 1 (inside progress_hook)
        hook_status = d.get('status')
        info_dict = d.get('info_dict')

        if hook_status == 'downloading':
            # Correct Indentation Level 2 (inside if)
            percent_str = d.get('_percent_str', '?%')
            speed_str = d.get('_speed_str', '?/s')
            eta_str = d.get('_eta_str', '?s')
            progress_placeholder.info(f"Downloading: {percent_str} ({speed_str} ETA: {eta_str})")
        elif hook_status == 'error':
            # Correct Indentation Level 2 (inside elif)
            progress_placeholder.error("Hook reported error.", icon="⚠️")
        elif hook_status == 'finished':
            # Correct Indentation Level 2 (inside elif)
            progress_placeholder.info("Download finished. Converting to WAV...")
            if info_dict:
                # Correct Indentation Level 3 (inside nested if)
                video_title = info_dict.get('title', video_title)
    # --- End of progress_hook ---

    ydl_opts['progress_hooks'] = [progress_hook]

    potential_final_path = None # Define before try block
    if temp_audio_intermediate_path:
        # yt-dlp *should* replace the suffix, but let's anticipate the double-suffix just in case
        potential_final_path = temp_audio_intermediate_path + ".wav"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.info("Running yt-dlp download & WAV conversion...")
            info_dict = ydl.extract_info(url, download=True)
            # Update title again after info extraction is fully complete
            video_title = info_dict.get('title', video_title)

            # Check logic: Does the intended final path exist?
            # yt-dlp with -x usually renames the final file correctly to the outtmpl path
            final_audio_path_check = temp_audio_intermediate_path # Path specified in outtmpl
            double_extension_path_check = potential_final_path # Path if suffix was added

            if os.path.exists(final_audio_path_check) and os.path.getsize(final_audio_path_check) > 0:
                 final_audio_path = final_audio_path_check
                 st.success(f"Audio conversion OK: '{video_title}' ({os.path.getsize(final_audio_path)/1024/1024:.2f} MB).")
            elif double_extension_path_check and os.path.exists(double_extension_path_check) and os.path.getsize(double_extension_path_check) > 0:
                 # It seems yt-dlp created the .wav.wav file
                 final_audio_path = double_extension_path_check
                 st.warning(f"Audio conversion created file with double extension: {os.path.basename(final_audio_path)}", icon="⚠️")
                 st.success(f"Audio conversion OK: '{video_title}' ({os.path.getsize(final_audio_path)/1024/1024:.2f} MB).")
                 # Clean up intermediate if it still exists
                 if os.path.exists(temp_audio_intermediate_path):
                      try: os.remove(temp_audio_intermediate_path)
                      except OSError: pass
            else:
                st.error("Download/Conversion failed: final WAV missing/empty.", icon="❌")
                # Attempt cleanup of both potential files
                if double_extension_path_check and os.path.exists(double_extension_path_check):
                     try: os.remove(double_extension_path_check)
                     except OSError: pass
                if os.path.exists(temp_audio_intermediate_path):
                     try: os.remove(temp_audio_intermediate_path)
                     except OSError: pass
                return None, None

        progress_placeholder.empty()
        return final_audio_path, video_title # Return the path we confirmed exists

    # --- Carefully Indented Except Blocks ---
    except yt_dlp.utils.DownloadError as e:
        progress_placeholder.empty()
        st.error(f"yt-dlp download/conversion failed: {e}. Check logs.", icon="❌")
        # Cleanup on error
        if temp_audio_intermediate_path and os.path.exists(temp_audio_intermediate_path):
            try:
                os.remove(temp_audio_intermediate_path)
            except OSError: pass
        if potential_final_path and os.path.exists(potential_final_path):
             try:
                 os.remove(potential_final_path)
             except OSError: pass
        return None, None
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"yt-dlp error: {e}", icon="❌")
        # Cleanup on error
        if temp_audio_intermediate_path and os.path.exists(temp_audio_intermediate_path):
             try:
                 os.remove(temp_audio_intermediate_path)
             except OSError: pass
        if potential_final_path and os.path.exists(potential_final_path):
             try:
                 os.remove(potential_final_path)
             except OSError: pass
        return None, None
    # --- End of Except Blocks ---


async def transcribe_audio_deepgram(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    """Uses Deepgram to transcribe the given audio data."""
    if not deepgram_client:
        return "[Deepgram Client Error]"
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(
            model="base", smart_format=True, punctuate=True, numerals=True,
            detect_language=True,
        )
        st.info(f"Sending '{filename_hint}' to Deepgram...", icon="📤")
        # Use the global client instance
        response = await deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
        transcript = ""
        detected_lang = "unknown"
        if response and response.results and response.results.channels:
            first_channel = response.results.channels[0]
            detected_lang = getattr(first_channel, 'detected_language', detected_lang)
            if first_channel.alternatives:
                first_alternative = first_channel.alternatives[0]
                transcript = getattr(first_alternative, 'transcript', '')
        if transcript:
            st.success(f"Deepgram OK! (Detected: {detected_lang})", icon="✅")
            return transcript
        else:
            st.warning(f"Deepgram OK but no text (Detected: {detected_lang}).", icon="⚠️")
            return "[Transcription empty or failed]"
    except Exception as e:
        st.error("Deepgram transcription failed.", icon="❌")
        st.exception(e)
        return ""


def transcribe_audio_openai(client: OpenAIClient, audio_file_obj, language_code: str, filename_hint: str = "audio") -> str:
    """Uses OpenAI Whisper (via API client) to transcribe the audio."""
    if not client:
        return "[OpenAI Client Error]"
    try:
        st.info(f"Sending '{filename_hint}' to OpenAI Whisper...", icon="📤")
        # Use the passed-in client instance
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_obj,
            language=language_code
            # response_format='text' # Can simplify response if needed
        )
        # Response is an object, access text attribute
        transcript = getattr(response, 'text', "")

        if transcript:
            st.success("OpenAI Whisper transcription received!", icon="✅")
            return transcript
        else:
            st.warning("OpenAI Whisper transcription completed but no text detected.", icon="⚠️")
            return "[Transcription empty or failed]"
    except Exception as e:
        st.error("OpenAI Whisper transcription failed.", icon="❌")
        st.exception(e)
        return ""

def create_word_document(text: str) -> io.BytesIO | None:
    """Creates a Word document in memory."""
    if not text or text == "[Transcription empty or failed]":
        return None
    try:
        doc=Document()
        doc.add_paragraph(text)
        buf=io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Word doc error: {e}", icon="📄")
        return None

# --- Main App UI ---
st.title("🎬 YouTube Video Transcriber")
st.markdown(
    """
Enter a YouTube URL, select language hint and engine. The app will download audio,
convert to WAV, transcribe, and provide the text & Word download.
*(Requires `ffmpeg` installed via `packages.txt`)*
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)
with col1:
    selected_language_name = st.selectbox(
        "Audio Language Hint:",
        options=list(SUPPORTED_LANGUAGES.keys()), index=0,
        help="Select expected language. Deepgram auto-detects, OpenAI uses this as hint."
    )
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

with col2:
    available_engines = []
    if deepgram_client: available_engines.append("Deepgram")
    if openai_client: available_engines.append("OpenAI Whisper")

    if not available_engines:
         st.error("No transcription engines available. Check API Keys in Secrets.")
         st.stop()

    transcription_engine = st.selectbox(
        "Transcription Engine", options=available_engines, index=0,
        help="Choose the transcription service."
    )

# Session state for results
if 'transcript_result' not in st.session_state: st.session_state.transcript_result = None
if 'video_title_result' not in st.session_state: st.session_state.video_title_result = None

# --- Transcription Button ---
if youtube_url:
    button_label=f"Transcribe with {transcription_engine}"
    if st.button(button_label, type="primary", key=f"transcribe_{transcription_engine}"):
        # Basic URL check
        if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
             st.warning("Please enter a valid URL.", icon="⚠️")
        else:
             st.session_state.transcript_result = None # Clear previous results
             st.session_state.video_title_result = None

             audio_filepath = None # Define before try block
             video_title = None
             transcript = "[Processing Error]" # Default error

             # --- Download Step ---
             with st.spinner("Step 1/2: Downloading and preparing audio..."):
                  audio_filepath, video_title = download_audio_yt_dlp(youtube_url)
             st.session_state.video_title_result = video_title if video_title else "Transcription" # Store title

             # --- Transcription Step ---
             if audio_filepath and os.path.exists(audio_filepath):
                filename_hint = sanitize_filename(video_title)
                with st.spinner(f"Step 2/2: Transcribing with {transcription_engine}..."):
                    try:
                        if transcription_engine == "Deepgram" and deepgram_client:
                            st.info("Reading audio for Deepgram...", icon="🎧")
                            with open(audio_filepath, "rb") as audio_file:
                                audio_data = audio_file.read()
                            if audio_data:
                                transcript = asyncio.run(
                                    transcribe_audio_deepgram(audio_data, selected_language_code, filename_hint)
                                )
                            else: transcript = "[File Read Error]"
                        elif transcription_engine == "OpenAI Whisper" and openai_client:
                             st.info("Reading audio for OpenAI Whisper...", icon="🎧")
                             # OpenAI needs file obj - reopen it
                             with open(audio_filepath, "rb") as audio_file_obj:
                                 transcript = transcribe_audio_openai(
                                     openai_client, audio_file_obj, selected_language_code, filename_hint
                                 )
                        else:
                             transcript = "[Engine Not Available]"

                    except Exception as e:
                        st.error(f"Transcription execution error: {e}", icon="❌")
                        st.exception(e) # Show traceback in UI
                        transcript = "[Transcription Error]"
                    finally:
                         # Cleanup temporary file
                         try:
                             os.remove(audio_filepath)
                             st.info("Temporary WAV file cleaned up.", icon="🧹")
                         except Exception as e:
                             st.warning(f"Could not remove temporary file: {e}", icon="⚠️")
                st.session_state.transcript_result = transcript # Store result
             else:
                  st.error("Download or conversion failed. Cannot proceed.", icon="❌")
                  st.session_state.transcript_result = "[Download Failed]"

# --- Display Results ---
if st.session_state.transcript_result is not None: # Check if a result exists
    st.subheader(f"📄 Transcription Result for '{st.session_state.video_title_result}'")
    error_states = [
        "[Transcription empty or failed]", "[Download Error]", "[File Read Error]",
        "[Transcription Error]", "[Download Failed]", "[Processing Error]",
        "[Engine Not Available]", "[Deepgram Client Error]", "[OpenAI Client Error]"
    ]
    is_valid = (st.session_state.transcript_result and
                st.session_state.transcript_result not in error_states)

    if is_valid:
        st.text_area(
            "Transcript Text:",
            st.session_state.transcript_result,
            height=350
        )
        st.subheader("⬇️ Download Transcript")
        word_buffer = create_word_document(st.session_state.transcript_result)
        if word_buffer:
            base_filename = sanitize_filename(st.session_state.video_title_result)
            file_name = f"{base_filename}_transcript.docx"
            st.download_button(
                label="Download as Word (.docx)",
                data=word_buffer,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    else:
        # Display tailored error/empty message
        result_msg = st.session_state.transcript_result
        display_msg = "Transcription failed or produced no result."
        if not result_msg or result_msg == "[Transcription empty or failed]": display_msg = "Transcription result is empty. No speech detected or wrong language hint?"
        elif result_msg == "[Download Failed]" or result_msg == "[Download Error]": display_msg = "Transcript failed: Audio download/conversion error."
        elif result_msg == "[File Read Error]": display_msg = "Transcript failed: Cannot read downloaded audio file."
        elif result_msg == "[Transcription Error]" or result_msg == "[Processing Error]": display_msg = "Transcript failed: Error during transcription processing."
        elif result_msg == "[Engine Not Available]": display_msg = "Transcript failed: Selected engine not available (check API Keys)."
        elif result_msg == "[Deepgram Client Error]" or result_msg == "[OpenAI Client Error]": display_msg = "Transcript failed: Could not initialize transcription client."
        st.warning(display_msg, icon="⚠️")

# --- Footer ---
st.markdown("---")
current_time_utc = datetime.utcnow()
current_time_local = datetime.now()
try: local_tz_name = current_time_local.astimezone().tzname()
except: local_tz_name = "Local" # Handle naive datetime
st.caption(f"Powered by Deepgram, OpenAI Whisper, yt-dlp, Streamlit | UTC: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')} | Server Time: {current_time_local.strftime('%Y-%m-%d %H:%M:%S')} ({local_tz_name})")
