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
import openai # Keep the import

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Transcriber", # Simpler title maybe
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
def load_api_key(key_name: str) -> str | None: # Return None on failure
    """Loads API key safely from Streamlit secrets."""
    try:
        api_key = st.secrets[key_name]
        # Add more robust check if needed, e.g., based on key prefix sk-... for OpenAI
        if not api_key or api_key == f"YOUR_{key_name}_HERE" or len(api_key) < 20:
            # Don't show error here, let caller handle missing key message
            # st.error(f"Error: {key_name} missing/invalid.", icon="🚨")
            return None
        return api_key
    except KeyError:
         # Key not found in secrets
         return None
    except Exception as e:
        st.error(f"Secrets error reading {key_name}: {e}", icon="🚨")
        return None

# --- API Key Loading and Client Initialization ---

# Load Deepgram Key and Init Client
DEEPGRAM_API_KEY = load_api_key("DEEPGRAM_API_KEY")
deepgram = None # Initialize to None
if not DEEPGRAM_API_KEY:
    st.warning("Deepgram API key missing. Deepgram transcription disabled.", icon="⚠️")
else:
    @st.cache_resource
    def get_deepgram_client(api_key):
        try:
            config = DeepgramClientOptions(verbose=False)
            client = DeepgramClient(api_key, config)
            st.info("Deepgram client ready.", icon="✔️")
            return client
        except Exception as e:
            st.error(f"Deepgram client init error: {e}", icon="🚨")
            # Don't st.stop() here, allow app to run if only OpenAI key exists
            return None
    deepgram = get_deepgram_client(DEEPGRAM_API_KEY)

# Load OpenAI Key and Init Client
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
openai_client = None # Initialize to None
if not OPENAI_API_KEY:
    st.warning("OpenAI API key missing. OpenAI Whisper transcription disabled.", icon="⚠️")
else:
    # Update OpenAI client initialization for v1.0.0+
    @st.cache_resource
    def get_openai_client(api_key):
        """Initializes OpenAI Client safely."""
        try:
            client = openai.OpenAI(api_key=api_key)
            # Optionally test connection (can add latency)
            # client.models.list()
            st.info("OpenAI client ready.", icon="✔️")
            return client
        except Exception as e:
            st.error(f"OpenAI client init error: {e}", icon="🚨")
            st.exception(e) # Show traceback
            return None
    openai_client = get_openai_client(OPENAI_API_KEY)

# Check if at least one client is available
if not deepgram and not openai_client:
     st.error("Neither Deepgram nor OpenAI API keys/clients are configured. Please add at least one API key in the Streamlit Secrets.", icon="🚨")
     st.stop()


SUPPORTED_LANGUAGES = { # Using standard ISO 639-1 codes where possible
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi",
    "Japanese": "ja", "Russian": "ru", "Chinese": "zh" # OpenAI uses 'zh' generally
}

def sanitize_filename(filename: str) -> str:
    if not filename: return "transcript"
    base=os.path.splitext(filename)[0]; san=re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+','_',base)
    return san.strip('_-') or "transcript"

# download_audio_yt_dlp function remains the same (converting to WAV)
def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """Downloads audio, converts to WAV."""
    # ... (Keep the exact implementation from the previous correct version) ...
    temp_audio_intermediate_path = None; final_audio_path = None; video_title = "audio_transcript"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio: temp_audio_intermediate_path = temp_audio.name
    except Exception as e: st.error(f"Temp file error: {e}", icon="❌"); return None, None
    ydl_opts = {
        'format': 'bestaudio/best', 'outtmpl': temp_audio_intermediate_path,
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav',}],
        'noplaylist': True, 'quiet': False, 'no_warnings': False, 'verbose': False, # Turn off verbose for cleaner UI unless debugging
        'socket_timeout': 45, 'retries': 2, 'overwrites': True, }
    st.info("Attempting download & conversion to WAV..."); progress_placeholder = st.empty(); progress_placeholder.info("Download/Conversion starting...")
    def progress_hook(d): nonlocal video_title; # ... (progress hook logic same as before) ...
        hook_status=d.get('status'); info_dict=d.get('info_dict')
        if hook_status == 'downloading': percent_str=d.get('_percent_str','?%'); speed_str=d.get('_speed_str','?/s'); eta_str=d.get('_eta_str','?s'); progress_placeholder.info(f"Downloading: {percent_str} ({speed_str} ETA: {eta_str})")
        elif hook_status == 'error': progress_placeholder.error("Hook error.", icon="⚠️")
        elif hook_status == 'finished': progress_placeholder.info("Download done. Converting...");
             if info_dict: video_title = info_dict.get('title', video_title)
    ydl_opts['progress_hooks'] = [progress_hook]
    potential_final_path = temp_audio_intermediate_path + ".wav" if temp_audio_intermediate_path else None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.info("Running yt-dlp download & WAV conversion...")
            info_dict = ydl.extract_info(url, download=True); video_title = info_dict.get('title', video_title)
            if potential_final_path and os.path.exists(potential_final_path) and os.path.getsize(potential_final_path) > 0:
                final_audio_path = potential_final_path; st.success(f"Audio conversion OK: '{video_title}' ({os.path.getsize(final_audio_path)/1024/1024:.2f} MB).");
                if temp_audio_intermediate_path != final_audio_path and os.path.exists(temp_audio_intermediate_path):
                    try: os.remove(temp_audio_intermediate_path)
                    except OSError: pass
            elif os.path.exists(temp_audio_intermediate_path) and os.path.getsize(temp_audio_intermediate_path) > 0:
                 final_audio_path = temp_audio_intermediate_path; st.warning("WAV conversion output path questionable.", icon="⚠️");
            else: st.error("Download/Conversion failed: final WAV missing/empty.", icon="❌");
                if potential_final_path and os.path.exists(potential_final_path): os.remove(potential_final_path)
                if os.path.exists(temp_audio_intermediate_path): os.remove(temp_audio_intermediate_path)
                return None, None
        progress_placeholder.empty(); return final_audio_path, video_title
    except Exception as e: # Catch all yt-dlp errors here
        progress_placeholder.empty(); st.error(f"yt-dlp failed: {e}. Check logs if verbose enabled.", icon="❌");
        if temp_audio_intermediate_path and os.path.exists(temp_audio_intermediate_path): try: os.remove(temp_audio_intermediate_path) catch OSError: pass
        if potential_final_path and os.path.exists(potential_final_path): try: os.remove(potential_final_path) catch OSError: pass
        return None, None


# transcribe_audio_deepgram remains mostly the same, uses 'deepgram' client global
def transcribe_audio_deepgram(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    """Uses Deepgram to transcribe the given audio data."""
    if not deepgram: return "[Deepgram Client Error]"
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(model="base", smart_format=True, punctuate=True, numerals=True, detect_language=True)
        st.info(f"Sending '{filename_hint}' to Deepgram...", icon="📤")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options) # Assuming async handled by asyncio.run later
        transcript = ""; detected_lang = "unknown"
        if response and response.results and response.results.channels:
            first_channel = response.results.channels[0]; detected_lang = getattr(first_channel, 'detected_language', detected_lang)
            if first_channel.alternatives: transcript = getattr(first_channel.alternatives[0], 'transcript', '')
        if transcript: st.success(f"Deepgram OK! (Detected: {detected_lang})", icon="✅"); return transcript
        else: st.warning(f"Deepgram OK but no text (Detected: {detected_lang}).", icon="⚠️"); return "[Transcription empty or failed]"
    except Exception as e: st.error("Deepgram transcription failed.", icon="❌"); st.exception(e); return ""

# <<< MODIFIED Function: Use OpenAI Client >>>
def transcribe_audio_openai(client: openai.OpenAI, audio_file_obj, language_code: str, filename_hint: str = "audio") -> str:
    """
    Uses OpenAI Whisper (via the new API client) to transcribe the audio.
    """
    if not client: return "[OpenAI Client Error]"
    try:
        st.info(f"Sending '{filename_hint}' to OpenAI Whisper...", icon="📤")
        # Use the passed-in client instance
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_obj, # Send the file object directly
            language=language_code  # Pass the language code hint
            # Consider adding response_format='text' if only text needed
        )
        # Response is now an object, access text attribute directly
        transcript = getattr(response, 'text', "")

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

# create_word_document remains the same
def create_word_document(text: str) -> io.BytesIO | None:
    if not text or text == "[Transcription empty or failed]": return None
    try: doc=Document(); doc.add_paragraph(text); buf=io.BytesIO(); doc.save(buf); buf.seek(0); return buf
    except Exception as e: st.error(f"Word doc error: {e}", icon="📄"); return None


# --- Main App UI ---
st.title("🎬 YouTube Video Transcriber")
st.markdown(
    """
Enter a YouTube URL, select language and engine. The app will download the audio,
convert to WAV, transcribe, and provide the text / Word download.
*(Requires `ffmpeg` installed via `packages.txt`)*
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)
with col1:
    selected_language_name = st.selectbox(
        "Audio Language Hint:",
        options=list(SUPPORTED_LANGUAGES.keys()), index=0,
        help="Select the expected language. Deepgram uses detection, OpenAI uses this as a hint."
    )
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

with col2:
    # Filter engine choices based on available clients
    available_engines = []
    if deepgram: available_engines.append("Deepgram")
    if openai_client: available_engines.append("OpenAI Whisper")

    if not available_engines:
         st.error("No transcription engines available. Check API Keys.")
         st.stop()

    transcription_engine = st.selectbox(
        "Transcription Engine",
        options=available_engines, index=0,
        help="Choose the transcription service."
    )

# Session state for results
if 'transcript_result' not in st.session_state: st.session_state.transcript_result = None
if 'video_title_result' not in st.session_state: st.session_state.video_title_result = None

# --- Transcription Button ---
if youtube_url:
    button_label=f"Transcribe with {transcription_engine}"
    if st.button(button_label, type="primary", key=f"transcribe_{transcription_engine}"):
        if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
             st.warning("Please enter a valid URL.", icon="⚠️")
        else:
            st.session_state.transcript_result = None # Clear previous results
            st.session_state.video_title_result = None

            with st.spinner("Step 1/2: Downloading and preparing audio..."):
                 audio_filepath, video_title = download_audio_yt_dlp(youtube_url)

            if audio_filepath and os.path.exists(audio_filepath):
                st.session_state.video_title_result = video_title if video_title else "Transcription"
                filename_hint = sanitize_filename(video_title)
                transcript = "[Processing Error]" # Default

                with st.spinner(f"Step 2/2: Transcribing with {transcription_engine}..."):
                    try:
                        if transcription_engine == "Deepgram" and deepgram:
                            st.info("Reading audio for Deepgram...", icon="🎧")
                            with open(audio_filepath, "rb") as audio_file:
                                audio_data = audio_file.read()
                            if audio_data:
                                # Deepgram function doesn't need asyncio.run here, it's already async def
                                # We need to run it within an async context or use asyncio.run
                                # Since Streamlit is sync, use asyncio.run
                                transcript = asyncio.run(
                                    transcribe_audio_deepgram(audio_data, selected_language_code, filename_hint)
                                )
                            else: transcript = "[File Read Error]"
                        elif transcription_engine == "OpenAI Whisper" and openai_client:
                             st.info("Reading audio for OpenAI Whisper...", icon="🎧")
                             # OpenAI needs a file object, not bytes
                             with open(audio_filepath, "rb") as audio_file_obj:
                                 transcript = transcribe_audio_openai(
                                     openai_client, audio_file_obj, selected_language_code, filename_hint
                                 )
                        else:
                             transcript = "[Engine Not Available]"

                    except Exception as e:
                        st.error(f"Transcription execution error: {e}", icon="❌")
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
                 st.session_state.transcript_result = "[Download Failed]" # Indicate failure

# --- Display Results ---
if st.session_state.transcript_result is not None:
    st.subheader(f"📄 Transcription Result for '{st.session_state.video_title_result}'")
    error_states = [
        "[Transcription empty or failed]", "[Download Error]", "[File Read Error]",
        "[Transcription Error]", "[Download Failed]", "[Processing Error]",
        "[Engine Not Available]", "[Deepgram Client Error]", "[OpenAI Client Error]"
    ]
    is_valid = (st.session_state.transcript_result and
                st.session_state.transcript_result not in error_states)

    if is_valid:
        st.text_area("Transcript Text:", st.session_state.transcript_result, height=350)
        word_buffer = create_word_document(st.session_state.transcript_result)
        if word_buffer:
            base_filename = sanitize_filename(st.session_state.video_title_result)
            file_name = f"{base_filename}_transcript.docx"
            st.download_button(label="Download as Word (.docx)", data=word_buffer, file_name=file_name, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        # Display tailored error/empty message
        result_msg = st.session_state.transcript_result
        display_msg = "Transcription failed or produced no result." # Default
        if not result_msg or result_msg == "[Transcription empty or failed]": display_msg = "Transcription result is empty. No speech detected or wrong language hint?"
        elif result_msg == "[Download Failed]" or result_msg == "[Download Error]": display_msg = "Transcript failed: Audio download/conversion error."
        elif result_msg == "[File Read Error]": display_msg = "Transcript failed: Cannot read downloaded audio file."
        elif result_msg == "[Transcription Error]" or result_msg == "[Processing Error]": display_msg = "Transcript failed: Error during transcription processing."
        elif result_msg == "[Engine Not Available]": display_msg = "Transcript failed: Selected engine is not available (check API Keys)."
        elif result_msg == "[Deepgram Client Error]" or result_msg == "[OpenAI Client Error]": display_msg = "Transcript failed: Could not initialize transcription client."

        st.warning(display_msg, icon="⚠️")


# --- Footer ---
st.markdown("---")
current_time_utc = datetime.utcnow()
current_time_local = datetime.now()
try: local_tz_name = current_time_local.astimezone().tzname()
except: local_tz_name = "Local"
st.caption(f"Powered by Deepgram, OpenAI Whisper, yt-dlp, Streamlit | UTC: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')} | Server Time: {current_time_local.strftime('%Y-%m-%d %H:%M:%S')} ({local_tz_name})")
