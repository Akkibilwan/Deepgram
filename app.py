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
**Dependency Alert:** This app relies on `ffmpeg` being installed via `packages.txt`. The current step attempts conversion to WAV using ffmpeg. Monitor logs for ffmpeg errors.
""", icon="ℹ️")

# --- Configuration ---
@st.cache_data
def load_api_key():
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"];
        if not api_key or api_key == "YOUR_DEEPGRAM_API_KEY_HERE" or len(api_key) < 20:
            st.error("Error: DEEPGRAM_API_KEY missing/invalid.", icon="🚨"); return None
        return api_key
    except KeyError: st.error("Error: DEEPGRAM_API_KEY not found.", icon="🚨"); return None
    except Exception as e: st.error(f"Secrets error: {e}", icon="🚨"); return None
DEEPGRAM_API_KEY = load_api_key()
if not DEEPGRAM_API_KEY: st.stop()

@st.cache_resource
def get_deepgram_client(api_key):
    try: config=DeepgramClientOptions(verbose=False); deepgram=DeepgramClient(api_key,config); return deepgram
    except Exception as e: st.error(f"Deepgram client init error: {e}", icon="🚨"); st.stop()
deepgram = get_deepgram_client(DEEPGRAM_API_KEY)

SUPPORTED_LANGUAGES = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Hindi": "hi", "Japanese": "ja", "Russian": "ru", "Chinese (Mandarin, Simplified)": "zh-CN"}

# --- Helper Functions ---

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    temp_audio_path = None; video_title = "audio_transcript"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio: # Using WAV
            temp_audio_path = temp_audio.name
    except Exception as e: st.error(f"Failed to create temp file: {e}", icon="❌"); return None, None
    ydl_opts = {
        'format': 'bestaudio/best', 'outtmpl': temp_audio_path,
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav',}], # Convert to WAV
        'noplaylist': True, 'quiet': False, 'no_warnings': False, 'verbose': True,
        'socket_timeout': 45, 'retries': 2, 'overwrites': True, }
    st.info(f"Attempting download & conversion to WAV (requires ffmpeg)...")
    progress_placeholder = st.empty(); progress_placeholder.info("Download/Conversion in progress...")
    def progress_hook(d):
        hook_status=d.get('status'); filename=d.get('filename',''); info_dict=d.get('info_dict')
        if hook_status == 'downloading':
            percent_str=d.get('_percent_str','?%'); speed_str=d.get('_speed_str','?/s'); eta_str=d.get('_eta_str','?s')
            progress_placeholder.info(f"Downloading: {percent_str} ({speed_str} ETA: {eta_str})")
        elif hook_status == 'error': progress_placeholder.error("Hook reported error.", icon="⚠️")
        elif hook_status == 'finished':
             progress_placeholder.info("Download finished. Converting to WAV...")
             if info_dict: nonlocal video_title; video_title = info_dict.get('title', video_title)
    ydl_opts['progress_hooks'] = [progress_hook]
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                st.info("Running yt-dlp download & WAV conversion (verbose mode)...")
                info_dict = ydl.extract_info(url, download=True); video_title = info_dict.get('title', video_title)
                actual_filepath = temp_audio_path
                if not os.path.exists(actual_filepath) or os.path.getsize(actual_filepath) == 0:
                    st.error(f"Download/Conversion failed: output file missing/empty.", icon="❌"); progress_placeholder.empty()
                    if os.path.exists(actual_filepath): os.remove(actual_filepath)
                    return None, None
                st.success(f"Audio download & WAV conversion completed: '{video_title}' ({os.path.getsize(actual_filepath)/1024/1024:.2f} MB).")
                progress_placeholder.empty(); return actual_filepath, video_title
            except yt_dlp.utils.DownloadError as e:
                st.error(f"yt-dlp download/conversion failed: {e}. Check logs.", icon="❌"); progress_placeholder.empty();
                if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                return None, None
            except Exception as e:
                 st.error(f"yt-dlp execution error: {e}", icon="❌"); progress_placeholder.empty()
                 if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                 return None, None
    except Exception as e:
        st.error(f"yt-dlp initialization failed: {e}", icon="❌")
        if temp_audio_path and os.path.exists(temp_audio_path):
             try: os.remove(temp_audio_path)
             except OSError: pass
        return None, None

async def transcribe_audio_data(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    try:
        payload: FileSource = {"buffer": audio_data}
        options: PrerecordedOptions = PrerecordedOptions(
            model="base", smart_format=True, punctuate=True, numerals=True,
            detect_language=True, # Keep language detection
        )
        st.info(f"Sending '{filename_hint}' (approx {len(audio_data)/1024:.1f} KB) to Deepgram (detecting language)...", icon="📤")
        response = await deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = ""; detected_lang = "unknown"
        if response and response.results and response.results.channels:
            first_channel = response.results.channels[0]
            detected_lang = getattr(first_channel, 'detected_language', detected_lang)
            if first_channel and first_channel.alternatives:
                first_alternative = first_channel.alternatives[0]
                if first_alternative and hasattr(first_alternative, 'transcript'):
                     transcript = first_alternative.transcript
        if transcript:
             st.success(f"Transcription received! (Detected Language: {detected_lang})", icon="✅")
             return transcript
        else:
             st.warning(f"Transcription done (detected: {detected_lang}), but no text found.", icon="⚠️")
             return "[Transcription empty or failed]"
    except Exception as e:
        st.error(f"Deepgram transcription failed (detect language):", icon="❌")
        st.exception(e)
        return ""

def create_word_document(text: str) -> io.BytesIO | None:
    if not text or text == "[Transcription empty or failed]": return None
    try: doc=Document(); doc.add_paragraph(text); buf=io.BytesIO(); doc.save(buf); buf.seek(0); return buf
    except Exception as e: st.error(f"Word doc error: {e}", icon="📄"); return None

def sanitize_filename(filename: str) -> str:
    if not filename: return "transcript"
    base=os.path.splitext(filename)[0]; san=re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+','_',base)
    san=san.strip('_-'); san=san if san else "transcript"; return san[:100]

# --- Streamlit App UI ---

st.title("🎬 YouTube Video Transcriber")
st.markdown("""
Enter a YouTube URL below. The app will attempt to download the audio track,
transcribe it using Deepgram, and provide the text for download.
*(Requires `ffmpeg` installed in the backend via `packages.txt`)*
""")

# --- MODIFIED Input Section (No Columns) ---
youtube_url = st.text_input(
    "Enter YouTube URL:", # Label seems fine, error likely elsewhere if persists
    placeholder="e.g., https://www.youtube.com/watch?v=..."
)

selected_language_name = st.selectbox(
    "Audio Language (Note: Language detection enabled)", # Updated label slightly
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0, # Default English
    help="Select the primary expected language. Deepgram will attempt auto-detection."
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]
# --- End of Modification ---


# --- Session State Initialization ---
if 'transcript' not in st.session_state: st.session_state.transcript = None
if 'video_title' not in st.session_state: st.session_state.video_title = "transcript"
if 'processing' not in st.session_state: st.session_state.processing = False
if 'current_url' not in st.session_state: st.session_state.current_url = ""
if 'show_result' not in st.session_state: st.session_state.show_result = False

# --- Transcription Button and Logic ---
if youtube_url:
    transcribe_button = st.button(
        f"Transcribe '{youtube_url[:50]}...' (Detect Lang.)", # Update button label
        type="primary", disabled=st.session_state.processing, key="transcribe_button"
    )
    if transcribe_button and not st.session_state.processing:
        if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
             st.warning("Please enter a valid starting with http:// or https://", icon="⚠️")
        else:
            st.session_state.processing=True; st.session_state.transcript=None; st.session_state.show_result=False
            st.session_state.current_url=youtube_url; st.session_state.video_title="transcript"; st.rerun()

# --- Processing Block ---
if st.session_state.processing:
    url_to_process=st.session_state.current_url; lang_code_to_process=selected_language_code
    if not url_to_process:
        st.error("Internal state error: URL missing."); st.session_state.processing=False; st.session_state.show_result=False; st.rerun()
    else:
        st.info(f"Processing URL: {url_to_process}", icon="⏳"); audio_filepath=None; transcript_text=""
        with st.spinner(f"Step 1/2: Downloading & Converting audio to WAV..."):
            try:
                audio_filepath, video_title = download_audio_yt_dlp(url_to_process)
                st.session_state.video_title = video_title if video_title else "downloaded_audio"
            except Exception as e:
                st.error(f"Download/Conversion error: {e}", icon="❌"); st.session_state.processing = False
                st.session_state.show_result=True; st.session_state.transcript="[Download Error]"; st.rerun()
        if audio_filepath and os.path.exists(audio_filepath):
            with st.spinner(f"Step 2/2: Transcribing WAV audio using Deepgram (detecting language)..."):
                try:
                    st.info("Reading downloaded WAV data...", icon="🎧");
                    with open(audio_filepath, "rb") as audio_file: audio_data = audio_file.read()
                    if not audio_data:
                        st.error("Failed to read WAV data.", icon="⚠️"); transcript_text = "[File Read Error]"
                    else:
                        filename_hint = sanitize_filename(st.session_state.video_title)
                        transcript_text = asyncio.run(transcribe_audio_data(audio_data, lang_code_to_process, filename_hint))
                except Exception as e:
                    st.error(f"Transcription error: {e}", icon="❌"); transcript_text = "[Transcription Error]"
                finally:
                    if os.path.exists(audio_filepath):
                        try: os.remove(audio_filepath); st.info("Temp WAV file cleaned up.", icon="🧹")
                        except Exception as e: st.warning(f"Could not remove temp WAV {audio_filepath}: {e}", icon="⚠️")
        else:
            if audio_filepath is None: st.warning("Transcription skipped: download/conversion failed.", icon="⚠️")
            else: st.warning("Transcription skipped: download/conversion failed (file missing/empty).", icon="⚠️")
            transcript_text = "[Download Failed]"
        st.session_state.transcript=transcript_text; st.session_state.processing=False
        st.session_state.show_result=True; st.rerun()

# --- Display Transcript & Download ---
if st.session_state.show_result:
    st.subheader(f"📄 Transcription Result for '{st.session_state.video_title}'")
    # Check for valid transcript content (not None, not empty, not specific error markers)
    valid_transcript = (st.session_state.transcript and st.session_state.transcript not in [
        "[Transcription empty or failed]", "[Download Error]", "[File Read Error]", "[Transcription Error]", "[Download Failed]"
    ])
    if valid_transcript:
        st.text_area("Transcript Text:", st.session_state.transcript, height=350, key="transcript_display_area")
        st.subheader("⬇️ Download Transcript"); word_buffer = create_word_document(st.session_state.transcript)
        if word_buffer:
            base_filename = sanitize_filename(st.session_state.video_title); file_name = f"{base_filename}_transcript.docx"
            st.download_button(label="Download as Word (.docx)", data=word_buffer, file_name=file_name, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="download_word_button", help="...")
    else:
        # Display specific failure/empty message
        result_message = st.session_state.transcript;
        if not result_message: result_message = "Transcription result is empty. No speech detected?"
        elif result_message == "[Transcription empty or failed]": result_message = "Transcription result is empty or failed (language detected). No speech detected or audio quality issue?"
        elif result_message.startswith("[Download"): result_message = "Transcript generation failed: Audio download/conversion error."
        elif result_message == "[File Read Error]": result_message = "Transcript generation failed: Cannot read downloaded WAV file."
        elif result_message == "[Transcription Error]": result_message = "Transcript generation failed: Deepgram API error. Check logs/error details above."
        st.warning(result_message, icon="⚠️")

# --- Footer ---
st.markdown("---")
current_time_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by Deepgram, yt-dlp, and Streamlit. | App loaded: {current_time_str}")
