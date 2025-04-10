# -*- coding: utf-8 -*-
import streamlit as st
import os
import io
import tempfile
import yt_dlp  # For downloading and converting audio
import re
import subprocess
import json
from datetime import datetime
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document
import openai
import math
import shutil # Added for cleanup consistency

# --- (Keep Page Config, Warnings, Helper Functions like load_api_key, get_deepgram_client, SUPPORTED_LANGUAGES, sanitize_filename as before) ---
st.set_page_config(
    page_title="YouTube Transcriber (yt-dlp)",
    layout="wide",
    initial_sidebar_state="auto"
)

st.warning(
    """
**Dependency Alert:** This app relies on `ffmpeg` and `ffprobe` being installed and accessible (e.g., via `packages.txt`).
It uses ffmpeg/ffprobe for audio conversion, duration checking, and splitting. Please monitor logs for any related errors.
""",
    icon="ℹ️"
)

# --- Helper Functions ---

@st.cache_data
def load_api_key(key_name: str) -> str:
    try:
        api_key = st.secrets[key_name]
        if not api_key or api_key == f"YOUR_{key_name}_HERE" or len(api_key) < 20:
            st.error(f"Error: {key_name} missing/invalid in Streamlit secrets.", icon="🚨")
            return ""
        return api_key
    except Exception as e:
        st.error(f"Secrets error accessing {key_name}: {e}", icon="🚨")
        return ""

# Load API keys and initialize clients
DEEPGRAM_API_KEY = load_api_key("DEEPGRAM_API_KEY")
openai_available = False # Assume unavailable initially
if DEEPGRAM_API_KEY:
     try:
        config = DeepgramClientOptions(verbose=False)
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
     except Exception as e:
        st.error(f"Deepgram client init error: {e}", icon="🚨")
        st.stop() # Stop if Deepgram fails
else:
    st.warning("Deepgram API key missing/invalid. Deepgram engine disabled.", icon="⚠️")
    # Don't stop here, OpenAI might still work

OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY  # Set OpenAI API key
        openai_available = True
    except Exception as e:
         st.error(f"Failed to set OpenAI API key: {e}", icon="🚨")
else:
    st.warning("OpenAI API key missing/invalid. OpenAI Whisper engine disabled.", icon="⚠️")

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
    "Chinese (Mandarin, Simplified)": "zh-CN"
}

def sanitize_filename(filename: str) -> str:
    if not filename:
        return "transcript"
    base, ext = os.path.splitext(filename)
    san_base = re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+', '_', base)
    san_base = san_base.strip('_-') or "transcript"
    return san_base + ext.lower()

# --- (Keep download_audio_yt_dlp, transcribe_audio_deepgram, get_audio_duration as previously refined) ---
def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None, str | None]:
    """
    Downloads audio from a YouTube URL and converts it to a WAV file.
    Returns (file_path, video_title, temp_directory_path).
    """
    video_title = "audio_transcript"
    temp_dir = tempfile.mkdtemp(prefix="yt_dlp_") # Create a dedicated temp directory
    base_filename = os.path.join(temp_dir, "audio") # Base path within the temp dir
    final_output_path = base_filename + ".wav" # Define the final desired output path (WAV)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_filename, # Let ffmpeg add the .wav extension
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'noplaylist': True,
        'quiet': True, # Keep True unless debugging yt-dlp itself
        'no_warnings': True,
        'socket_timeout': 60,
        'retries': 3,
        'overwrites': True,
        'ffmpeg_location': 'ffmpeg',
    }

    st.info("Downloading and converting audio to WAV...")
    download_success = False
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', video_title)
            if os.path.exists(final_output_path):
                 download_success = True
            else:
                 st.warning(f"Expected WAV file '{final_output_path}' not found directly. Checking temp dir...")
                 found_alternative = False
                 if os.path.exists(temp_dir):
                     for potential_file in os.listdir(temp_dir):
                         if potential_file.startswith("audio.") and potential_file.endswith(".wav"):
                             potential_path = os.path.join(temp_dir, potential_file)
                             st.info(f"Found potential audio file: {potential_path}. Renaming.")
                             try:
                                os.rename(potential_path, final_output_path)
                                download_success = True
                                found_alternative = True
                                break
                             except OSError as rename_e:
                                 st.error(f"Failed to rename {potential_path} to {final_output_path}: {rename_e}", icon="❌")
                 if not found_alternative:
                    st.error(f"Could not find the converted WAV file in {temp_dir}.", icon="❌")

    except yt_dlp.utils.DownloadError as de:
        st.error(f"yt-dlp Download Error: {de}", icon="❌")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, None, None
    except Exception as e:
        st.error(f"yt-dlp processing error: {e}", icon="❌")
        st.exception(e) # Show full traceback
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, None, None

    if not download_success:
         st.error("Audio download or conversion step failed. Output file not found.", icon="❌")
         if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
         return None, None, None

    if not os.path.exists(final_output_path) or os.path.getsize(final_output_path) == 0:
        st.error(f"Conversion failed: output file '{final_output_path}' missing or empty.", icon="❌")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, None, None

    file_size_mb = os.path.getsize(final_output_path)/1024/1024
    st.success(f"Audio download & conversion completed: '{video_title}' ({file_size_mb:.2f} MB).")
    return final_output_path, video_title, temp_dir

def transcribe_audio_deepgram(audio_data: bytes, language_code: str | None, filename_hint: str = "audio") -> str:
    # Same as before
    if not DEEPGRAM_API_KEY:
         st.error("Deepgram API Key not configured.", icon="🚨")
         return "[Deepgram Error: API Key Missing]"
    try:
        payload: FileSource = {"buffer": audio_data}
        options_dict = {
            "model": "nova-2",
            "smart_format": True,
            "punctuate": True,
            "numerals": True,
            "detect_language": True,
        }
        if language_code: # Pass lang hint if provided
           options_dict["language"] = language_code

        options = PrerecordedOptions(**options_dict)
        st.info(f"Sending '{filename_hint}' ({len(audio_data)/1024/1024:.2f} MB) to Deepgram...", icon="📤")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = ""
        detected_lang = "N/A"
        if hasattr(response, 'results') and response.results:
            if hasattr(response.results, 'channels') and response.results.channels:
                first_channel = response.results.channels[0]
                detected_lang = "unknown"
                if hasattr(response, 'metadata') and response.metadata and hasattr(response.metadata, 'detected_language'): # Check metadata first
                    detected_lang = response.metadata.detected_language
                elif hasattr(first_channel, 'detected_language'): # Fallback to channel
                     detected_lang = first_channel.detected_language

                if hasattr(first_channel, 'alternatives') and first_channel.alternatives:
                    first_alternative = first_channel.alternatives[0]
                    if hasattr(first_alternative, 'transcript'):
                        transcript = first_alternative.transcript
        if transcript:
            st.success(f"Deepgram transcription received! (Detected Lang: {detected_lang})", icon="✅")
            return transcript
        else:
            st.warning(f"Deepgram: No text detected. (Detected Lang: {detected_lang}). Resp: {response}", icon="⚠️")
            return "[Transcription empty or failed]"
    except Exception as e:
        st.error("Deepgram transcription failed.", icon="❌")
        st.exception(e)
        return "[Deepgram Transcription Error]"


def get_audio_duration(file_path: str) -> float:
    # Same as before
    if not os.path.exists(file_path):
        st.error(f"Cannot get duration: File not found at {file_path}", icon="❌")
        return 0.0
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    try:
        st.info(f"Running ffprobe for duration: {os.path.basename(file_path)}")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=60)
        st.info("ffprobe successful.")
        duration = float(result.stdout.strip())
        st.info(f"Detected duration: {duration:.2f} seconds")
        return duration
    except FileNotFoundError:
        st.error("`ffprobe` not found. Ensure ffmpeg/ffprobe is installed.", icon="🚨")
        return 0.0
    except subprocess.CalledProcessError as e:
        st.error(f"ffprobe failed (code {e.returncode}): {e.stderr}", icon="❌")
        return 0.0
    except subprocess.TimeoutExpired:
        st.error("ffprobe timed out.", icon="❌")
        return 0.0
    except ValueError as e:
         st.error(f"Could not parse ffprobe duration output '{result.stdout}': {e}", icon="❌")
         return 0.0
    except Exception as e:
        st.error(f"Unexpected ffprobe error: {e}", icon="❌")
        st.exception(e)
        return 0.0


def split_audio_file(input_path: str, segment_duration_secs: float, output_dir: str, re_encode=False) -> list:
    """
    Splits the audio file into segments using ffmpeg.
    Set re_encode=True to force re-encoding (slower, potentially more accurate segment sizes).
    Returns a sorted list of generated segment file paths.
    """
    if not os.path.exists(input_path):
        st.error(f"Cannot split: Input file not found at {input_path}", icon="❌")
        return []
    if segment_duration_secs <= 5: # Ensure segments are reasonably long
        st.warning(f"Segment duration {segment_duration_secs:.1f}s is very short. Adjusting to 10s minimum.", icon="⚠️")
        segment_duration_secs = 10

    base, ext = os.path.splitext(os.path.basename(input_path))
    output_pattern = os.path.join(output_dir, f"{base}_chunk_%04d{ext}")

    command = [
        "ffmpeg",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(int(segment_duration_secs)), # Use integer seconds for segment time
        "-map", "0:a",
        "-y" # Overwrite
    ]

    if re_encode:
        st.warning("Splitting with re-encoding (slower, may fix size issues)...", icon="⏳")
        # Standard WAV format: PCM signed 16-bit little-endian
        command.extend(["-c:a", "pcm_s16le"])
    else:
        st.info("Splitting with -c copy (faster)...")
        command.extend(["-c", "copy", "-reset_timestamps", "1"]) # Keep reset_timestamps with copy

    command.append(output_pattern)

    try:
        st.info(f"Running ffmpeg split command:")
        st.code(" ".join(command))
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        st.info("ffmpeg splitting completed.")
        st.text(f"ffmpeg stderr:\n{result.stderr}") # Show stderr for info/warnings
    except FileNotFoundError:
        st.error("`ffmpeg` not found. Ensure ffmpeg is installed.", icon="🚨")
        return []
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpeg failed splitting (code {e.returncode}):", icon="❌")
        st.error(f"ffmpeg stdout:\n{e.stdout}")
        st.error(f"ffmpeg stderr:\n{e.stderr}")
        return []
    except subprocess.TimeoutExpired:
         st.error("ffmpeg timed out during splitting.", icon="❌")
         return []
    except Exception as e:
        st.error(f"Unexpected error during ffmpeg splitting: {e}", icon="❌")
        st.exception(e)
        return []

    # List and sort the chunk files
    try:
        chunks = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.startswith(f"{base}_chunk_") and f.endswith(ext)
        ])
        if not chunks:
             st.error("ffmpeg ran but created no segment files! Check ffmpeg stderr output above.", icon="❌")
             return []
        st.success(f"Successfully created {len(chunks)} audio segments in {output_dir}.")
        return chunks
    except Exception as e:
        st.error(f"Error listing created chunk files: {e}", icon="❌")
        return []

# *** MAIN CHANGE AREA ***
def transcribe_audio_openai(file_path: str, language_code: str | None, filename_hint: str = "audio") -> str:
    """
    Transcribes audio using OpenAI Whisper. Splits the file if it exceeds size limit.
    Includes extra logging and uses a smaller target size.
    """
    global openai_available
    if not openai_available:
        st.error("OpenAI API Key not configured.", icon="🚨")
        return "[OpenAI Error: API Key Missing]"
    if not os.path.exists(file_path):
        st.error(f"Cannot transcribe: File not found at {file_path}", icon="❌")
        return "[Transcription Error: File Missing]"

    try:
        file_size = os.stat(file_path).st_size
        limit = 25 * 1024 * 1024  # 25 MiB absolute limit
        # *** REDUCED TARGET SIZE SIGNIFICANTLY ***
        safe_limit = 20 * 1024 * 1024 # Use 20 MiB as the target split size

        st.info(f"DEBUG: Original File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        st.info(f"DEBUG: OpenAI absolute limit: {limit} bytes")
        st.info(f"DEBUG: Using safe target limit for splitting: {safe_limit} bytes ({safe_limit / 1024 / 1024:.1f} MB)")

        if file_size > safe_limit:
            st.warning(f"Audio file ({file_size / 1024 / 1024:.2f} MB) exceeds safe limit ({safe_limit / 1024 / 1024:.1f} MB). Splitting...", icon="🔄")

            duration = get_audio_duration(file_path)
            if duration <= 0:
                st.error("Failed to get audio duration. Cannot calculate split size.", icon="❌")
                return "[Transcription Error: Cannot determine duration for splitting]"

            # Calculate target segment duration
            bytes_per_sec = file_size / duration if duration > 0 else 0
            if bytes_per_sec <= 0:
                 st.error(f"Invalid bytes per second calculation ({bytes_per_sec:.2f}). Cannot split.", icon="❌")
                 return "[Transcription Error: Invalid size/duration]"

            # Calculate duration that *should* result in a size <= safe_limit
            target_segment_duration = math.floor(safe_limit / bytes_per_sec)

            min_segment_duration = 15 # Minimum segment length in seconds
            segment_duration = max(target_segment_duration, min_segment_duration)

            st.info(f"DEBUG: Audio duration: {duration:.2f} seconds.")
            st.info(f"DEBUG: Approx bytes per second: {bytes_per_sec:.2f}")
            st.info(f"DEBUG: Calculated target segment duration: {segment_duration:.1f} seconds (min {min_segment_duration}s).")

            segment_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
            st.info(f"DEBUG: Created temporary directory for segments: {segment_dir}")

            # *** TRY SPLITTING WITH -c copy first ***
            segments = split_audio_file(file_path, segment_duration, segment_dir, re_encode=False)

            # *** ADDING A FALLBACK TO RE-ENCODE IF COPY FAILS OR PRODUCES OVERSIZE CHUNKS LATER ***
            # (This fallback logic is complex to add here directly, focus on logging first)
            # For now, rely on the check *inside* the loop.

            if not segments:
                st.error("Failed to split the audio file using '-c copy'. Check ffmpeg install/logs.", icon="❌")
                if os.path.exists(segment_dir): shutil.rmtree(segment_dir)
                return "[Transcription Error: Splitting Failed]"

            transcripts = []
            total_segments = len(segments)
            st.info(f"Starting transcription for {total_segments} segments...")
            progress_bar = st.progress(0)
            any_segment_too_large = False # Flag to track oversized segments

            for i, seg_path in enumerate(segments):
                segment_basename = os.path.basename(seg_path)
                st.info(f"--- Processing Segment {i+1}/{total_segments}: {segment_basename} ---")
                try:
                    if not os.path.exists(seg_path):
                        st.error(f"Segment file {seg_path} missing!", icon="❓")
                        transcripts.append(f"[Error: Segment {i+1} File Missing]")
                        continue

                    seg_size = os.stat(seg_path).st_size
                    st.info(f"DEBUG: Segment {i+1} Size: {seg_size} bytes ({seg_size / 1024 / 1024:.2f} MB)")

                    # *** CRITICAL CHECK: Verify size BEFORE sending ***
                    if seg_size > limit:
                         st.error(f"FATAL: Segment {i+1} ({segment_basename}) with size {seg_size / 1024 / 1024:.2f} MB EXCEEDS the 25MB OpenAI limit!", icon="🔥")
                         st.error("This happened DESPITE splitting. Check ffmpeg splitting accuracy or reduce 'safe_limit' further.")
                         any_segment_too_large = True # Set the flag
                         transcripts.append(f"[Error: Segment {i+1} Exceeded Size Limit]")
                         # Optionally `continue` here to skip transcription attempt, or try anyway and let OpenAI fail
                         continue # Skip sending this oversized segment

                    # Check if segment is empty
                    if seg_size == 0:
                         st.warning(f"Segment {i+1} ({segment_basename}) is empty (0 bytes). Skipping.", icon="⚠️")
                         transcripts.append("") # Add empty string for empty segment
                         continue

                    # Send to OpenAI
                    st.info(f"Transcribing segment {i+1} ({seg_size / 1024 / 1024:.2f} MB)...", icon="📤")
                    with open(seg_path, "rb") as af:
                        transcript_payload = {"model": "whisper-1", "file": af}
                        if language_code:
                            transcript_payload["language"] = language_code
                        else:
                            st.info("No language hint provided to Whisper, using auto-detect.")

                        resp = openai.Audio.transcribe(**transcript_payload)
                        seg_transcript = resp.get("text", "")
                        if seg_transcript:
                           st.info(f"Segment {i+1} transcribed successfully.")
                           transcripts.append(seg_transcript)
                        else:
                           st.warning(f"Segment {i+1} resulted in empty transcript.", icon="⚠️")
                           transcripts.append("")

                except openai.APIError as api_err:
                     # Check if THIS specific API call failed with 413, even though we checked size before
                     if "413" in str(api_err) or (hasattr(api_err, 'http_status') and api_err.http_status == 413):
                         st.error(f"FATAL: OpenAI returned 413 ERROR for Segment {i+1} ({segment_basename}) with reported size {seg_size / 1024 / 1024:.2f} MB.", icon="🔥")
                         st.error("This indicates an issue with OpenAI's size check or our local size check being inaccurate.")
                         any_segment_too_large = True # Also flag this as a size issue
                     else:
                         st.error(f"OpenAI API Error on segment {i+1} ({segment_basename}): {api_err}", icon="❌")
                     transcripts.append(f"[API Error Segment {i+1}]")
                except Exception as seg_e:
                    st.error(f"Error processing segment {i+1} ({segment_basename}): {seg_e}", icon="❌")
                    st.exception(seg_e)
                    transcripts.append(f"[Error Segment {i+1}]")
                finally:
                    # Clean up segment file
                    if os.path.exists(seg_path):
                        try: os.remove(seg_path)
                        except Exception as rm_e: st.warning(f"Could not remove segment file {seg_path}: {rm_e}", icon="⚠️")
                    # Update progress
                    progress_bar.progress((i + 1) / total_segments)

            # Cleanup segment directory
            if os.path.exists(segment_dir):
                st.info(f"Removing temporary segment directory: {segment_dir}")
                try: shutil.rmtree(segment_dir)
                except Exception as rmdir_e: st.warning(f"Could not remove segment dir {segment_dir}: {rmdir_e}", icon="⚠️")

            # Check if the main reason for failure was oversized segments
            if any_segment_too_large:
                 st.error("One or more segments exceeded the size limit AFTER splitting. Transcription may be incomplete or failed.", icon="❌")
                 # Consider suggesting the re-encode split method if this happens
                 st.warning("Suggestion: If this happens consistently, try modifying the code to use `re_encode=True` in the `split_audio_file` call (this will be slower).")


            # Join the transcripts
            st.success("All segments processed. Joining transcripts...", icon="🔗")
            full_transcript = " ".join(transcripts).strip()

            if all(t.startswith("[Error") or t.startswith("[API Error") for t in transcripts if t):
                 st.error("Transcription failed for all segments.", icon="❌")
                 return "[Transcription Failed: All Segments Errored]"
            if not full_transcript and not any(t for t in transcripts): # Check if all segments were empty/skipped
                 st.warning("Transcription resulted in empty text, possibly due to empty/skipped segments.", icon ="⚠️")
                 return "[Transcription resulted in empty text after processing segments]"

            return full_transcript

        else:
            # File is within the SAFE limit, transcribe directly
            st.info(f"Audio file ({file_size / 1024 / 1024:.2f} MB) is within safe limit. Transcribing directly...", icon="✅")
            with open(file_path, "rb") as af:
                 transcript_payload = {"model": "whisper-1", "file": af}
                 if language_code:
                    transcript_payload["language"] = language_code
                 else:
                     st.info("No language hint provided to Whisper, using auto-detect.")

                 response = openai.Audio.transcribe(**transcript_payload)
                 transcript = response.get("text", "")
                 st.success("Direct transcription complete.", icon="✅")
                 return transcript if transcript else "[Transcription empty or failed]"

    except openai.APIError as api_err:
         st.error(f"OpenAI API Error: {api_err}", icon="❌")
         if "413" in str(api_err) or (hasattr(api_err, 'http_status') and api_err.http_status == 413):
              st.error("Received 413 Error. This shouldn't happen with direct transcription if size check passed. File size: {file_size / 1024 / 1024:.2f} MB", icon="🔥")
         return "[OpenAI Transcription API Error]"
    except FileNotFoundError as fnf_err:
         st.error(f"File Not Found Error during transcription: {fnf_err}. Check paths and ffmpeg/ffprobe installation.", icon="🚨")
         return "[Transcription Error: File/Command Not Found]"
    except Exception as e:
        st.error("Unexpected error during OpenAI Whisper transcription.", icon="❌")
        st.exception(e)
        return "[OpenAI Transcription Error]"


def translate_to_english(text: str) -> str:
    # Same as before
    global openai_available
    if not openai_available:
         st.error("OpenAI API Key not configured. Cannot translate.", icon="🚨")
         return "[Translation Error: API Key Missing]"
    if not text or text.startswith("["):
         st.warning("Skipping translation for empty or error text.", icon="⚠️")
         return text
    try:
        st.info("Translating transcript to English using GPT-3.5-turbo...", icon="🔄")
        prompt = f"Translate the following text accurately to English. Maintain the original meaning and tone as much as possible:\n\n---\n{text}\n---"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text to English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        translation = response["choices"][0]["message"]["content"].strip()
        st.success("Translation completed!", icon="✅")
        return translation
    except openai.APIError as api_err:
         st.error(f"OpenAI API Error during translation: {api_err}", icon="❌")
         return f"[Translation API Error] {text}"
    except Exception as e:
        st.error("Translation failed.", icon="❌")
        st.exception(e)
        return f"[Translation Error] {text}"

def create_word_document(text: str) -> io.BytesIO | None:
    # Same as before
    if not text or text.startswith("["):
        st.warning("Cannot create Word document for empty or error transcript.", icon="⚠️")
        return None
    try:
        doc = Document()
        cleaned_text = re.sub(r'\n\s*\n', '\n', text).strip()
        para = doc.add_paragraph()
        para.add_run(cleaned_text)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.info("Word document created successfully.", icon="📄")
        return buf
    except Exception as e:
        st.error(f"Error creating Word document: {e}", icon="❌")
        st.exception(e)
        return None


# --- Main App UI (Mostly unchanged, ensure options adapt to available engines) ---
st.title("🎬 YouTube Video Transcriber")
st.markdown(
    """
Enter a YouTube URL. The app downloads audio, transcribes using Deepgram or OpenAI Whisper (splitting large files for Whisper),
optionally translates Hindi to English, and allows downloading as Word (.docx).
**Important:** Requires `ffmpeg` and `ffprobe`. Check logs for errors.
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)

with col1:
    selected_language_name = st.selectbox(
        "Audio Language Hint (Optional)",
        options=["Auto-Detect"] + list(SUPPORTED_LANGUAGES.keys()),
        index=0,
        help="Select the primary audio language. Helps Whisper, optional for Deepgram. Auto-Detect is default."
    )
    selected_language_code = None
    if selected_language_name != "Auto-Detect":
        selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

with col2:
    transcription_options = []
    if DEEPGRAM_API_KEY: # Check if key loaded successfully
        transcription_options.append("Deepgram")
    if openai_available: # Check if key loaded AND client init potentially ok
        transcription_options.append("OpenAI Whisper")

    if not transcription_options:
         st.error("No transcription engines available. Check API Keys in Streamlit Secrets.", icon="🚨")
         # Disable engine selection if none available
         transcription_engine = st.selectbox("Transcription Engine", ["None Available"], disabled=True)
    else:
        transcription_engine = st.selectbox(
            "Transcription Engine",
            options=transcription_options,
            index=0,
            help="Choose service. Deepgram='nova-2'. Whisper='whisper-1' (splits large files)."
        )

# --- Transcribe Button Logic (Mostly unchanged, ensure cleanup logic is robust) ---
process_button_disabled = not transcription_options
if st.button("Transcribe Audio", disabled=process_button_disabled):
    # Basic URL check
    if not youtube_url or not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        st.warning("Please enter a valid YouTube URL.", icon="⚠️")
    # Check selected engine is still valid (in case keys failed after UI rendered)
    elif transcription_engine == "OpenAI Whisper" and not openai_available:
         st.error("OpenAI Whisper selected, but key/client unavailable.", icon="🚨")
    elif transcription_engine == "Deepgram" and not DEEPGRAM_API_KEY:
          st.error("Deepgram selected, but key/client unavailable.", icon="🚨")
    elif transcription_engine == "None Available":
         st.error("No transcription engine is configured.", icon="🚨")
    else:
        st.info(f"Processing URL: {youtube_url}", icon="⏳")
        st.info(f"Selected Engine: {transcription_engine}")
        st.info(f"Language Hint: {selected_language_name}")

        audio_filepath, video_title, temp_dir_path = None, None, None
        transcript_text = "[Processing Error]" # Default error
        transcription_successful = False

        try:
            # --- Download ---
            audio_filepath, video_title, temp_dir_path = download_audio_yt_dlp(youtube_url)

            if audio_filepath and os.path.exists(audio_filepath):
                # --- Transcription ---
                filename_hint = sanitize_filename(video_title or "youtube_audio") + ".wav"

                if transcription_engine == "Deepgram":
                    st.info("Reading WAV data for Deepgram...", icon="🎧")
                    try:
                        with open(audio_filepath, "rb") as af: audio_data = af.read()
                        if not audio_data:
                            transcript_text = "[File Read Error: Zero Bytes]"
                        else:
                            dg_lang_code = None if selected_language_name == "Auto-Detect" else selected_language_code
                            transcript_text = transcribe_audio_deepgram(audio_data, dg_lang_code, filename_hint)
                    except Exception as read_err:
                        st.error(f"Error reading audio file {audio_filepath}: {read_err}", icon="❌")
                        transcript_text = "[File Read Error]"

                elif transcription_engine == "OpenAI Whisper":
                    st.info("Starting OpenAI Whisper process...", icon="🎧")
                    transcript_text = transcribe_audio_openai(audio_filepath, selected_language_code, filename_hint)

                # --- Check Success ---
                if transcript_text and not transcript_text.startswith("["):
                    transcription_successful = True
            else:
                 st.error("Download/Conversion failed. Cannot transcribe.", icon="❌")
                 transcript_text = "[Download/Conversion Failed]"

        except Exception as main_err:
             st.error(f"An error occurred during the main processing workflow: {main_err}", icon="💥")
             st.exception(main_err)
             transcript_text = "[Main Workflow Error]"
        finally:
            # --- Cleanup ---
            if temp_dir_path and os.path.exists(temp_dir_path):
                 st.info(f"Cleaning up temporary directory: {temp_dir_path}")
                 try:
                     shutil.rmtree(temp_dir_path)
                     st.info("Cleanup successful.", icon="🧹")
                 except Exception as e:
                     st.warning(f"Could not remove temporary directory {temp_dir_path}: {e}", icon="⚠️")
            # Redundant check in case temp_dir_path wasn't set but file exists
            elif audio_filepath and os.path.exists(audio_filepath) and tempfile.gettempdir() in audio_filepath:
                 st.info(f"Cleaning up loose temporary file: {audio_filepath}")
                 try:
                    os.remove(audio_filepath)
                 except Exception as e:
                    st.warning(f"Could not remove loose temp file {audio_filepath}: {e}", icon="⚠️")


        # --- Post-processing & Display ---
        final_transcript = transcript_text
        if transcription_successful and selected_language_name == "Hindi":
            st.info("Attempting English translation for Hindi transcript...", icon="🌐")
            translated = translate_to_english(transcript_text)
            if not translated.startswith("[Translation"):
                 final_transcript = translated
            else:
                 st.warning("Translation failed. Displaying original transcript.", icon="⚠️")
                 # final_transcript remains original transcript_text

        st.subheader(f"📄 Transcription Result for '{video_title or 'Unknown Video'}'")
        if transcription_successful:
            st.text_area("Transcript Text:", final_transcript, height=350, key="transcript_area")
            word_buffer = create_word_document(final_transcript)
            if word_buffer:
                base_filename_noext = os.path.splitext(sanitize_filename(video_title or "transcript"))[0]
                file_name = f"{base_filename_noext}_transcript.docx"
                st.download_button(
                    label="Download as Word (.docx)", data=word_buffer, file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_docx"
                )
            else:
                st.warning("Could not generate Word document.", icon="⚠️")
        else:
            st.error(f"Transcription failed: {final_transcript}", icon="❌")


# --- Footer ---
st.markdown("---")
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by Deepgram, OpenAI Whisper, yt-dlp, ffmpeg, Streamlit. | Loaded: {current_time_str}")
