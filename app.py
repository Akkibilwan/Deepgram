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
import math # Added for ceil

# --- PAGE CONFIGURATION ---
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
if not DEEPGRAM_API_KEY:
    st.stop()
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # No st.stop() here, allow Deepgram to still work
    st.error("OpenAI API key missing/invalid. OpenAI Whisper will not be available.", icon="🚨")
    # Set a flag or handle this state later if OpenAI is selected
    openai_available = False
else:
    openai.api_key = OPENAI_API_KEY  # Set OpenAI API key
    openai_available = True


@st.cache_resource
def get_deepgram_client(api_key):
    try:
        config = DeepgramClientOptions(verbose=False)
        return DeepgramClient(api_key, config)
    except Exception as e:
        st.error(f"Deepgram client init error: {e}", icon="🚨")
        st.stop() # Stop if Deepgram fails, as it's the default

if DEEPGRAM_API_KEY:
    deepgram = get_deepgram_client(DEEPGRAM_API_KEY)
else:
    # Handle case where Deepgram key might be missing but OpenAI isn't
    # Or just rely on the st.stop() in load_api_key for Deepgram
    pass


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
    # Keep the extension if it exists
    base, ext = os.path.splitext(filename)
    # Sanitize the base name
    san_base = re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+', '_', base)
    san_base = san_base.strip('_-') or "transcript" # Ensure not empty
    # Return sanitized base + original extension (lowercased)
    return san_base + ext.lower()

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio from a YouTube URL and converts it to a WAV file.
    Returns (file_path, video_title).
    """
    video_title = "audio_transcript"
    temp_dir = tempfile.mkdtemp() # Create a dedicated temp directory
    base_filename = os.path.join(temp_dir, "audio") # Base path within the temp dir

    # Define the final desired output path (WAV)
    final_output_path = base_filename + ".wav"

    ydl_opts = {
        'format': 'bestaudio/best',
        # yt-dlp needs a template *without* the final extension for post-processing
        'outtmpl': base_filename, # Let ffmpeg add the .wav extension
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'noplaylist': True,
        'quiet': False, # Set to False to potentially see more detailed yt-dlp/ffmpeg logs
        'no_warnings': False, # Set to False for more info
        'socket_timeout': 60, # Increased timeout
        'retries': 3,         # Increased retries
        'overwrites': True,
        # 'verbose': True, # Uncomment for extreme debugging from yt-dlp
        'ffmpeg_location': 'ffmpeg', # Explicitly state ffmpeg command if needed
    }

    st.info("Downloading and converting audio to WAV... (requires ffmpeg)")
    download_success = False
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True) # Download is implicitly handled by postprocessor
            video_title = info_dict.get('title', video_title)
            # Check if the expected output file exists after download/conversion
            if os.path.exists(final_output_path):
                 download_success = True
            else:
                 # Sometimes the extension might be added differently, check common patterns
                 st.warning(f"Expected WAV file '{final_output_path}' not found directly. Checking alternatives...")
                 # Check if yt-dlp outputted with a different extension that ffmpeg should have handled
                 # (This part is less common with 'wav' codec specified but good to be cautious)
                 found_alternative = False
                 for potential_file in os.listdir(temp_dir):
                     if potential_file.startswith("audio.") and potential_file.endswith(".wav"): # Matches base_filename.*.wav
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
        # Clean up partial files/directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        return None, None
    except Exception as e:
        st.error(f"yt-dlp processing error: {e}", icon="❌")
        st.exception(e) # Show full traceback for debugging
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        return None, None

    if not download_success:
         st.error("Audio download or conversion step failed. Output file not found.", icon="❌")
         if os.path.exists(temp_dir):
             import shutil
             shutil.rmtree(temp_dir)
         return None, None

    # Final check on file existence and size
    if not os.path.exists(final_output_path) or os.path.getsize(final_output_path) == 0:
        st.error(f"Conversion failed: output file '{final_output_path}' missing or empty.", icon="❌")
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir) # Clean up directory
        return None, None

    st.success(f"Audio download & conversion completed: '{video_title}' "
               f"({os.path.getsize(final_output_path)/1024/1024:.2f} MB). Saved to {final_output_path}")
    # Return the path to the final WAV file, the title, and the temp directory for cleanup later
    return final_output_path, video_title, temp_dir


def transcribe_audio_deepgram(audio_data: bytes, language_code: str, filename_hint: str = "audio") -> str:
    """
    Transcribes audio using Deepgram.
    """
    if not DEEPGRAM_API_KEY:
         st.error("Deepgram API Key not configured.", icon="🚨")
         return "[Deepgram Error: API Key Missing]"
    try:
        payload: FileSource = {"buffer": audio_data}
        # Use 'nova-2' for potentially better accuracy and language detection
        # Specify language if known, otherwise rely on detect_language
        options_dict = {
            "model": "nova-2", # Or "base" if preferred/cost-sensitive
            "smart_format": True,
            "punctuate": True,
            "numerals": True,
            "detect_language": True, # Keep this enabled
        }
        # If a specific language is selected (and not relying purely on detection), add it
        # Deepgram's detect_language works best when *not* specifying a language,
        # but you might want to guide it if confident. Test this behaviour.
        # if language_code:
        #    options_dict["language"] = language_code

        options = PrerecordedOptions(**options_dict)

        st.info(f"Sending '{filename_hint}' ({len(audio_data)/1024/1024:.2f} MB) to Deepgram...", icon="📤")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        transcript = ""
        detected_lang = "N/A" # Default if not found

        # Safer attribute checking
        if hasattr(response, 'results') and response.results:
            if hasattr(response.results, 'channels') and response.results.channels:
                first_channel = response.results.channels[0]
                # Check for detected language (might be in metadata or channel)
                detected_lang = "unknown"
                if hasattr(response, 'metadata') and response.metadata and hasattr(response.metadata, 'language'):
                    detected_lang = response.metadata.language
                elif hasattr(first_channel, 'detected_language'):
                     detected_lang = first_channel.detected_language

                if hasattr(first_channel, 'alternatives') and first_channel.alternatives:
                    first_alternative = first_channel.alternatives[0]
                    if hasattr(first_alternative, 'transcript'):
                        transcript = first_alternative.transcript

        if transcript:
            st.success(f"Deepgram transcription received! (Detected Language: {detected_lang})", icon="✅")
            return transcript
        else:
            st.warning(f"Deepgram transcription completed but no text was detected. (Detected Language: {detected_lang}). Response: {response}", icon="⚠️")
            return "[Transcription empty or failed]"
    except Exception as e:
        st.error("Deepgram transcription failed.", icon="❌")
        st.exception(e)
        return "[Deepgram Transcription Error]"


def get_audio_duration(file_path: str) -> float:
    """
    Uses ffprobe to return the duration (in seconds) of the audio file.
    Returns 0.0 on error.
    """
    if not os.path.exists(file_path):
        st.error(f"Cannot get duration: File not found at {file_path}", icon="❌")
        return 0.0

    command = [
        'ffprobe',
        '-v', 'error',              # Only show errors
        '-show_entries', 'format=duration', # Get duration from format section
        '-of', 'default=noprint_wrappers=1:nokey=1', # Output just the value
        file_path
    ]
    try:
        st.info(f"Running ffprobe to get duration for: {os.path.basename(file_path)}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True, # Raise exception on non-zero exit code
            timeout=60  # Add a timeout
        )
        st.info("ffprobe executed successfully.")
        duration = float(result.stdout.strip())
        return duration
    except FileNotFoundError:
        st.error("`ffprobe` command not found. Please ensure ffmpeg (which includes ffprobe) is installed and in your PATH.", icon="🚨")
        return 0.0
    except subprocess.CalledProcessError as e:
        st.error(f"ffprobe failed with error code {e.returncode}:", icon="❌")
        st.error(f"ffprobe stderr: {e.stderr}")
        return 0.0
    except subprocess.TimeoutExpired:
        st.error("ffprobe timed out while getting duration.", icon="❌")
        return 0.0
    except ValueError as e:
         st.error(f"Could not parse ffprobe duration output: {e}", icon="❌")
         st.error(f"ffprobe stdout: {result.stdout}")
         return 0.0
    except Exception as e:
        st.error(f"An unexpected error occurred while getting duration with ffprobe: {e}", icon="❌")
        st.exception(e)
        return 0.0

def split_audio_file(input_path: str, segment_duration_secs: float, output_dir: str) -> list:
    """
    Splits the audio file into segments using ffmpeg.
    Returns a sorted list of generated segment file paths.
    """
    if not os.path.exists(input_path):
        st.error(f"Cannot split: Input file not found at {input_path}", icon="❌")
        return []
    if segment_duration_secs <= 0:
        st.error("Segment duration must be positive.", icon="❌")
        return []

    base, ext = os.path.splitext(os.path.basename(input_path))
    output_pattern = os.path.join(output_dir, f"{base}_chunk_%04d{ext}") # Use 4 digits for more chunks

    command = [
        "ffmpeg",
        "-i", input_path,           # Input file
        "-f", "segment",            # Use segment muxer
        "-segment_time", str(segment_duration_secs), # Segment duration
        "-c", "copy",               # Copy codec (fast, no re-encoding)
        "-reset_timestamps", "1",   # Reset timestamps for each segment
        "-map", "0:a",              # Ensure only audio stream is mapped
        output_pattern,             # Output pattern
        "-y"                        # Overwrite existing files
    ]
    try:
        st.info(f"Running ffmpeg to split audio into segments of ~{segment_duration_secs:.1f}s...")
        st.code(" ".join(command)) # Show the command being run
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300) # 5 min timeout
        st.info("ffmpeg splitting completed.")
        # st.text(f"ffmpeg stdout:\n{result.stdout}") # Optional: view ffmpeg output
        st.text(f"ffmpeg stderr:\n{result.stderr}") # Stderr often has progress info
    except FileNotFoundError:
        st.error("`ffmpeg` command not found. Please ensure ffmpeg is installed and in your PATH.", icon="🚨")
        return []
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpeg failed during splitting (code {e.returncode}):", icon="❌")
        st.error(f"ffmpeg stdout:\n{e.stdout}")
        st.error(f"ffmpeg stderr:\n{e.stderr}")
        return []
    except subprocess.TimeoutExpired:
         st.error("ffmpeg timed out during splitting.", icon="❌")
         return []
    except Exception as e:
        st.error(f"An unexpected error occurred during ffmpeg splitting: {e}", icon="❌")
        st.exception(e)
        return []

    # List and sort the chunk files created in the output directory
    try:
        chunks = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.startswith(f"{base}_chunk_") and f.endswith(ext)
        ])
        st.success(f"Successfully created {len(chunks)} audio segments.")
        return chunks
    except Exception as e:
        st.error(f"Error listing created chunk files: {e}", icon="❌")
        return []


def transcribe_audio_openai(file_path: str, language_code: str | None, filename_hint: str = "audio") -> str:
    """
    Transcribes audio using OpenAI Whisper via openai.Audio.transcribe.
    Splits the file if it exceeds the size limit.
    """
    global openai_available # Use the global flag
    if not openai_available:
        st.error("OpenAI API Key not configured. Cannot transcribe with Whisper.", icon="🚨")
        return "[OpenAI Error: API Key Missing]"
    if not os.path.exists(file_path):
        st.error(f"Cannot transcribe: File not found at {file_path}", icon="❌")
        return "[Transcription Error: File Missing]"

    try:
        file_size = os.stat(file_path).st_size
        # Increased safety margin: target ~24MB to be safer
        # OpenAI limit is 25 MiB (25 * 1024 * 1024 = 26,214,400 bytes)
        limit = 25 * 1024 * 1024
        safe_limit = 24 * 1024 * 1024 # Use 24 MiB as the target split size

        st.info(f"File size: {file_size / 1024 / 1024:.2f} MB. OpenAI limit: {limit / 1024 / 1024:.0f} MB.")

        if file_size > safe_limit:
            st.warning("Audio file exceeds safety limit (~24MB). Splitting into smaller segments for OpenAI Whisper...", icon="🔄")

            duration = get_audio_duration(file_path)
            if duration <= 0:
                st.error("Failed to get audio duration. Cannot calculate split size.", icon="❌")
                return "[Transcription Error: Cannot determine duration for splitting]"

            # Calculate target segment duration
            # Aim for segments slightly smaller than safe_limit
            bytes_per_sec = file_size / duration
            # Calculate max duration for a segment to be under safe_limit
            target_segment_duration = math.floor(safe_limit / bytes_per_sec)

            # Ensure duration is reasonable (e.g., at least 10 seconds)
            min_segment_duration = 10
            segment_duration = max(target_segment_duration, min_segment_duration)

            st.info(f"Audio duration: {duration:.2f} seconds.")
            st.info(f"Approx bytes per second: {bytes_per_sec:.2f}")
            st.info(f"Calculated target segment duration: {segment_duration:.1f} seconds.")

            # Create a temporary directory for segments
            segment_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
            st.info(f"Created temporary directory for segments: {segment_dir}")

            segments = split_audio_file(file_path, segment_duration, segment_dir)

            if not segments:
                st.error("Failed to split the audio file. Check ffmpeg installation and logs.", icon="❌")
                # Clean up segment directory if splitting failed
                if os.path.exists(segment_dir):
                    import shutil
                    shutil.rmtree(segment_dir)
                return "[Transcription Error: Splitting Failed]"

            transcripts = []
            total_segments = len(segments)
            st.info(f"Starting transcription for {total_segments} segments...")
            progress_bar = st.progress(0)

            for i, seg_path in enumerate(segments):
                segment_basename = os.path.basename(seg_path)
                st.info(f"Transcribing segment {i+1}/{total_segments}: {segment_basename}", icon="📤")
                try:
                    seg_size = os.stat(seg_path).st_size
                    if seg_size > limit:
                         st.error(f"Segment {segment_basename} ({seg_size / 1024 / 1024:.2f} MB) STILL exceeds the 25MB limit! Check splitting logic/ffmpeg.", icon="🔥")
                         # Skip this segment or handle differently? For now, add error marker.
                         transcripts.append(f"[Error: Segment {i+1} too large]")
                         continue # Move to next segment

                    with open(seg_path, "rb") as af:
                        # Always use language if provided, Whisper benefits from it.
                        # If language_code is None, Whisper will auto-detect.
                        transcript_payload = {
                            "model": "whisper-1",
                            "file": af,
                        }
                        if language_code:
                            transcript_payload["language"] = language_code
                        else:
                            st.info("No language code provided to Whisper, auto-detection will be used.")

                        resp = openai.Audio.transcribe(**transcript_payload)
                        seg_transcript = resp.get("text", "")
                        if seg_transcript:
                           st.info(f"Segment {i+1} transcribed successfully.")
                           transcripts.append(seg_transcript)
                        else:
                           st.warning(f"Segment {i+1} resulted in empty transcript.", icon="⚠️")
                           transcripts.append("") # Append empty string to maintain order

                except openai.APIError as api_err:
                     st.error(f"OpenAI API Error on segment {i+1} ({segment_basename}): {api_err}", icon="❌")
                     transcripts.append(f"[API Error Segment {i+1}]")
                except Exception as seg_e:
                    st.error(f"Error transcribing segment {i+1} ({segment_basename}): {seg_e}", icon="❌")
                    st.exception(seg_e) # Show traceback for segment errors
                    transcripts.append(f"[Error Segment {i+1}]")
                finally:
                    # Clean up segment file immediately after processing
                    try:
                        os.remove(seg_path)
                    except Exception as rm_e:
                        st.warning(f"Could not remove segment file {seg_path}: {rm_e}", icon="⚠️")
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_segments)

            # Clean up the temporary segment directory
            try:
                st.info(f"Removing temporary segment directory: {segment_dir}")
                import shutil
                shutil.rmtree(segment_dir)
            except Exception as rmdir_e:
                st.warning(f"Could not remove temporary segment directory {segment_dir}: {rmdir_e}", icon="⚠️")

            # Join the transcripts from all segments
            st.success("All segments processed. Joining transcripts...", icon="🔗")
            full_transcript = " ".join(transcripts).strip() # Join with space, remove leading/trailing whitespace

            # Check if the result is only error messages
            if all("[Error" in t or "[API Error" in t for t in transcripts if t):
                 st.error("Transcription failed for all segments.", icon="❌")
                 return "[Transcription Failed: All Segments Errored]"

            return full_transcript if full_transcript else "[Transcription resulted in empty text after joining segments]"

        else:
            # File is within the safe limit, transcribe directly
            st.info("Audio file is within size limit. Transcribing directly with OpenAI Whisper...", icon="✅")
            with open(file_path, "rb") as af:
                 transcript_payload = {
                    "model": "whisper-1",
                    "file": af,
                 }
                 if language_code:
                    transcript_payload["language"] = language_code
                 else:
                    st.info("No language code provided to Whisper, auto-detection will be used.")

                 response = openai.Audio.transcribe(**transcript_payload)
                 transcript = response.get("text", "")
                 st.success("Direct transcription complete.", icon="✅")
                 return transcript if transcript else "[Transcription empty or failed]"

    except openai.APIError as api_err:
         st.error(f"OpenAI API Error: {api_err}", icon="❌")
         # Check specifically for 413 again, though splitting should prevent it
         if "413" in str(api_err):
              st.error("Received 413 Error even after attempting to split. There might be an issue with file size calculation or splitting.", icon="🔥")
         return "[OpenAI Transcription API Error]"
    except FileNotFoundError:
         # This might catch ffmpeg/ffprobe not found if called earlier in the try block
         st.error("A required file or command (like ffmpeg/ffprobe) was not found. Check installation.", icon="🚨")
         return "[Transcription Error: File/Command Not Found]"
    except Exception as e:
        st.error("An unexpected error occurred during OpenAI Whisper transcription.", icon="❌")
        st.exception(e)
        return "[OpenAI Transcription Error]"


def translate_to_english(text: str) -> str:
    """
    Translates the provided text to English using OpenAI's ChatCompletion API.
    """
    global openai_available
    if not openai_available:
         st.error("OpenAI API Key not configured. Cannot translate.", icon="🚨")
         return "[Translation Error: API Key Missing]"
    if not text or text.startswith("["): # Avoid translating error messages
         st.warning("Skipping translation for empty or error text.", icon="⚠️")
         return text
    try:
        st.info("Translating transcript to English using GPT-3.5-turbo...", icon="🔄")
        prompt = f"Translate the following text accurately to English. Maintain the original meaning and tone as much as possible:\n\n---\n{text}\n---"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # Or consider gpt-4-turbo-preview if available/needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text to English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 # Lower temperature for more deterministic translation
        )
        translation = response["choices"][0]["message"]["content"].strip()
        st.success("Translation completed!", icon="✅")
        return translation
    except openai.APIError as api_err:
         st.error(f"OpenAI API Error during translation: {api_err}", icon="❌")
         return f"[Translation API Error] {text}" # Return original text with error marker
    except Exception as e:
        st.error("Translation failed.", icon="❌")
        st.exception(e)
        return f"[Translation Error] {text}" # Return original text with error marker

def create_word_document(text: str) -> io.BytesIO | None:
    """Creates a Word document (.docx) in memory from the given text."""
    if not text or text.startswith("["): # Don't create doc for errors
        st.warning("Cannot create Word document for empty or error transcript.", icon="⚠️")
        return None
    try:
        doc = Document()
        # Add paragraph with the text
        para = doc.add_paragraph()
        # Add text run - allows preserving line breaks potentially
        # Simple approach: replace multiple newlines with single, then add
        cleaned_text = re.sub(r'\n\s*\n', '\n', text).strip() # Consolidate multiple blank lines
        para.add_run(cleaned_text)

        # Style examples (optional)
        # from docx.shared import Inches
        # doc.sections[0].left_margin = Inches(0.5)
        # doc.sections[0].right_margin = Inches(0.5)
        # style = doc.styles['Normal']
        # font = style.font
        # font.name = 'Calibri'
        # font.size = Pt(11)

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.info("Word document created successfully.", icon="📄")
        return buf
    except Exception as e:
        st.error(f"Error creating Word document: {e}", icon="❌")
        st.exception(e)
        return None

# --- Main App UI ---
st.title("🎬 YouTube Video Transcriber")
st.markdown(
    """
Enter a YouTube URL below. The app will download the audio track (as WAV), transcribe it using your chosen engine,
and display the transcript. For non-English audio (specifically Hindi in this setup), it can optionally translate the result to English.
You can download the final transcript as a Word (.docx) file.

**Important:** Requires `ffmpeg` and `ffprobe` installed in the backend environment (e.g., via `packages.txt` on Streamlit Cloud). Check logs for errors if downloads or processing fails.
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)

with col1:
    selected_language_name = st.selectbox(
        "Audio Language (Optional - Helps Whisper)",
        options=["Auto-Detect"] + list(SUPPORTED_LANGUAGES.keys()),
        index=0, # Default to Auto-Detect
        help="Select the primary language spoken in the audio. 'Auto-Detect' lets the engine decide (recommended for Deepgram, works for Whisper). Whisper performs best if the language is specified."
    )
    # Get language code, handle Auto-Detect case
    selected_language_code = None
    if selected_language_name != "Auto-Detect":
        selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

with col2:
    # Only show OpenAI option if the key is loaded
    transcription_options = ["Deepgram"]
    if openai_available:
        transcription_options.append("OpenAI Whisper")

    transcription_engine = st.selectbox(
        "Transcription Engine",
        options=transcription_options,
        index=0, # Default to Deepgram (or Whisper if it's the only one available)
        help="Choose the transcription service. Deepgram uses 'nova-2' with language detection. OpenAI uses 'whisper-1' (splits large files)."
    )

# Ensure button is only active if a valid engine option exists
process_button_disabled = len(transcription_options) == 0
if process_button_disabled:
    st.warning("No transcription engines available. Please check API key configurations in secrets.", icon="⚠️")

if st.button("Transcribe Audio", disabled=process_button_disabled):
    # Simple URL validation
    if not youtube_url or not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        st.warning("Please enter a valid YouTube URL starting with http:// or https://", icon="⚠️")
    # Ensure selected engine is actually available
    elif transcription_engine == "OpenAI Whisper" and not openai_available:
         st.error("OpenAI Whisper selected, but API key is missing or invalid.", icon="🚨")
    elif transcription_engine == "Deepgram" and not DEEPGRAM_API_KEY:
          st.error("Deepgram selected, but API key is missing or invalid.", icon="🚨")
    else:
        st.info(f"Processing URL: {youtube_url}", icon="⏳")
        st.info(f"Selected Engine: {transcription_engine}")
        if selected_language_code:
             st.info(f"Selected Language Hint: {selected_language_name} ({selected_language_code})")
        else:
             st.info("Language Hint: Auto-Detect")

        # --- Download ---
        audio_filepath, video_title, temp_dir_path = None, None, None # Initialize
        try:
            # Assign result of download function
            audio_filepath, video_title, temp_dir_path = download_audio_yt_dlp(youtube_url)
        except Exception as dl_err:
            st.error(f"An unexpected error occurred during download initiation: {dl_err}", icon="💥")
            # Ensure cleanup happens even if download_audio_yt_dlp itself raises an error before returning
            if temp_dir_path and os.path.exists(temp_dir_path):
                 import shutil
                 try:
                     shutil.rmtree(temp_dir_path)
                     st.info("Cleaned up temporary download directory after error.", icon="🧹")
                 except Exception as clean_e:
                     st.warning(f"Could not clean up temp directory {temp_dir_path} after error: {clean_e}", icon="⚠️")

        if audio_filepath is None or not os.path.exists(audio_filepath):
            st.error("Download or conversion failed. Cannot proceed with transcription.", icon="❌")
            # No 'else' needed here, the rest of the code is inside the 'else' below
        else:
            # --- Transcription ---
            filename_hint = sanitize_filename(video_title or "youtube_audio") + ".wav"
            transcript_text = ""
            transcription_successful = False
            try:
                if transcription_engine == "Deepgram":
                    st.info("Reading downloaded WAV data for Deepgram...", icon="🎧")
                    try:
                        with open(audio_filepath, "rb") as af:
                            audio_data = af.read()
                        if not audio_data:
                            st.error("Failed to read downloaded audio data.", icon="⚠️")
                            transcript_text = "[File Read Error]"
                        else:
                            # Use None for language_code to force Deepgram's auto-detect if selected
                            dg_lang_code = None if selected_language_name == "Auto-Detect" else selected_language_code
                            transcript_text = transcribe_audio_deepgram(audio_data, dg_lang_code, filename_hint)
                    except Exception as read_err:
                        st.error(f"Error reading audio file {audio_filepath}: {read_err}", icon="❌")
                        transcript_text = "[File Read Error]"

                elif transcription_engine == "OpenAI Whisper":
                    st.info("Starting OpenAI Whisper transcription process...", icon="🎧")
                    # Pass selected_language_code (can be None for auto-detect)
                    transcript_text = transcribe_audio_openai(audio_filepath, selected_language_code, filename_hint)

                # Check if transcription produced valid text (not an error message)
                if transcript_text and not transcript_text.startswith("["):
                    transcription_successful = True

            except Exception as trans_err:
                st.error(f"Transcription process failed: {trans_err}", icon="❌")
                st.exception(trans_err) # Log full traceback
                transcript_text = "[Transcription Process Error]"
            finally:
                # --- Cleanup Downloaded/Temporary Files ---
                # The original downloaded WAV file might be in temp_dir_path
                # Splitting function cleans its own segments and temp dir
                if temp_dir_path and os.path.exists(temp_dir_path):
                     import shutil
                     try:
                         shutil.rmtree(temp_dir_path)
                         st.info("Cleaned up temporary download directory and WAV file.", icon="🧹")
                     except Exception as e:
                         st.warning(f"Could not remove temporary download directory {temp_dir_path}: {e}", icon="⚠️")
                # Fallback if temp_dir_path wasn't set but audio_filepath exists somehow (less ideal)
                elif audio_filepath and os.path.exists(audio_filepath) and os.path.dirname(audio_filepath) == tempfile.gettempdir():
                     try:
                         os.remove(audio_filepath)
                         st.info("Cleaned up temporary WAV file.", icon="🧹")
                     except Exception as e:
                          st.warning(f"Could not remove temporary file {audio_filepath}: {e}", icon="⚠️")


            # --- Post-processing (Translation for Hindi) ---
            final_transcript = transcript_text
            if transcription_successful and selected_language_name == "Hindi":
                st.info("Hindi language selected, attempting translation to English...", icon="🌐")
                final_transcript = translate_to_english(transcript_text)
                # Check if translation failed
                if final_transcript.startswith("[Translation"):
                    st.warning("Translation failed. Displaying original Hindi transcript.", icon="⚠️")
                    final_transcript = transcript_text # Revert to original if translation fails

            # --- Display Results ---
            st.subheader(f"📄 Transcription Result for '{video_title}'")
            if transcription_successful:
                st.text_area("Transcript Text:", final_transcript, height=350, key="transcript_area")

                # --- Download Button ---
                word_buffer = create_word_document(final_transcript)
                if word_buffer:
                    base_filename = sanitize_filename(video_title or "transcript")
                    # Remove any potential original extension from sanitize_filename result before adding .docx
                    base_filename_noext = os.path.splitext(base_filename)[0]
                    file_name = f"{base_filename_noext}_transcript.docx"
                    st.download_button(
                        label="Download as Word (.docx)",
                        data=word_buffer,
                        file_name=file_name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="download_docx",
                        help="Download the transcript as a Word document."
                    )
                else:
                    st.warning("Could not generate Word document for download.", icon="⚠️")
            else:
                # Display the error message stored in final_transcript
                st.error(f"Transcription failed: {final_transcript}", icon="❌")


# Footer
st.markdown("---")
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by Deepgram, OpenAI Whisper, yt-dlp, ffmpeg, and Streamlit. | App loaded: {current_time_str}")
