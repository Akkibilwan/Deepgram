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
import nltk
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from docx import Document
import openai

# Download nltk punkt data if not already present
nltk.download('punkt', quiet=True)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Transcriber (SRT Output)",
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

def format_time(seconds: float) -> str:
    """Formats seconds as an SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

def build_single_srt(transcript: str, duration: float) -> str:
    """Builds a single SRT block covering the entire duration."""
    start_time = "00:00:00,000"
    end_time = format_time(duration)
    return f"1\n{start_time} --> {end_time}\n{transcript.strip()}\n\n"

def generate_srt_by_sentence(transcript: str, total_duration: float) -> str:
    """
    Generates SRT subtitles by splitting the transcript into sentences,
    then further subdividing sentences that are too long such that
    each block is estimated to be at most 10 seconds.
    The allocation is done proportionally based on character counts.
    """
    # Split transcript into sentences.
    sentences = nltk.sent_tokenize(transcript)
    if not sentences:
        return ""
    total_chars = sum(len(s) for s in sentences)
    # seconds per character
    ratio = total_duration / total_chars if total_chars > 0 else 0
    # Maximum number of characters in one 10-second block:
    max_chars = int(10 / ratio) if ratio > 0 else 1000

    blocks = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            blocks.append(sentence)
        else:
            # If the sentence is longer than max_chars, split by spaces
            words = sentence.split()
            current_block = ""
            for word in words:
                if current_block == "":
                    current_block = word
                elif len(current_block) + 1 + len(word) <= max_chars:
                    current_block += " " + word
                else:
                    blocks.append(current_block)
                    current_block = word
            if current_block:
                blocks.append(current_block)
    # Now assign timestamps proportionally.
    srt_text = ""
    current_time = 0.0
    for i, block in enumerate(blocks, start=1):
        block_duration = len(block) * ratio
        # Ensure block duration does not exceed 10 sec (should be the case)
        if block_duration > 10:
            block_duration = 10
        start_ts = format_time(current_time)
        end_ts = format_time(current_time + block_duration)
        srt_text += f"{i}\n{start_ts} --> {end_ts}\n{block.strip()}\n\n"
        current_time += block_duration
    # Adjust the last block end time to equal total_duration
    if current_time < total_duration:
        parts = srt_text.strip().split("\n\n")
        if parts:
            last_block = parts[-1].split("\n")
            if len(last_block) >= 2:
                start_end = last_block[1].split(" --> ")
                if len(start_end) == 2:
                    last_block[1] = f"{start_end[0]} --> {format_time(total_duration)}"
                    parts[-1] = "\n".join(last_block)
                    srt_text = "\n\n".join(parts) + "\n\n"
    return srt_text

@st.cache_data
def load_api_key(key_name: str) -> str:
    try:
        key = st.secrets[key_name]
        if not key or key == f"YOUR_{key_name}_HERE" or len(key) < 20:
            st.error(f"Error: {key_name} missing/invalid.", icon="🚨")
            return ""
        return key
    except Exception as e:
        st.error(f"Secrets error: {e}", icon="🚨")
        return ""

# Load API keys and initialize clients
DEEPGRAM_API_KEY = load_api_key("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    st.stop()
OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key missing. Please add OPENAI_API_KEY to your secrets.", icon="🚨")
    st.stop()

@st.cache_resource
def get_deepgram_client(api_key):
    try:
        config = DeepgramClientOptions(verbose=False)
        return DeepgramClient(api_key, config)
    except Exception as e:
        st.error(f"Deepgram client init error: {e}", icon="🚨")
        st.stop()

deepgram = get_deepgram_client(DEEPGRAM_API_KEY)
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key

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
    base = os.path.splitext(filename)[0]
    san = re.sub(r'[<>:"/\\|?*\s\.\t\n\r\f\v]+', '_', base)
    return san.strip('_-') or "transcript"

def download_audio_yt_dlp(url: str) -> tuple[str | None, str | None]:
    """
    Downloads audio from a YouTube URL and converts it to a WAV file.
    Returns (file_path, video_title).
    """
    video_title = "audio_transcript"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="") as temp_audio:
            temp_audio_path = temp_audio.name
    except Exception as e:
        st.error(f"Failed to create temporary file: {e}", icon="❌")
        return None, None

    output_template = temp_audio_path + ".wav"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'noplaylist': True,
        'quiet': False,
        'no_warnings': True,
        'socket_timeout': 45,
        'retries': 2,
        'overwrites': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.188 Safari/537.36'
        },
        # Optionally, add 'cookies': 'path/to/cookies.txt' if needed.
    }

    st.info("Downloading and converting audio to WAV... (requires ffmpeg)")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', video_title)
    except Exception as e:
        st.error(f"yt-dlp error: {e}", icon="❌")
        if os.path.exists(output_template):
            os.remove(output_template)
        return None, None

    actual_filepath = output_template
    if not os.path.exists(actual_filepath):
        candidate = output_template + ".wav"
        if os.path.exists(candidate):
            actual_filepath = candidate

    if not os.path.exists(actual_filepath) or os.path.getsize(actual_filepath) == 0:
        st.error("Download/Conversion failed: output file missing/empty.", icon="❌")
        if os.path.exists(actual_filepath):
            os.remove(actual_filepath)
        return None, None

    st.success(f"Audio download & conversion completed: '{video_title}' ({os.path.getsize(actual_filepath)/1024/1024:.2f} MB).")
    return actual_filepath, video_title

def get_audio_duration(file_path: str) -> float:
    """
    Uses ffprobe to return the duration (in seconds) of the audio file.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except Exception as e:
        st.error(f"Error getting duration: {e}", icon="❌")
        return 0.0

def split_audio_file(input_path: str, segment_duration: float) -> list:
    """
    Splits the audio file into segments of the given duration (in seconds) using ffmpeg.
    Returns a sorted list of segment file paths.
    """
    base, ext = os.path.splitext(input_path)
    output_pattern = base + "_chunk_%03d" + ext
    command = [
        "ffmpeg", "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-c", "copy",
        output_pattern,
        "-y"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except Exception as e:
        st.error(f"Error during splitting: {e}", icon="❌")
        return []
    chunk_dir = os.path.dirname(input_path)
    chunks = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
                     if f.startswith(os.path.basename(base + "_chunk_")) and f.endswith(ext)])
    return chunks

def transcribe_audio_text(engine: str, file_path: str, language_code: str, filename_hint: str) -> (str, float):
    """
    Transcribes audio (using either Deepgram or OpenAI Whisper) and returns a transcript string along with the full duration.
    For OpenAI Whisper, if the file exceeds the safe limit (25MB minus margin), it is split into segments and the results are concatenated.
    """
    full_duration = get_audio_duration(file_path)
    if engine == "Deepgram":
        with open(file_path, "rb") as af:
            audio_data = af.read()
        try:
            payload = {"buffer": audio_data}
            options = PrerecordedOptions(
                model="base",
                smart_format=True,
                punctuate=True,
                numerals=True,
                detect_language=True,
            )
            st.info(f"Sending '{filename_hint}' to Deepgram...", icon="📤")
            response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
            transcript = ""
            if response and response.results and response.results.channels:
                transcript = response.results.channels[0].alternatives[0].transcript.strip() \
                             if response.results.channels[0].alternatives else ""
            return transcript, full_duration
        except Exception as e:
            st.error("Deepgram transcription failed.", icon="❌")
            st.exception(e)
            return "[Transcription Error]", full_duration
    elif engine == "OpenAI Whisper":
        file_size = os.stat(file_path).st_size
        safe_limit = 25 * 1024 * 1024 - 2048  # 25MB minus 2KB margin
        if file_size > safe_limit:
            st.info("Audio file exceeds safe limit. Splitting into segments for OpenAI Whisper...", icon="🔄")
            duration = full_duration
            if duration <= 0:
                return "[Transcription Error: unable to determine duration]", full_duration
            bytes_per_sec = file_size / duration
            max_seg_duration = (safe_limit / bytes_per_sec) * 0.98  # apply extra 2% margin
            st.info(f"Splitting audio into segments of ~{max_seg_duration:.1f} seconds...", icon="🔄")
            segments = split_audio_file(file_path, max_seg_duration)
            if not segments:
                st.error("Failed to split the audio file.", icon="❌")
                return "[Transcription Error]", full_duration
            transcripts = []
            for seg in segments:
                with open(seg, "rb") as af:
                    resp = openai.Audio.transcribe(
                        model="whisper-1",
                        file=af,
                        language=language_code
                    )
                    seg_transcript = resp.get("text", "").strip()
                    transcripts.append(seg_transcript)
                os.remove(seg)
            transcript = " ".join(transcripts)
            return transcript, full_duration
        else:
            with open(file_path, "rb") as af:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=af,
                    language=language_code
                )
            transcript = response.get("text", "").strip()
            return transcript, full_duration
    else:
        return "[Transcription Error]", full_duration

def generate_srt_by_sentence(transcript: str, total_duration: float) -> str:
    """
    Splits the transcript into sentences and further subdivides long sentences so that
    each block's estimated duration (based on character count) is at most 10 seconds.
    Timestamps are assigned proportionally across the total duration.
    """
    import nltk
    sentences = nltk.sent_tokenize(transcript)
    if not sentences:
        return ""
    total_chars = sum(len(s) for s in sentences)
    ratio = total_duration / total_chars if total_chars > 0 else 0  # seconds per character
    max_chars = int(10 / ratio) if ratio > 0 else 1000  # max characters that roughly correspond to 10 sec
    blocks = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            blocks.append(sentence)
        else:
            words = sentence.split()
            current_block = ""
            for word in words:
                if current_block == "":
                    current_block = word
                elif len(current_block) + 1 + len(word) <= max_chars:
                    current_block += " " + word
                else:
                    blocks.append(current_block)
                    current_block = word
            if current_block:
                blocks.append(current_block)
    srt_text = ""
    current_time = 0.0
    for i, block in enumerate(blocks, start=1):
        block_duration = len(block) * ratio
        # Ensure block duration does not exceed 10 sec
        if block_duration > 10:
            block_duration = 10
        start_ts = format_time(current_time)
        end_ts = format_time(current_time + block_duration)
        srt_text += f"{i}\n{start_ts} --> {end_ts}\n{block.strip()}\n\n"
        current_time += block_duration
    # Adjust the final block end time to equal total_duration if necessary.
    if current_time < total_duration:
        parts = srt_text.strip().split("\n\n")
        if parts:
            last_block = parts[-1].split("\n")
            if len(last_block) >= 2:
                start_end = last_block[1].split(" --> ")
                if len(start_end) == 2:
                    last_block[1] = f"{start_end[0]} --> {format_time(total_duration)}"
                parts[-1] = "\n".join(last_block)
                srt_text = "\n\n".join(parts) + "\n\n"
    return srt_text

def translate_to_english(text: str) -> str:
    """
    Translates the provided SRT transcript to English using OpenAI ChatCompletion,
    keeping timestamps intact.
    """
    try:
        st.info("Translating transcript to English...", icon="🔄")
        prompt = f"Translate the following subtitles to English while keeping the timestamps unchanged. Only translate the dialogue lines:\n\n{text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        translation = response["choices"][0]["message"]["content"].strip()
        st.success("Translation completed!", icon="✅")
        return translation
    except Exception as e:
        st.error("Translation failed.", icon="❌")
        st.exception(e)
        return text

def create_word_document(text: str) -> io.BytesIO | None:
    if not text or text == "[Transcription empty or failed]":
        return None
    try:
        doc = Document()
        doc.add_paragraph(text)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error creating Word document: {e}", icon="❌")
        return None

# --- Main App UI ---
st.title("🎬 YouTube Video Transcriber (SRT Output)")
st.markdown(
    """
Enter a YouTube URL below. The app will download the audio track, transcribe it using either Deepgram or OpenAI Whisper,
and generate the transcript in SRT (SubRip Subtitle) format with meaningful, segmented timestamps (each block up to 10 seconds).
If the audio language is Hindi, the transcript will be translated to English (timestamps will remain unchanged).
*(Requires `ffmpeg` installed in the backend via packages.txt)*
    """
)

youtube_url = st.text_input("Enter YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
selected_language_name = st.selectbox(
    "Audio Language (Language detection enabled)",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0,
    help="Select the expected audio language. (If Hindi is selected, the transcript will be translated to English.)"
)
selected_language_code = SUPPORTED_LANGUAGES[selected_language_name]

transcription_engine = st.selectbox(
    "Transcription Engine",
    options=["Deepgram", "OpenAI Whisper"],
    index=0,
    help="Choose whether to use Deepgram or OpenAI Whisper for transcription."
)

if st.button("Transcribe"):
    if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        st.warning("Please enter a valid URL starting with http:// or https://", icon="⚠️")
    else:
        st.info(f"Processing URL: {youtube_url}", icon="⏳")
        audio_filepath, video_title = download_audio_yt_dlp(youtube_url)
        if audio_filepath is None:
            st.error("Download or conversion failed. Cannot proceed with transcription.", icon="❌")
        else:
            filename_hint = sanitize_filename(video_title)
            # Get full audio duration.
            full_duration = get_audio_duration(audio_filepath)
            # Transcribe to get a full text transcript.
            transcript_text, _ = transcribe_audio_text(transcription_engine, audio_filepath, selected_language_code, filename_hint)
            try:
                # Generate SRT using sentence segmentation.
                transcript_srt = generate_srt_by_sentence(transcript_text, full_duration)
            except Exception as e:
                st.error(f"SRT generation error: {e}", icon="❌")
                transcript_srt = "[SRT generation error]"
            finally:
                try:
                    os.remove(audio_filepath)
                    st.info("Temporary WAV file cleaned up.", icon="🧹")
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {e}", icon="⚠️")
            
            # If Hindi was selected, translate the SRT transcript to English.
            if selected_language_name.lower() == "hindi" and transcript_srt not in ["", "[Transcription empty or failed]"]:
                transcript_srt = translate_to_english(transcript_srt)
            
            st.subheader(f"📄 SRT Transcript for '{video_title}'")
            if transcript_srt and transcript_srt not in ["", "[Transcription empty or failed]"]:
                st.text_area("Transcript (SRT Format):", transcript_srt, height=350)
                word_buffer = create_word_document(transcript_srt)
                if word_buffer:
                    base_filename = sanitize_filename(video_title)
                    file_name = f"{base_filename}_transcript.srt.docx"
                    st.download_button(
                        label="Download as Word (.docx)",
                        data=word_buffer,
                        file_name=file_name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        help="Download your SRT transcript as a Word document."
                    )
            else:
                st.warning("No valid transcript was generated.", icon="⚠️")

st.markdown("---")
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Powered by Deepgram, OpenAI Whisper, yt-dlp, and Streamlit. | App loaded: {current_time_str}")
