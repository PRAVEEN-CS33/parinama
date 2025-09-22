import os
import soundfile as sf
import numpy as np
import tempfile
from pydub import AudioSegment
from gtts import gTTS 
import librosa
import pyrubberband as pyrb

from app.config import Config

# ---------- Supported Indic Languages ----------
# gTTS language codes (supports many Indian languages)
INDIC_MODELS = {
    "hi": "hi",     # Hindi
    "ta": "ta",     # Tamil
    "te": "te",     # Telugu
    "kn": "kn",     # Kannada
    "ml": "ml",     # Malayalam
    "bn": "bn",     # Bengali
    "gu": "gu",     # Gujarati
    "pa": "pa",     # Punjabi
    "mr": "mr",     # Marathi
    "or": "or",     # Odia
    "as": "as"      # Assamese
}

def tts_thread(text: str, lang: str = "hi"):
    """
    Generate speech using gTTS (Google TTS).
    Args:
        text (str): Input text
        lang (str): Language code (hi, ta, te, kn, etc.)
    Returns:
        np.ndarray: audio array
        int: sampling rate
    """
    if lang not in INDIC_MODELS:
        raise ValueError(f"❌ Unsupported language: {lang}")

    # Save gTTS output to temporary mp3
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts = gTTS(text=text, lang=INDIC_MODELS[lang])
    tts.save(temp_out.name)

    # Convert mp3 → wav → numpy
    audio = AudioSegment.from_file(temp_out.name, format="mp3")
    sr = audio.frame_rate
    y = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)

    os.remove(temp_out.name)
    return y, sr
def stretch_to_duration(audio_arr, sr, target_dur):
    """
    Stretch/compress audio to match target duration using pyrubberband (pitch-preserving).

    Args:
        audio_arr (np.ndarray): Input audio array (mono)
        sr (int): Sampling rate
        target_dur (float): Desired duration in seconds

    Returns:
        np.ndarray: Stretched audio array
    """
    cur_dur = len(audio_arr) / sr
    if cur_dur <= 0 or target_dur <= 0:
        return audio_arr

    # Convert to mono if needed
    if audio_arr.ndim > 1:
        audio_arr = np.mean(audio_arr, axis=1)

    # Compute stretch rate
    rate = cur_dur / target_dur

    # Stretch using rubberband (pitch-preserving)
    y_stretched = pyrb.time_stretch(audio_arr, sr, rate)

    # Ensure float32
    y_stretched = y_stretched.astype(np.float32)
    return y_stretched

def text_to_speech(data, lang="hi"):
    """
    Convert segmented text data to aligned audio file (gTTS-based).
    Args:
        data (list): list of dicts with {t_word, start, end}
        lang (str): language code
    Returns:
        str: path to final wav file
    """
    if lang not in INDIC_MODELS:
        raise ValueError(f"❌ Language {lang} not supported. Choose from {list(INDIC_MODELS.keys())}")

    out_path = os.path.join(Config.OUTPUT_FOLDER, "output_audio.wav")
    final = AudioSegment.silent(duration=0)

    for i, seg in enumerate(data):
        txt = seg["translated_text"]
        start, end = seg["start"], seg["end"]
        target_dur = end - start

        # Generate audio
        audio_arr, sr = tts_thread(txt, lang=lang)

        # Stretch/compress to sync with original timing
        y = stretch_to_duration(audio_arr, sr, target_dur)

        temp_wav = os.path.join(Config.UPLOAD_FOLDER, f"seg_{i}.wav")
        sf.write(temp_wav, y, sr)

        segment_audio = AudioSegment.from_wav(temp_wav)
        final += segment_audio

        # Handle gap between segments
        if i < len(data) - 1:
            gap = data[i+1]["start"] - end
            if gap > 0:
                final += AudioSegment.silent(duration=gap*1000)

        os.remove(temp_wav)
        print(f"✅ {txt[:30]}... ({lang}) -> {target_dur:.2f}s")

    final.export(out_path, format="wav")
    print(f"🎵 Final synced audio saved: {out_path}")
    return out_path