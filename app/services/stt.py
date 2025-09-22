import assemblyai as aai
import whisper
import json
import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from app.config import Config
from app.services.consts import WHISPER_LANGUAGES as LANG
from app.services.translate import translate_text

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Transcription
def speech_to_text_synthesizer(audio_path: str, src_lang: str = "english"):
    """Transcribes audio file using Whisper."""
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path, language=src_lang, fp16=False, verbose=False)
    return result

# Step 2: Translation in ONE CALL
CHUNK_SIZE = 10  # number of segments per LLM call

def chunk_segments(segments, chunk_size=CHUNK_SIZE):
    for i in range(0, len(segments), chunk_size):
        yield segments[i:i+chunk_size]

def translate_segments_with_context(result, dest_lang: str, model_id: str = "llama-3.3-70b-versatile"):
    all_translated_segments = []
    full_translated_text = ""

    for chunk in chunk_segments(result["segments"]):
        segments_text = "\n".join([f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}" for seg in chunk])

        full_prompt = f"""
        You are a professional translator. 
        Translate the following transcript into {dest_lang}.
        Return ONLY valid JSON with:
        1. "segments" -> list of objects with "start", "end", "original_text", "translated_text"
        2. "full_translated_text" -> the full translated text for this chunk

        Transcript chunk:
        {segments_text}
        """

        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2,
            max_completion_tokens=2000,
            top_p=1,
        )

        response_text = completion.choices[0].message.content.strip()

        # Extract JSON safely
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return valid JSON. Raw response:\n" + response_text)

        json_str = match.group(0)
        # Remove trailing commas
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
        # Replace single quotes
        json_str = json_str.replace("'", '"')
        # Fix incomplete end
        if not json_str.endswith('}'):
            last_brace = json_str.rfind('}')
            json_str = json_str[:last_brace+1]

        try:
            chunk_result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nResponse:\n{response_text}")

        all_translated_segments.extend(chunk_result["segments"])
        full_translated_text += " " + chunk_result.get("full_translated_text", "")

    return {
        "segments": all_translated_segments,
        "full_translated_text": full_translated_text.strip()
    }


# Step 3: Full pipeline
def speech_to_text(src_lang: str, dest_lang: str, model_id: str = "llama-3.3-70b-versatile"):
    audio_path = Config.UPLOAD_FOLDER + "/input_audio.wav"
    result = speech_to_text_synthesizer(audio_path, src_lang)
    output = translate_segments_with_context(result, dest_lang, model_id=model_id)
    print(output)
    return output