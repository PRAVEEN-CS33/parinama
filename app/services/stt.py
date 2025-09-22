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
def translate_segments_with_context(result, dest_lang: str, model_id: str = "llama-3.3-70b-versatile"):
    """
    Translate the entire transcript + all segments in ONE LLM call,
    ensuring context is preserved.
    """
    # Prepare prompt with segments + transcript
    segments_text = "\n".join(
        [f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}" for seg in result["segments"]]
    )

    full_prompt = f"""
            You are a professional translator. 
            Your task is to translate the given transcript into {dest_lang}.
            Make sure to preserve the meaning and context across the entire transcript.
            Return the output in **valid JSON format only**, with exactly two top-level fields:
            1. "segments" -> list of objects with "start", "end", "original_text", "translated_text".
            2. "full_translated_text" -> the full transcript translated with context.

            DO NOT include any explanations or text outside of the JSON.

            Here is the transcript with timestamps:

            {segments_text}

            Full transcript:
            {result['text']}
    """

    completion = client.chat.completions.create(
        model=model_id,  # using a valid supported model
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2,
        max_completion_tokens=3000,
        top_p=1,
    )

    response_text = completion.choices[0].message.content.strip()

    # Try to extract JSON substring
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        raise ValueError("LLM did not return valid JSON. Raw response:\n" + response_text)

    json_str = match.group(0)

    try:
        translated_output = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nResponse:\n{response_text}")

    return translated_output


# Step 3: Full pipeline
def speech_to_text(src_lang: str, dest_lang: str, model_id: str = "llama-3.3-70b-versatile"):
    audio_path = Config.UPLOAD_FOLDER + "/input_audio.wav"
    result = speech_to_text_synthesizer(audio_path, src_lang)
    output = translate_segments_with_context(result, dest_lang, model_id=model_id)
    print(output['segments'])
    return output['segments']