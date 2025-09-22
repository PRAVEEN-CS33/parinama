# # import os
# # import whisper
# # import json
# # import re
# # from dotenv import load_dotenv
# # from groq import Groq

# # load_dotenv()
# # client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # # Step 1: Transcription
# # def speech_to_text(audio_path: str, src_lang: str = "english"):
# #     """Transcribes audio file using Whisper."""
# #     model = whisper.load_model("base")
# #     result = model.transcribe(audio_path, language=src_lang, fp16=False, verbose=False)
# #     return result

# # # Step 2: Translation in ONE CALL
# # def translate_segments_with_context(result, dest_lang: str, model_id: str = "llama-3.3-70b-versatile"):
# #     """
# #     Translate the entire transcript + all segments in ONE LLM call,
# #     ensuring context is preserved.
# #     """
# #     # Prepare prompt with segments + transcript
# #     segments_text = "\n".join(
# #         [f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}" for seg in result["segments"]]
# #     )

# #     full_prompt = f"""
# # You are a professional translator. 
# # Your task is to translate the given transcript into {dest_lang}.
# # Make sure to preserve the meaning and context across the entire transcript.
# # Return the output in **valid JSON format only**, with exactly two top-level fields:
# # 1. "segments" -> list of objects with "start", "end", "original_text", "translated_text".
# # 2. "full_translated_text" -> the full transcript translated with context.

# # DO NOT include any explanations or text outside of the JSON.

# # Here is the transcript with timestamps:

# # {segments_text}

# # Full transcript:
# # {result['text']}
# # """

# #     completion = client.chat.completions.create(
# #         model=model_id,  # using a valid supported model
# #         messages=[
# #             {"role": "system", "content": "You are a professional translator."},
# #             {"role": "user", "content": full_prompt}
# #         ],
# #         temperature=0.2,
# #         max_completion_tokens=3000,
# #         top_p=1,
# #     )

# #     response_text = completion.choices[0].message.content.strip()

# #     # Try to extract JSON substring
# #     match = re.search(r"\{.*\}", response_text, re.DOTALL)
# #     if not match:
# #         raise ValueError("LLM did not return valid JSON. Raw response:\n" + response_text)

# #     json_str = match.group(0)

# #     try:
# #         translated_output = json.loads(json_str)
# #     except json.JSONDecodeError as e:
# #         raise ValueError(f"Invalid JSON: {e}\nResponse:\n{response_text}")

# #     return translated_output


# # # Step 3: Full pipeline
# # def process_video_dubbing(audio_path: str, src_lang: str, dest_lang: str, model_id: str = "llama-3.3-70b-versatile"):
# #     result = speech_to_text(audio_path, src_lang)
# #     output = translate_segments_with_context(result, dest_lang, model_id=model_id)
# #     return output


# # if __name__ == "__main__":
# #     audio_file = r"L:\My projects\parinama\videoplayback (1) (1).wav"
# #     # Here pass the supported model
# #     output = process_video_dubbing(audio_file, "english", "tamil", model_id="llama-3.3-70b-versatile")

# #     with open("translated_output.json", "w", encoding="utf-8") as f:
# #         json.dump(output, f, ensure_ascii=False, indent=4)

# #     print("✅ Translation Done! Check translated_output.json")



















# # import os
# # import torch
# # import soundfile as sf
# # import numpy as np
# # import librosa
# # from pydub import AudioSegment
# # from parler_tts import ParlerTTSForConditionalGeneration
# # from transformers import AutoTokenizer
# # from pydub import AudioSegment
# # import numpy as np
# # import tempfile

# # from app.config import Config

# # # ---------- Load TTS model ----------
# # device = "cuda:0" if torch.cuda.is_available() else "cpu"
# # # model_name = "ai4bharat/indic-parler-tts"
# # model_mini = 'parler-tts/parler-tts-mini-v1.1'

# # model = ParlerTTSForConditionalGeneration.from_pretrained(model_mini).to(device)
# # tokenizer = AutoTokenizer.from_pretrained(model_mini)
# # description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# # def tts_thread(    
# #     text: str,
# #     speaker: str = None,
# #     emotion: str = None,
# #     pitch: str = "moderate",
# #     rate: str = "normal",
# #     expressivity: str = "neutral",
# #     background: str = "clear",
# #     reverberation: str = "close",
# #     accent: str = None,
# #     quality: str = "very high quality",
# #     ):
# #     """
# #         Generate speech from text using Indic Parler-TTS with full customization.

# #         Args:
# #             text (str): Input text to convert to speech.
# #             output_path (str): Path to save the output WAV file.
# #             speaker (str): Specific speaker (e.g., "Rohit", "Divya", "Aditi").
# #             emotion (str): Emotion (e.g., "Happy", "Sad", "Anger", "Neutral").
# #             pitch (str): "low", "high", "moderate".
# #             rate (str): "slow", "fast", "normal".
# #             expressivity (str): "monotone", "expressive", "slightly expressive".
# #             background (str): "clear", "slightly noisy".
# #             reverberation (str): "close", "distant".
# #             accent (str): Optionally specify accent (e.g., "Indian English", "British").
# #             quality (str): "basic", "refined", "very high quality".

# #         Returns:
# #             str: Path to saved audio file.
# #     """
    
# #     # Build description dynamically
# #     description_parts = []

# #     if speaker:
# #         description_parts.append(f"{speaker}'s voice")

# #     if emotion:
# #         description_parts.append(f"with a {emotion.lower()} tone")

# #     if rate:
# #         description_parts.append(f"speaks at a {rate} pace")

# #     if pitch:
# #         description_parts.append(f"with a {pitch} pitch")

# #     if expressivity:
# #         description_parts.append(f"in a {expressivity} style")

# #     if reverberation:
# #         description_parts.append(f"in a {reverberation}-sounding environment")

# #     if background:
# #         description_parts.append(f"with {background} background noise")

# #     if accent:
# #         description_parts.append(f"and an {accent} accent")

# #     if quality:
# #         description_parts.append(f"recorded in {quality} audio")

# #     description = ", ".join(description_parts)

# #     description_inputs = description_tokenizer(description, return_tensors="pt").to(device)
# #     prompt_inputs = tokenizer(text, return_tensors="pt").to(device)

# #     generation = model.generate(
# #         input_ids=description_inputs.input_ids,
# #         attention_mask=description_inputs.attention_mask,
# #         prompt_input_ids=prompt_inputs.input_ids,
# #         prompt_attention_mask=prompt_inputs.attention_mask,
# #     )

# #     audio_arr = generation.cpu().numpy().squeeze()
# #     return audio_arr, model.config.sampling_rate

# # def stretch_to_duration(audio_arr, sr, target_dur):
# #     """Stretch/compress audio to match target duration using pydub."""
# #     # Save numpy array temporarily
# #     temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
# #     sf.write(temp_in.name, audio_arr, sr)

# #     audio = AudioSegment.from_file(temp_in.name, format="wav")
# #     cur_dur = len(audio) / 1000.0
# #     if cur_dur <= 0:
# #         return audio_arr

# #     speed = cur_dur / target_dur

# #     # Change playback speed
# #     stretched = audio._spawn(audio.raw_data, overrides={
# #         "frame_rate": int(audio.frame_rate * speed)
# #     }).set_frame_rate(audio.frame_rate)

# #     # Convert back to numpy
# #     y = np.array(stretched.get_array_of_samples()).astype(np.float32) / (2**15)
# #     return y

# # def text_to_speech(data):
# #     out_path = os.path.join(Config.OUTPUT_FOLDER, "output_audio.wav")
# #     final = AudioSegment.silent(duration=0)

# #     for i, seg in enumerate(data):
# #         txt = seg["t_word"]
# #         start, end = seg["start"], seg["end"]
# #         target_dur = end - start

# #         # TTS generation
# #         audio_arr, sr = tts_thread(txt)

# #         # Stretch to match original duration
# #         y = stretch_to_duration(audio_arr, sr, target_dur)

# #         # Save temp wav
# #         temp_wav = os.path.join(Config.UPLOAD_FOLDER, f"seg_{i}.wav")
# #         sf.write(temp_wav, y, sr)

# #         # Add to final audio
# #         segment_audio = AudioSegment.from_wav(temp_wav)
# #         final += segment_audio

# #         # If there's a gap before next segment → insert silence
# #         if i < len(data) - 1:
# #             gap = data[i+1]["start"] - end
# #             if gap > 0:
# #                 final += AudioSegment.silent(duration=gap*1000)

# #         # Cleanup temp file
# #         os.remove(temp_wav)

# #         print(f"✅ {txt[:30]}... -> {target_dur:.2f}s")

# #     # Export combined file
# #     final.export(out_path, format="wav")
# #     print(f"🎵 Final synced audio saved: {out_path}")
# #     return out_path






# import os
# import soundfile as sf
# import numpy as np
# import tempfile
# from pydub import AudioSegment
# from gtts import gTTS 
# import librosa

# from app.config import Config

# # ---------- Supported Indic Languages ----------
# # gTTS language codes (supports many Indian languages)
# INDIC_MODELS = {
#     "hi": "hi",     # Hindi
#     "ta": "ta",     # Tamil
#     "te": "te",     # Telugu
#     "kn": "kn",     # Kannada
#     "ml": "ml",     # Malayalam
#     "bn": "bn",     # Bengali
#     "gu": "gu",     # Gujarati
#     "pa": "pa",     # Punjabi
#     "mr": "mr",     # Marathi
#     "or": "or",     # Odia
#     "as": "as"      # Assamese
# }

# def tts_thread(text: str, lang: str = "hi"):
#     """
#     Generate speech using gTTS (Google TTS).
#     Args:
#         text (str): Input text
#         lang (str): Language code (hi, ta, te, kn, etc.)
#     Returns:
#         np.ndarray: audio array
#         int: sampling rate
#     """
#     if lang not in INDIC_MODELS:
#         raise ValueError(f"❌ Unsupported language: {lang}")

#     # Save gTTS output to temporary mp3
#     temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts = gTTS(text=text, lang=INDIC_MODELS[lang])
#     tts.save(temp_out.name)

#     # Convert mp3 → wav → numpy
#     audio = AudioSegment.from_file(temp_out.name, format="mp3")
#     sr = audio.frame_rate
#     y = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)

#     os.remove(temp_out.name)
#     return y, sr

# import pyrubberband as pyrb

# def stretch_to_duration(audio_arr, sr, target_dur):
#     """
#     Stretch/compress audio to match target duration using pyrubberband (pitch-preserving).

#     Args:
#         audio_arr (np.ndarray): Input audio array (mono)
#         sr (int): Sampling rate
#         target_dur (float): Desired duration in seconds

#     Returns:
#         np.ndarray: Stretched audio array
#     """
#     cur_dur = len(audio_arr) / sr
#     if cur_dur <= 0 or target_dur <= 0:
#         return audio_arr

#     # Convert to mono if needed
#     if audio_arr.ndim > 1:
#         audio_arr = np.mean(audio_arr, axis=1)

#     # Compute stretch rate
#     rate = cur_dur / target_dur

#     # Stretch using rubberband (pitch-preserving)
#     y_stretched = pyrb.time_stretch(audio_arr, sr, rate)

#     # Ensure float32
#     y_stretched = y_stretched.astype(np.float32)
#     return y_stretched
# # def stretch_to_duration(audio_arr, sr, target_dur):
# #     """
# #     Stretch/compress audio to match target duration using pydub (safe for gTTS audio).
# #     """
# #     # Save numpy array to temp wav
# #     temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
# #     sf.write(temp_wav.name, audio_arr, sr)

# #     audio = AudioSegment.from_wav(temp_wav.name)
# #     cur_dur = len(audio) / 1000.0  # duration in seconds

# #     if cur_dur <= 0 or target_dur <= 0:
# #         os.remove(temp_wav.name)
# #         return audio_arr

# #     # Calculate speed factor
# #     speed = cur_dur / target_dur

# #     # Change playback speed
# #     stretched = audio._spawn(audio.raw_data, overrides={
# #         "frame_rate": int(audio.frame_rate * speed)
# #     }).set_frame_rate(audio.frame_rate)

# #     # Convert back to numpy array
# #     y = np.array(stretched.get_array_of_samples()).astype(np.float32) / (2**15)
# #     os.remove(temp_wav.name)
# #     return y

# def text_to_speech(data, lang="hi"):
#     """
#     Convert segmented text data to aligned audio file (gTTS-based).
#     Args:
#         data (list): list of dicts with {t_word, start, end}
#         lang (str): language code
#     Returns:
#         str: path to final wav file
#     """
#     if lang not in INDIC_MODELS:
#         raise ValueError(f"❌ Language {lang} not supported. Choose from {list(INDIC_MODELS.keys())}")

#     out_path = os.path.join(Config.OUTPUT_FOLDER, "output_audio.wav")
#     final = AudioSegment.silent(duration=0)

#     for i, seg in enumerate(data):
#         txt = seg["translated_text"]
#         start, end = seg["start"], seg["end"]
#         target_dur = end - start

#         # Generate audio
#         audio_arr, sr = tts_thread(txt, lang=lang)

#         # Stretch/compress to sync with original timing
#         y = stretch_to_duration(audio_arr, sr, target_dur)

#         temp_wav = os.path.join(Config.UPLOAD_FOLDER, f"seg_{i}.wav")
#         sf.write(temp_wav, y, sr)

#         segment_audio = AudioSegment.from_wav(temp_wav)
#         final += segment_audio

#         # Handle gap between segments
#         if i < len(data) - 1:
#             gap = data[i+1]["start"] - end
#             if gap > 0:
#                 final += AudioSegment.silent(duration=gap*1000)

#         os.remove(temp_wav)
#         print(f"✅ {txt[:30]}... ({lang}) -> {target_dur:.2f}s")

#     final.export(out_path, format="wav")
#     print(f"🎵 Final synced audio saved: {out_path}")
#     return out_path



# result = [{'start': 0.0, 'end': 6.8, 'original_text': 'Welcome to English in a minute.', 'translated_text': 'ஆங்கிலம் ஒரு நிமிடத்திற்கு வரவேற்றுக்கொள்கிறோம்.'}, {'start': 6.8, 'end': 11.38, 'original_text': "If you've ever gardened, you know weeds grow super fast.", 'translated_text': 'நீங்கள் ஒருபோதும் தோட்டம் செய்தால், வீட்டுக்கள் மிகவும் வேகமாக வளரும் என்பதை நீங்கள் அறிவீர்கள்.'}]
# text_to_speech(result, lang="ta")

from app.config import Config
import os
from app.services.embedder import embed

audio_path = '/home/praveen/skcet/SIH2025/parinama/app/outputs/output_audio.wav'
video_path = os.path.join(Config.UPLOAD_FOLDER, "input_video.mp4")
output_path = os.path.join(Config.OUTPUT_FOLDER, "output_video.mp4")
print(audio_path, video_path, output_path)
embed(audio_path, video_path, output_path)