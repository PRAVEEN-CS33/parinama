import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Load model and tokenizers only once (global initialization)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)


def text_to_speech(
    text: str,
    output_path: str = "indic_tts_out.wav",
    speaker: str = None,
    emotion: str = None,
    pitch: str = "moderate",
    rate: str = "normal",
    expressivity: str = "neutral",
    background: str = "clear",
    reverberation: str = "close",
    accent: str = None,
    quality: str = "very high quality",
):
    """
    Generate speech from text using Indic Parler-TTS with full customization.

    Args:
        text (str): Input text to convert to speech.
        output_path (str): Path to save the output WAV file.
        speaker (str): Specific speaker (e.g., "Rohit", "Divya", "Aditi").
        emotion (str): Emotion (e.g., "Happy", "Sad", "Anger", "Neutral").
        pitch (str): "low", "high", "moderate".
        rate (str): "slow", "fast", "normal".
        expressivity (str): "monotone", "expressive", "slightly expressive".
        background (str): "clear", "slightly noisy".
        reverberation (str): "close", "distant".
        accent (str): Optionally specify accent (e.g., "Indian English", "British").
        quality (str): "basic", "refined", "very high quality".

    Returns:
        str: Path to saved audio file.
    """

    # Build description dynamically
    description_parts = []

    if speaker:
        description_parts.append(f"{speaker}'s voice")

    if emotion:
        description_parts.append(f"with a {emotion.lower()} tone")

    if rate:
        description_parts.append(f"speaks at a {rate} pace")

    if pitch:
        description_parts.append(f"with a {pitch} pitch")

    if expressivity:
        description_parts.append(f"in a {expressivity} style")

    if reverberation:
        description_parts.append(f"in a {reverberation}-sounding environment")

    if background:
        description_parts.append(f"with {background} background noise")

    if accent:
        description_parts.append(f"and an {accent} accent")

    if quality:
        description_parts.append(f"recorded in {quality} audio")

    description = ", ".join(description_parts)

    # Tokenize inputs
    description_inputs = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate audio
    generation = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
    )

    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_path, audio_arr, model.config.sampling_rate)

    print(f"✅ Audio saved at: {output_path}")
    return output_path
