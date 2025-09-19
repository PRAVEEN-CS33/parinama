import assemblyai as aai
import whisper
import torch
import json
import os
from dotenv import load_dotenv
load_dotenv()

from config import Config
from app.services.consts import WHISPER_LANGUAGES as LANG
from app.services.translate import translate_text

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
transcriber = aai.Transcriber()

def speech_to_text(src, dest):
    path = Config.OUTPUT_FOLDER +"/input_audio.wav"
    print("path", path)
    src = LANG.get(src)
    model = whisper.load_model("base")
    result = model.transcribe(path, language=src, fp16=False, verbose=True)
    print("result done")
    
    result_data = []
    result_data1 = []
    for i, seg in enumerate(result['segments']):
        # save each segment as a separate JSON file
        tw = translate_text(seg['text'], src, dest)
        entry = {
            'start': seg['start'],
            'end': seg['end'],
            'word': seg['text'],
            'tword':tw,
        }
        result_data.append(entry)
        print(entry)
        
        tw1 = translate_text(seg['text'], src, dest)
        entry1 = {
                'start': seg['start'],
                'end': seg['end'],
                'word':tw1,
        }
        result_data1.append(entry1)
        print(entry1)

    with open('result1.json', 'w') as json_file:
        json.dump(result_data, json_file, indent=2)
        
    with open('result2.json', 'w') as json_file:
        json.dump(result_data1, json_file, indent=2) 
          
    return result_data