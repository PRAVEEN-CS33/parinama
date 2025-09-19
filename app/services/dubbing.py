from flask import jsonify
import json
import os

from app.services.splitter import video_audio_splitter
from app.services.stt import speech_to_text
from app.services.tts import text_to_speech
from app.services.embedder import embed


def video_dubbing(request):
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']

    if(video_file.filename):
        path = "assets/data.json"
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        data['name'] = video_file.filename
        with open("assets/data.json", 'w') as json_file:
            json.dump(data, json_file, indent=4)
    
    src = request.form['src']
    dest = request.form['dest']
    voice = request.form['voice']
    print(src, dest, voice)

    video_audio_splitter(video_file)
    speech_to_text(src, dest)
    text_to_speech(dest, voice)
    embed("output_audio.wav", "input_video.mp4", "output_video.mp4")



