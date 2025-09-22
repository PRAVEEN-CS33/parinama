from flask import jsonify
import json
import os

from app.config import Config
from app.services.splitter import video_audio_splitter
from app.services.stt import speech_to_text
from app.services.tts import text_to_speech
from app.services.embedder import embed


def video_dubbing(request):
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']

    if(video_file.filename):
        path = Config.OUTPUT_FOLDER + "/data.json"
                # Ensure outputs folder exists
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

        # If data.json does not exist, create it with empty dict
        if not os.path.exists(path):
            with open(path, 'w') as json_file:
                json.dump({}, json_file)

        with open(path, 'r') as json_file:
            data = json.load(json_file)

        data['name'] = video_file.filename

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    

    video_audio_splitter(video_file)

    src = request.form['src']
    dest = request.form['dest']
    voice = request.form['voice']
    print(src, dest, voice)
    result = speech_to_text(src, dest)

    text_to_speech(result, dest)

    audio_path = os.path.join(Config.OUTPUT_FOLDER, "output_audio.wav")
    video_path = os.path.join(Config.UPLOAD_FOLDER, "input_video.mp4")
    output_path = os.path.join(Config.OUTPUT_FOLDER, "output_video.mp4")
    print(audio_path, video_path, output_path)
    embed(audio_path, video_path, output_path)



