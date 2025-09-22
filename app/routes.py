from server import app
from flask import request, jsonify, send_file
import traceback
import os

from app.config import Config
from app.services.dubbing import video_dubbing
from app.services.stt import speech_to_text
from app.services.tts import text_to_speech

@app.route('/video_dubbing', methods=['GET','POST'])
def video_dubbing_route():
    try:
        video_dubbing(request)
        
        output_view_path = os.path.join(Config.OUTPUT_FOLDER, "output_video.mp4")
        return send_file(output_view_path, as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/audio_dubbing', methods=['POST'])
def audio_dubbing_route():
    try:
        print("Entering into audio_dubbing process!!!")

        # If JSON was sent
        if request.is_json:
            data = request.get_json()
            src_lang = data.get("src_lang")
            dest_lang = data.get("dest_lang")
            model_id = data.get("model_id", "llama-3.3-70b-versatile")

        # If form-data was sent
        else:
            src_lang = request.form.get("src_lang")
            dest_lang = request.form.get("dest_lang")
            model_id = request.form.get("model_id", "llama-3.3-70b-versatile")

            # Save uploaded file if present
            if "file" in request.files:
                audio_file = request.files["file"]
                audio_path = os.path.join(Config.UPLOAD_FOLDER, "input_audio.wav")
                audio_file.save(audio_path)

        if not src_lang or not dest_lang:
            return jsonify({"error": "src_lang and dest_lang are required"}), 400

        output_segments = speech_to_text(src_lang, dest_lang, model_id)

        return jsonify({"text": output_segments['full_translated_text']}), 200

    except Exception as e:
        print("Lead to exception in audio_dubbing_route")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech_route():
    try:
        print("🎤 Entering into text_to_speech process!!!")

        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415

        data = request.get_json()
        text = data.get("text")
        lang = data.get("language", "en")
        voice_type = data.get("voice_type", "female")
        speed = float(data.get("speed", 1.0))
        pitch = float(data.get("pitch", 1.0))

        if not text:
            return jsonify({"error": "Text is required"}), 400

        # 🔹 Wrap text into one segment so it works with your existing function
        segments = [{"translated_text": text, "start": 0, "end": len(text.split()) / 2}]  
        # approx duration = words / 2 seconds (adjust later if needed)

        out_path = text_to_speech(segments, lang)

        print(f"✅ Generated speech saved at {out_path}")

        return send_file(out_path, as_attachment=True, download_name="generated.wav")

    except Exception as e:
        print("❌ Exception in text_to_speech_route")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500