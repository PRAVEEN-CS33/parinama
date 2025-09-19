from server import app
from flask import request, jsonify, send_file
import traceback

from app.services.dubbing import video_dubbing

@app.route('/video_dubbing', methods=['GET','POST'])
def video_dubbing_route():
    try:
        video_dubbing(request)
        return send_file("output_video.mp4", as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
