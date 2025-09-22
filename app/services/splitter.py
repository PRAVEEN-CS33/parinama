from moviepy import VideoFileClip
import tempfile
import os

from app.config import Config

def video_audio_splitter(video_file):
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_file_path = temp_video_file.name
     
    # Extract audio and save as WAV 
    video_clip = VideoFileClip(temp_video_file_path)
    audio_clip = video_clip.audio
    audio_file_path = os.path.join(Config.UPLOAD_FOLDER, "input_audio.wav")
    audio_clip.write_audiofile(audio_file_path, codec='pcm_s16le', ffmpeg_params=["-ac", "2"])

    # Save video without audio
    clip = video_clip.with_audio(None)
    video_file_path = os.path.join(Config.UPLOAD_FOLDER, "input_video.mp4")
    clip.write_videofile(video_file_path, codec="libx264", audio_codec="aac")

    # Clean up temporary files
    video_clip.close()
    audio_clip.close()
    os.remove(temp_video_file_path)
