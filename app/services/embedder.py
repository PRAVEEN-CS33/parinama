from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import tempfile
import os

def embed(audio_path, video_path, output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # Check durations
    video_dur = video_clip.duration
    audio_dur = audio_clip.duration
    print(f"Video duration: {video_dur}s, Audio duration: {audio_dur}s")

    # Pad audio if shorter than video
    if audio_dur < video_dur:
        silence_dur = video_dur - audio_dur
        temp_audio = AudioSegment.from_file(audio_path)
        silence = AudioSegment.silent(duration=int(silence_dur * 1000))
        padded_audio = temp_audio + silence

        # Export padded audio to temp wav
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        padded_audio.export(temp_file.name, format="wav")
        audio_clip.close()
        audio_clip = AudioFileClip(temp_file.name)
        temp_file_path = temp_file.name
    else:
        temp_file_path = None

    # Attach audio
    video_with_audio = video_clip.with_audio(audio_clip)

    # Write final video
    video_with_audio.write_videofile(
        output_path,
        # codec="libx264",
        # audio_codec="aac",
    )

    # Cleanup
    video_clip.close()
    audio_clip.close()
    if temp_file_path:
        os.remove(temp_file_path)

    print(f"Final video with audio saved at: {output_path}")
    return output_path
