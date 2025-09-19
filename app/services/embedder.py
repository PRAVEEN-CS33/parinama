from moviepy import VideoFileClip, AudioFileClip

def embed(audio_path, video_path, output_path=None):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path).with_duration(video_clip.duration)

    # Attach audio
    video_with_audio = video_clip.with_audio(audio_clip)
    print(video_clip.duration)
    print(audio_clip.duration)

    # Write video with audio
    video_with_audio.write_videofile(
        output_path,
        # codec="libx264",
        # audio_codec="aac",
    )

    print(f"Final video with audio saved at: {output_path}")

    # close resources
    video_clip.close()
    audio_clip.close()
    return output_path