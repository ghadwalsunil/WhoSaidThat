from moviepy.editor import VideoFileClip
import os

# Define the input video file and output audio file
video_dir = "Dataset"
video_files = [
    f
    for f in os.listdir(video_dir)
    if os.path.isfile(os.path.join(video_dir, f)) and os.path.splitext(os.path.join(video_dir, f))[1] in [".mp4"]
]

for f in video_files:
    # Load the video clip
    video_clip = VideoFileClip(os.path.join(video_dir, f))

    # Extract the audio from the video clip
    audio_clip = video_clip.audio

    # Write the audio to a separate file
    audio_clip.write_audiofile(os.path.join(video_dir, os.path.splitext(f)[0] + ".mp3"))

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()

    print(f)
