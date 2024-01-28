from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

video_name = "StarTalk_Cosmic.mp4"

start_time = 440  # Start time in seconds (e.g., 1 minute 30 seconds)
end_time = 532  # End time in seconds (e.g., 2 minutes 30 seconds)

input_video = "../../Videos/" + video_name
output_video = f"../../Clips/{video_name.split('.')[0]}_{str(start_time)}_{str(end_time)}.{video_name.split('.')[-1]}"

ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_video)
