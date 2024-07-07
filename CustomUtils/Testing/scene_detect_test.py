from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

scene_list = detect("video.avi", AdaptiveDetector(), show_progress=True, start_in_scene=True)
split_video_ffmpeg("video.avi", scene_list, "detected_scenes")
