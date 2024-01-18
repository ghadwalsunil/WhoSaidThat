import sys, time, os, tqdm, glob, subprocess, warnings, pickle

from shutil import rmtree
import face_recognition
from scenedetect.platform import tqdm as tqdm_progress
import numpy as np
import pandas as pd
import face_recognition
from pydub import AudioSegment
from utils.components import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def convert_to_ranges(lst, frame_rate):
    ranges = []
    start = lst[0]

    threshold = round(frame_rate / 10) * 3

    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] > threshold:
            if lst[i - 1] - start > threshold:
                ranges.append((round(start / frame_rate, 2), round(lst[i - 1] / frame_rate, 2)))
            start = lst[i]

    # Add the last range
    if lst[-1] - start > threshold:
        ranges.append((round(start / frame_rate, 2), round(lst[-1] / frame_rate, 2)))

    return ranges


def get_track_face_encodings(tracks, scores, args):
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Starting generation of face encodings")
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))

    flist.sort()

    video_duration = len(AudioSegment.from_file(os.path.join(args.savePath, "pyavi", "audio.wav")))
    args.frameRate = len(flist) / (video_duration / 1000)

    faces = {}
    # Get all the frames and faces where the speaker is speaking,i.e. s>0
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]  # average smoothing
            s = float(np.mean(s))
            if s > 0:
                track_data = {
                    "score": s,
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                    "frame": frame,
                }
                if frame in faces.keys():
                    faces[frame][tidx] = track_data
                else:
                    faces[frame] = {tidx: track_data}

    # Generate face encodings for all the selected speaking faces
    total_frame_tracks = 0
    for frame_id in faces.keys():
        for track_id in faces[frame_id].keys():
            total_frame_tracks += 1

    progress_bar = tqdm_progress(total=total_frame_tracks, unit="faces", dynamic_ncols=True)
    for fidx, frame_id in enumerate(faces.keys()):
        image = face_recognition.load_image_file(flist[frame_id])
        for track_id in faces[frame_id].keys():
            faces[frame_id][track_id]["encoding"] = face_recognition.face_encodings(
                image,
                [
                    (
                        int(faces[frame_id][track_id]["y"] - faces[frame_id][track_id]["s"]),
                        int(faces[frame_id][track_id]["x"] + faces[frame_id][track_id]["s"]),
                        int(faces[frame_id][track_id]["y"] + faces[frame_id][track_id]["s"]),
                        int(faces[frame_id][track_id]["x"] - faces[frame_id][track_id]["s"]),
                    )
                ],
                model="small",
            )[0]
            progress_bar.update(1)
    progress_bar.close()
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face encoding generation completed")

    rows = []
    for frame, tracks in faces.items():
        for track_id, track_data in tracks.items():
            rows.append(
                {
                    "Frame": frame,
                    "Track": track_id,
                    "Score": track_data["score"],
                    "S": track_data["s"],
                    "X": track_data["x"],
                    "Y": track_data["y"],
                    "Encoding": track_data["encoding"],
                }
            )

    df = pd.DataFrame(rows)
    savePath = os.path.join(args.pyworkPath, "encoding_df.pckl")
    df.to_pickle(savePath)

    return df, args


def get_final_tracks(df, args):
    final_tracks = {}
    for idx in df["Clusters"].unique():
        speaker_key = "SPEAKER_{:02d}".format(idx)
        final_tracks[speaker_key] = df[df["Clusters"] == idx]["Frame"].to_list()

    for key in final_tracks.keys():
        final_tracks[key] = convert_to_ranges(final_tracks[key], args.frameRate)

    return final_tracks


def preprocess(args):
    # Initialization
    args.pyaviPath = os.path.join(args.savePath, "pyavi")
    args.pyframesPath = os.path.join(args.savePath, "pyframes")
    args.pyworkPath = os.path.join(args.savePath, "pywork")
    args.pycropPath = os.path.join(args.savePath, "pycrop")
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok=True)  # The path for the input video, input audio, output video
    os.makedirs(args.pyframesPath, exist_ok=True)  # Save all the video frames
    os.makedirs(args.pyworkPath, exist_ok=True)  # Save the results in this process by the pckl method
    os.makedirs(args.pycropPath, exist_ok=True)  # Save the detected face clips (audio+video) in this process

    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, "video.avi")
    # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
    if args.duration == 0:
        command = "ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % (
            args.videoPath,
            args.nDataLoaderThread,
            args.videoFilePath,
        )
    else:
        command = "ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % (
            args.videoPath,
            args.nDataLoaderThread,
            args.start,
            args.start + args.duration,
            args.videoFilePath,
        )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" % (args.videoFilePath)
    )

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, "audio.wav")
    command = "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % (
        args.videoFilePath,
        args.nDataLoaderThread,
        args.audioFilePath,
    )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" % (args.audioFilePath)
    )

    # Extract the video frames
    command = "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % (
        args.videoFilePath,
        args.nDataLoaderThread,
        os.path.join(args.pyframesPath, "%06d.jpg"),
    )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" % (args.pyframesPath)
    )

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" % (args.pyworkPath))

    # Face detection for the video frames
    faces = inference_video(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" % (args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:  # Discard the shot frames less than minTrack frames
            allTracks.extend(
                track_shot(args, faces[shot[0].frame_num : shot[1].frame_num])
            )  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % len(allTracks))

    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, "%05d" % ii)))
    savePath = os.path.join(args.pyworkPath, "tracks.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" % args.pycropPath)
    fil = open(savePath, "rb")
    vidTracks = pickle.load(fil)

    # Active Speaker Detection by TalkNet
    files = glob.glob("%s/*.avi" % args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, "scores.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" % args.pyworkPath)

    # Visualization, save the result as the new video
    # visualization(vidTracks, scores, args)
    return vidTracks, scores, args
