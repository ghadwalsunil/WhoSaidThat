import glob
import os
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np
import tqdm
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scenedetect import detect, FrameTimecode

from who_said_that.talkNetASD import util_components
from who_said_that.utils.model.faceDetector import S3FD


def scene_detect(videoFilePath: str, pyworkPath: str, pyframesPath: str):
    # CPU: Scene detection, output is the list of each shot's time duration
    # videoManager = VideoManager([videoFilePath])
    # statsManager = StatsManager()
    # sceneManager = SceneManager(statsManager)
    # sceneManager.add_detector(AdaptiveDetector())
    # baseTimecode = videoManager.get_base_timecode()
    # videoManager.set_downscale_factor()
    # videoManager.start()
    # sceneManager.detect_scenes(video=videoManager)
    # sceneList = sceneManager.get_scene_list(baseTimecode)
    # if sceneList == []:
    #     sceneList = [
    #         (
    #             videoManager.get_base_timecode(),
    #             videoManager.get_current_timecode(),
    #         )
    #     ]
    sceneList = detect(videoFilePath, AdaptiveDetector(), show_progress=True, start_in_scene=True)
    if sceneList == []:
        flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))
        flist.sort()
        sceneList = [
            (
                FrameTimecode(0, 25),
                FrameTimecode(int(flist[-1]), 25),
            )
        ]

    savePath = os.path.join(pyworkPath, "scene.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(sceneList, fil)

    sys.stderr.write("%s - scenes detected %d\n" % (videoFilePath, len(sceneList)))
    return sceneList


def inference_video(
    videoFilePath: str,
    pyframesPath: str,
    pyworkPath: str,
    facedetScale: float = 0.25,
):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device="cuda")
    flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[facedetScale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
            )  # dets has the frames info, bbox info, conf info
        sys.stderr.write("%s-%05d; %d dets\r" % (videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(dets, fil)
    return dets


def track_faces(pyworkPath, minTrack=10):
    faces = pickle.load(open(os.path.join(pyworkPath, "faces.pckl"), "rb"))
    scene = pickle.load(open(os.path.join(pyworkPath, "scene.pckl"), "rb"))
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= minTrack:  # Discard the shot frames less than minTrack frames
            allTracks.extend(
                util_components.track_shot(faces[shot[0].frame_num : shot[1].frame_num])
            )  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces

    with open(os.path.join(pyworkPath, "allTracks.pckl"), "wb") as fil:
        pickle.dump(allTracks, fil)

    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % len(allTracks))


def crop_face_clips(pyworkPath, pycropPath, pyframesPath, audioFilePath):
    vidTracks = []
    allTracks = pickle.load(open(os.path.join(pyworkPath, "allTracks.pckl"), "rb"))
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(
            util_components.crop_video(
                track=track,
                cropFile=os.path.join(pycropPath, "%05d" % ii),
                pyframesPath=pyframesPath,
                audioFilePath=audioFilePath,
            )
        )
    savePath = os.path.join(pyworkPath, "tracks.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" % pycropPath)


def talknet_speaker_detection(pycropPath, pyworkPath, talkNetModel):
    files = glob.glob("%s/*.avi" % pycropPath)
    files.sort()
    scores = util_components.evaluate_network(files=files, pycropPath=pycropPath, talkNetModel=talkNetModel)
    savePath = os.path.join(pyworkPath, "scores.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" % pyworkPath)


def visualization(pyframesPath, pyaviPath, pyworkPath, pywavPath, nDataLoaderThread=10, saveMarkedFrames=False):
    # CPU: visulize the result for video format
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Saving marked video \r\n")
    tracks = pickle.load(open(os.path.join(pyworkPath, "tracks.pckl"), "rb"))
    scores = pickle.load(open(os.path.join(pyworkPath, "scores.pckl"), "rb"))
    flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    if saveMarkedFrames:
        markedFramesPath = os.path.join(pyaviPath, "marked")
        if not os.path.exists(markedFramesPath):
            os.makedirs(markedFramesPath)
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]  # average smoothing
            s = np.mean(s)
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(pyaviPath, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
            if saveMarkedFrames:
                cv2.imwrite(
                    os.path.join(markedFramesPath, "asd_%05d.jpg" % fidx),
                    image,
                )
        vOut.write(image)
    vOut.release()
    command = "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % (
        os.path.join(pyaviPath, "video_only.avi"),
        os.path.join(pywavPath, "audio.wav"),
        nDataLoaderThread,
        os.path.join(pyaviPath, "video_out.avi"),
    )
    output = subprocess.call(command, shell=True, stdout=None)
