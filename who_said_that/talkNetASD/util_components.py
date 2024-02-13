import glob
import math
import os
import subprocess

import cv2
import numpy as np
import python_speech_features
import torch
import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile


def track_shot(sceneFaces, numFailedDet=10, minFaceSize=1, minTrack=10):
    # CPU: Face tracking
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= numFailedDet:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > minTrack:
            frameNum = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])
            frameI = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if (
                max(
                    np.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                    np.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                )
                > minFaceSize
            ):
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def crop_video(
    track,
    cropFile,
    pyframesPath,
    audioFilePath,
    nDataLoaderThread=10,
    cropScale=0.40,
):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )  # Write video
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:  # Read the tracks
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = cropScale
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (audioFilePath, nDataLoaderThread, audioStart, audioEnd, audioTmp)
    )
    output = subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = (
        "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
        % (
            cropFile,
            audioTmp,
            nDataLoaderThread,
            cropFile,
        )
    )  # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")
    return {"track": track, "proc_track": dets}


def evaluate_network(files, talkNetModel, pycropPath):
    # GPU: active speaker detection by pretrained TalkNet
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        4,
        5,
        6,
    }  # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split("/")[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(pycropPath, fileName + ".wav"))
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )
        video = cv2.VideoCapture(os.path.join(pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = np.array(videoFeature)
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25,
        )
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]
        allScore = []  # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = (
                        torch.FloatTensor(
                            audioFeature[
                                i * duration * 100 : (i + 1) * duration * 100, :
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )
                    inputV = (
                        torch.FloatTensor(
                            videoFeature[
                                i * duration * 25 : (i + 1) * duration * 25,
                                :,
                                :,
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )
                    embedA = talkNetModel.model.forward_audio_frontend(inputA)
                    embedV = talkNetModel.model.forward_visual_frontend(inputV)
                    embedA, embedV = talkNetModel.model.forward_cross_attention(
                        embedA, embedV
                    )
                    out = talkNetModel.model.forward_audio_visual_backend(
                        embedA, embedV
                    )
                    score = talkNetModel.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = np.round((np.mean(np.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores
