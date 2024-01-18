import sys, os, tqdm, torch, glob, subprocess, cv2, pickle, numpy, math, python_speech_features

from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from model.faceDetector.s3fd import S3FD
from talkNet import talkNet


def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(video=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, "scene.pckl")
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
    with open(savePath, "wb") as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write("%s - scenes detected %d\n" % (args.videoFilePath, len(sceneList)))
    return sceneList


def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device="cuda")
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
            )  # dets has the frames info, bbox info, conf info
        sys.stderr.write("%s-%05d; %d dets\r" % (args.videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(args.pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(dets, fil)
    return dets


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


def track_shot(args, sceneFaces):
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
                elif face["frame"] - track[-1]["frame"] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f["frame"] for f in track])
            bboxes = numpy.array([numpy.array(f["bbox"]) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if (
                max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]), numpy.mean(bboxesI[:, 3] - bboxesI[:, 1]))
                > args.minFaceSize
            ):
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks


def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224))  # Write video
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:  # Read the tracks
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = args.cropScale
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)
    )
    output = subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % (
        cropFile,
        audioTmp,
        args.nDataLoaderThread,
        cropFile,
    )  # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")
    return {"track": track, "proc_track": dets}


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)  # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split("/")[-1].replace(".wav", ".npy"))
    numpy.save(featuresPath, mfcc)


def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.pretrainModel)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}  # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split("/")[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + ".wav"))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)) : int(112 + (112 / 2)), int(112 - (112 / 2)) : int(112 + (112 / 2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]
        allScore = []  # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = (
                        torch.FloatTensor(audioFeature[i * duration * 100 : (i + 1) * duration * 100, :])
                        .unsqueeze(0)
                        .cuda()
                    )
                    inputV = (
                        torch.FloatTensor(videoFeature[i * duration * 25 : (i + 1) * duration * 25, :, :])
                        .unsqueeze(0)
                        .cuda()
                    )
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]  # average smoothing
            s = numpy.mean(s)
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
        os.path.join(args.pyaviPath, "video_only.avi"), cv2.VideoWriter_fourcc(*"XVID"), 25, (fw, fh)
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
        vOut.write(image)
    vOut.release()
    command = "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % (
        os.path.join(args.pyaviPath, "video_only.avi"),
        os.path.join(args.pyaviPath, "audio.wav"),
        args.nDataLoaderThread,
        os.path.join(args.pyaviPath, "video_out.avi"),
    )
    output = subprocess.call(command, shell=True, stdout=None)


def evaluate_col_ASD(tracks, scores, args):
    txtPath = args.videoFolder + "/col_labels/fusion/*.txt"  # Load labels
    predictionSet = {}
    for name in {"long", "bell", "boll", "lieb", "sick", "abbas"}:
        predictionSet[name] = [[], []]
    dictGT = {}
    txtFiles = glob.glob("%s" % txtPath)
    for file in txtFiles:
        lines = open(file).read().splitlines()
        idName = file.split("/")[-1][:-4]
        for line in lines:
            data = line.split("\t")
            frame = int(int(data[0]) / 29.97 * 25)
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[1]) + int(data[3])
            y2 = int(data[2]) + int(data[3])
            gt = int(data[4])
            if frame in dictGT:
                dictGT[frame].append([x1, y1, x2, y2, gt, idName])
            else:
                dictGT[frame] = [[x1, y1, x2, y2, gt, idName]]
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))  # Load files
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = numpy.mean(score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)])  # average smoothing
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        if fidx in dictGT:  # This frame has label
            for gtThisFrame in dictGT[fidx]:  # What this label is ?
                faceGT = gtThisFrame[0:4]
                labelGT = gtThisFrame[4]
                idGT = gtThisFrame[5]
                ious = []
                for face in faces[fidx]:  # Find the right face in my result
                    faceLocation = [
                        int(face["x"] - face["s"]),
                        int(face["y"] - face["s"]),
                        int(face["x"] + face["s"]),
                        int(face["y"] + face["s"]),
                    ]
                    faceLocation_new = [
                        int(face["x"] - face["s"]) // 2,
                        int(face["y"] - face["s"]) // 2,
                        int(face["x"] + face["s"]) // 2,
                        int(face["y"] + face["s"]) // 2,
                    ]
                    iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol=True)
                    if iou > 0.5:
                        ious.append([iou, round(face["score"], 2)])
                if len(ious) > 0:  # Find my result
                    ious.sort()
                    labelPredict = ious[-1][1]
                else:
                    labelPredict = 0
                x1 = faceGT[0]
                y1 = faceGT[1]
                width = faceGT[2] - faceGT[0]
                predictionSet[idGT][0].append(labelPredict)
                predictionSet[idGT][1].append(labelGT)
    names = ["long", "bell", "boll", "lieb", "sick", "abbas"]  # Evaluate
    names.sort()
    F1s = 0
    for i in names:
        scores = numpy.array(predictionSet[i][0])
        labels = numpy.array(predictionSet[i][1])
        scores = numpy.int64(scores > 0)
        F1 = f1_score(labels, scores)
        ACC = accuracy_score(labels, scores)
        if i != "abbas":
            F1s += F1
            print("%s, ACC:%.2f, F1:%.2f" % (i, 100 * ACC, 100 * F1))
    print("Average F1:%.2f" % (100 * (F1s / 5)))
