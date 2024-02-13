import glob
import os
import pickle
import sys
import time

import face_recognition
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm_progress
from video_diarization import util_components


def get_track_face_encodings(tracks, scores, pyframesPath, pyworkPath):
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Starting generation of face encodings"
    )

    flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))
    flist.sort()

    faces = {}
    # Get all the frames and faces where the speaker is speaking,i.e. s>0
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[
                max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
            ]  # average smoothing
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

    progress_bar = tqdm_progress(
        total=total_frame_tracks, unit="faces", dynamic_ncols=True
    )
    for fidx, frame_id in enumerate(faces.keys()):
        image = face_recognition.load_image_file(flist[frame_id])
        for track_id in faces[frame_id].keys():
            faces[frame_id][track_id]["encoding"] = face_recognition.face_encodings(
                image,
                [
                    (
                        int(
                            faces[frame_id][track_id]["y"]
                            - faces[frame_id][track_id]["s"]
                        ),
                        int(
                            faces[frame_id][track_id]["x"]
                            + faces[frame_id][track_id]["s"]
                        ),
                        int(
                            faces[frame_id][track_id]["y"]
                            + faces[frame_id][track_id]["s"]
                        ),
                        int(
                            faces[frame_id][track_id]["x"]
                            - faces[frame_id][track_id]["s"]
                        ),
                    )
                ],
                model="small",
            )[0]
            progress_bar.update(1)
    progress_bar.close()
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Face encoding generation completed"
    )

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
    df["Clusters"] = util_components.perform_clustering(df["Encoding"].to_list())

    savePath = os.path.join(pyworkPath, "encoding_df.pckl")
    df.to_pickle(savePath)

    return df


def get_final_tracks(pyworkPath, frameRate):
    final_tracks = {}
    df = pd.read_pickle(os.path.join(pyworkPath, "encoding_df.pckl"))
    for idx in df["Clusters"].unique():
        speaker_key = "SPEAKER_{:02d}".format(idx)
        final_tracks[speaker_key] = df[df["Clusters"] == idx]["Frame"].to_list()

    for key in final_tracks.keys():
        final_tracks[key] = util_components.convert_to_ranges(
            final_tracks[key], frameRate
        )

    with open(os.path.join(pyworkPath, "video_diarization_output.pckl"), "wb") as fil:
        pickle.dump(final_tracks, fil)

    return final_tracks
