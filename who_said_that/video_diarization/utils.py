import glob
import os
import pickle
import sys
import time

import face_recognition
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm_progress

from who_said_that.video_diarization import util_components


def get_track_face_encodings(tracks, scores, pyframesPath, pyworkPath):

    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Starting generation of face encodings \r\n"
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

    # df.sort_values(by=["Track", "Frame"], inplace=True)

    # df = util_components.mark_frames(df)

    # tqdm_progress.pandas()
    # df["Encoding"] = df.progress_apply(
    #     lambda row: (
    #         util_components.generate_encodings(flist[row["Frame"]], row["S"], row["X"], row["Y"])
    #         if row["Marked"]
    #         else None
    #     ),
    #     axis=1,
    # )

    savePath = os.path.join(pyworkPath, "encoding_df.pckl")
    df.to_pickle(savePath)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Face encoding generation completed \r\n"
    )

    return df


# def perform_clustering(pyworkPath, simple_clustering=True, enhanced_clustering=True):
#     df: pd.DataFrame = pd.read_pickle(os.path.join(pyworkPath, "encoding_df.pckl"))
#     temp_df = df[df["Marked"]].copy()
#     if simple_clustering:
#         temp_df["InitialClusters"] = util_components.perform_clustering(
#             temp_df["Encoding"].to_list(), max_clusters=6, silhouette_threshold=0.2
#         )
#         track_clusters = {}
#         # Get the cluster for each track based on majority
#         for track in temp_df["Track"].unique():
#             track_clusters[track] = temp_df[temp_df["Track"] == track]["InitialClusters"].mode().values[0]

#         temp_df["SimpleClusters"] = temp_df["Track"].apply(lambda x: track_clusters[x])

#         df = df.merge(temp_df[["Track", "SimpleClusters"]].drop_duplicates(), on="Track")

#     if enhanced_clustering:
#         final_df_cents, track_centroids_df = util_components.get_subset(temp_df)
#         track_centroids_df["Track_Cluster_Cluster_ID"] = util_components.perform_clustering(
#             list(track_centroids_df["Track_Cluster_Centroid"]), max_clusters=12
#         )

#         final_df_cents = final_df_cents.merge(
#             track_centroids_df[["Track", "Track_Cluster_ID", "Track_Cluster_Cluster_ID"]],
#             on=["Track", "Track_Cluster_ID"],
#         )

#         final_df_cents = util_components.get_final_cluster_ids(final_df_cents)

#         df = df.merge(final_df_cents[["Track", "EnhancedClusters"]].drop_duplicates(), on="Track")

#     savePath = os.path.join(pyworkPath, "encoding_df_w_clustering.pckl")
#     df.to_pickle(savePath)

#     return df


def perform_clustering(pyworkPath, simple_clustering=True, enhanced_clustering=True):
    df: pd.DataFrame = pd.read_pickle(os.path.join(pyworkPath, "encoding_df.pckl"))
    if simple_clustering:
        df["SimpleClusters"] = util_components.perform_clustering(
            df["Encoding"].to_list(), max_clusters=6, silhouette_threshold=0
        )

    if enhanced_clustering:
        final_df_cents, track_centroids_df = util_components.get_subset(df)
        track_centroids_df["Track_Cluster_Cluster_ID"] = (
            util_components.perform_clustering(
                list(track_centroids_df["Track_Cluster_Centroid"]), max_clusters=12
            )
        )

        final_df_cents = final_df_cents.merge(
            track_centroids_df[
                ["Track", "Track_Cluster_ID", "Track_Cluster_Cluster_ID"]
            ],
            on=["Track", "Track_Cluster_ID"],
        )

        final_df_cents = util_components.get_final_cluster_ids(final_df_cents)

        df = df.merge(
            final_df_cents[["Track", "EnhancedClusters"]].drop_duplicates(), on="Track"
        )

    savePath = os.path.join(pyworkPath, "encoding_df_w_clustering.pckl")
    df.to_pickle(savePath)

    return df


def get_final_tracks(pyworkPath, frameRate, cluster_column="SimpleClusters"):
    final_tracks = {}
    df = pd.read_pickle(os.path.join(pyworkPath, "encoding_df_w_clustering.pckl"))
    df = df.sort_values(by=[cluster_column, "Frame"])
    for idx in df[cluster_column].unique():
        speaker_key = "SPEAKER_{:02d}".format(idx)
        final_tracks[speaker_key] = df[df[cluster_column] == idx]["Frame"].to_list()

    for key in final_tracks.keys():
        final_tracks[key] = util_components.convert_to_ranges(
            final_tracks[key], frameRate
        )

    with open(os.path.join(pyworkPath, "video_diarization_output.pckl"), "wb") as fil:
        pickle.dump(final_tracks, fil)

    return final_tracks
