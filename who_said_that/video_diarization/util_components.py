import face_recognition
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from who_said_that.video_diarization.classes import TrackList


def mark_frames(df, min_track_frames=10, marked_percent=0.1):
    marked_frames = []
    track_ids = []

    for track_id, group in df.groupby("Track"):
        n_frames = len(group)

        if n_frames <= 5:
            min_n_marked = 1 * n_frames
        elif n_frames <= 10:
            min_n_marked = 5 + (n_frames - 5) * 0.5
        elif n_frames <= 20:
            min_n_marked = 7.5 + (n_frames - 10) * 0.25
        elif n_frames <= 50:
            min_n_marked = 10 + (n_frames - 20) * 0.1
        elif n_frames <= 200:
            min_n_marked = 15 + (n_frames - 50) * 0.05
        else:
            min_n_marked = 25 + (n_frames - 200) * 0.01

        n_marked = max(1, min(int(min_n_marked), n_frames))

        indices = np.linspace(0, n_frames - 1, num=n_marked, dtype=int)
        marked_frames.extend(list(group["Frame"].iloc[indices]))
        track_ids.extend([track_id] * len(indices))

    marked_df = pd.DataFrame(marked_frames, columns=["Frame"])
    marked_df["Track"] = track_ids
    marked_df["Marked"] = True
    df = df.merge(
        marked_df[["Track", "Frame", "Marked"]], on=["Track", "Frame"], how="left"
    )
    df["Marked"].fillna(False, inplace=True)

    return df


def generate_encodings(image_file, s, x, y):
    image = face_recognition.load_image_file(image_file)
    return face_recognition.face_encodings(
        image,
        [
            (
                int(y - s),
                int(x + s),
                int(y + s),
                int(x - s),
            )
        ],
        model="small",
    )[0]


def perform_clustering(
    encoding_list,
    min_clusters=2,
    max_clusters=6,
    input_clusters=-1,
    silhouette_threshold=0.25,
):

    optimal_cluster = -1
    max_silhouette = 0

    num_points = len(encoding_list)

    if input_clusters < 0:
        for num_clusters in range(min_clusters, max_clusters + 1):
            if num_clusters >= num_points:
                break
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(encoding_list)

            silhouette_avg = silhouette_score(encoding_list, cluster_labels)

            if silhouette_avg > max_silhouette:
                max_silhouette = silhouette_avg
                optimal_cluster = num_clusters
    else:
        optimal_cluster = input_clusters

    if max_silhouette >= silhouette_threshold:
        kmeans = KMeans(n_clusters=optimal_cluster, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(encoding_list)

    else:
        cluster_labels = [0] * len(encoding_list)

    return cluster_labels


def convert_to_ranges(lst, frame_rate):
    ranges = []
    start = lst[0]

    threshold = round((frame_rate / 10) * 3)

    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] > threshold:
            if lst[i - 1] - start > threshold:
                ranges.append(
                    (round(start / frame_rate, 2), round(lst[i - 1] / frame_rate, 2))
                )
            start = lst[i]

    # Add the last range
    if lst[-1] - start > threshold:
        ranges.append((round(start / frame_rate, 2), round(lst[-1] / frame_rate, 2)))

    return ranges


def find_centroid(points):
    # Calculate the mean along each dimension
    centroid = np.mean(points, axis=0)
    return centroid


def mini_track_group(track_id, temp_df):

    centroid_list = []

    for cluster_num in temp_df["Track_Cluster_ID"].unique():
        centroid = find_centroid(
            list(temp_df[temp_df["Track_Cluster_ID"] == cluster_num]["Encoding"])
        )
        centroid_list.append(
            {
                "Track": track_id,
                "Track_Cluster_ID": cluster_num,
                "Track_Cluster_Centroid": centroid,
            }
        )

    return centroid_list


def get_subset(df):

    if "Track" not in df.columns:
        print("Invalid df")
        return None
    if "Encoding" not in df.columns:
        print("Invalid df")
        return None

    def remove_first_two_rows(group):
        return group.iloc[2:]

    df_filtered = (
        df.groupby("Track").apply(remove_first_two_rows).reset_index(drop=True)
    )

    track_centroid_list = []
    df_list = []

    for track_id in list(df_filtered["Track"].unique()):
        temp = df_filtered[df_filtered["Track"] == track_id].copy()
        temp["Track_Cluster_ID"] = perform_clustering(
            list(temp["Encoding"]), max_clusters=3
        )

        track_centroid_list.extend(mini_track_group(track_id, temp))
        df_list.append(temp)

    return pd.concat(df_list), pd.DataFrame(track_centroid_list)


def get_final_cluster_ids(final_df):

    temp_df = final_df[["Track", "Track_Cluster_Cluster_ID"]].copy()

    #     Get count of each track

    track_count_df = (
        pd.DataFrame(temp_df[["Track"]].value_counts())
        .sort_values("Track")
        .reset_index()
        .rename(columns={"count": "Track_Count"})
    )

    #     Get count of each cluster in each track

    track_cluster_count_df = (
        pd.DataFrame(temp_df[["Track", "Track_Cluster_Cluster_ID"]].value_counts())
        .reset_index()
        .rename(columns={"count": "Track_Cluster_Cluster_Count"})
        .sort_values(["Track", "Track_Cluster_Cluster_ID"])
        .reset_index(drop=True)
    )

    temp_df = temp_df.merge(track_count_df, on="Track")
    temp_df = temp_df.merge(
        track_cluster_count_df, on=["Track", "Track_Cluster_Cluster_ID"]
    )

    temp_df = temp_df.drop_duplicates().reset_index(drop=True)

    #     Get percentage of track cluster count in total track count

    temp_df["Track_Cluster_Percent"] = temp_df.apply(
        lambda row: (row["Track_Cluster_Cluster_Count"] / row["Track_Count"]) * 100,
        axis=1,
    )

    track_list = TrackList(temp_df)

    final_cluster_dict = track_list.get_final_clusters()

    temp_df["EnhancedClusters"] = temp_df["Track"].apply(
        lambda x: int(final_cluster_dict.get(x, -1))
    )

    temp_df = temp_df.drop(columns=["TrackClusterClass"], axis=1)

    return temp_df
