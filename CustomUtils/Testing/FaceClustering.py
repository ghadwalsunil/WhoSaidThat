import pickle
import pandas as pd
import sys
import time
import numpy as np
from clusteval import clusteval
import umap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def perform_clustering(
    encoding_list,
    min_clusters=2,
    max_clusters=6,
    input_clusters=-1,
    silhouette_threshold=0.25,
    return_centers=False,
    verbose=False,
    perform_umap=False,
):

    optimal_cluster = -1
    max_silhouette = 0

    num_points = len(encoding_list)

    if perform_umap:
        reducer = umap.UMAP()
        encoding_list = reducer.fit_transform(encoding_list)

    if num_points <= 20:
        return [0] * num_points

    if input_clusters < 0:
        for num_clusters in range(min_clusters, max_clusters + 1):
            if num_clusters >= num_points:
                break
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(encoding_list)

            silhouette_avg = silhouette_score(encoding_list, cluster_labels)
            if verbose:
                print(f"Num of clusters - {num_clusters} - {silhouette_avg}")

            if silhouette_avg > max_silhouette:
                max_silhouette = silhouette_avg
                optimal_cluster = num_clusters
    else:
        optimal_cluster = input_clusters

    if max_silhouette >= silhouette_threshold:
        kmeans = KMeans(n_clusters=optimal_cluster, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(encoding_list)

    else:
        cluster_labels = [0] * num_points

    if return_centers:
        return cluster_labels, kmeans.cluster_centers_
    else:
        return cluster_labels


def find_centroid(points):
    # Calculate the mean along each dimension
    centroid = np.mean(points, axis=0)
    return centroid


def mini_track_group(track_id, temp_df, encoding_col):

    centroid_list = []

    for cluster_num in temp_df["Track_Cluster_ID"].unique():
        centroid = find_centroid(list(temp_df[temp_df["Track_Cluster_ID"] == cluster_num][encoding_col]))
        centroid_list.append({"Track": track_id, "Track_Cluster_ID": cluster_num, "Track_Cluster_Centroid": centroid})

    return centroid_list


def get_subset(df, encoding_col):

    if "Track" not in df.columns:
        print("Invalid df")
        return None
    if encoding_col not in df.columns:
        print("Invalid df")
        return None

    def remove_first_two_rows(group):
        return group.iloc[2:]

    df_filtered = df.groupby("Track").apply(remove_first_two_rows).reset_index(drop=True)

    track_centroid_list = []
    df_list = []

    for track_id in list(df_filtered["Track"].unique()):
        temp = df_filtered[df_filtered["Track"] == track_id].copy()
        #         print(f"track_id {track_id} - {len(temp)}")
        temp["Track_Cluster_ID"] = perform_clustering(
            list(temp[encoding_col]), max_clusters=3, silhouette_threshold=0.2
        )

        track_centroid_list.extend(mini_track_group(track_id, temp, encoding_col))
        df_list.append(temp)

    return pd.concat(df_list), pd.DataFrame(track_centroid_list)


class TrackCluster:
    def __init__(self, track_cluster_cluster_id, track_cluster_cluster_count, track_cluster_percent):
        self.track_cluster_cluster_id = track_cluster_cluster_id
        self.track_cluster_cluster_count = track_cluster_cluster_count
        self.track_cluster_percent = track_cluster_percent

    def __lt__(self, other):
        return self.track_cluster_cluster_count < other.track_cluster_cluster_count

    def __le__(self, other):
        return self.track_cluster_cluster_count <= other.track_cluster_cluster_count

    def __eq__(self, other):
        return self.track_cluster_cluster_count == other.track_cluster_cluster_count

    def __ne__(self, other):
        return self.track_cluster_cluster_count != other.track_cluster_cluster_count

    def __gt__(self, other):
        return self.track_cluster_cluster_count > other.track_cluster_cluster_count

    def __ge__(self, other):
        return self.track_cluster_cluster_count >= other.track_cluster_cluster_count

    def __repr__(self) -> str:
        return f"TrackCluster(track_cluster_cluster_id={self.track_cluster_cluster_id}, track_cluster_cluster_count={self.track_cluster_cluster_count}, track_cluster_percent={self.track_cluster_percent})"


class Track:
    def __init__(self, track_id, track_count, track_cluster_list):
        self.track_id = track_id
        self.track_count = track_count
        self.track_cluster_list = track_cluster_list
        self.track_cluster_list.sort(reverse=True)
        self.final_cluster = -1

    def get_cluster_set(self):
        cluster_set = set()
        for track_cluster in self.track_cluster_list:
            cluster_set.add(cluster_set)

        return cluster_set

    def get_majority_forming_cluster_id(self):

        if self.track_count <= 20:
            max_cluster_count = 0
            max_cluster_id = -1
            for track_cluster in self.track_cluster_list:
                if track_cluster.track_cluster_cluster_count > max_cluster_count:
                    max_cluster_count = track_cluster.track_cluster_cluster_count
                    max_cluster_id = track_cluster.track_cluster_cluster_id

            return max_cluster_id

        for track_cluster in self.track_cluster_list:
            if track_cluster.track_cluster_percent >= 50:
                return track_cluster.track_cluster_cluster_id
            else:
                break

        combined_percent = 0
        cluster_list = []

        for track_cluster in self.track_cluster_list:
            combined_percent += track_cluster.track_cluster_percent
            cluster_list.append(track_cluster.track_cluster_cluster_id)
            if combined_percent >= 50:
                break

        return cluster_list

    def __repr__(self) -> str:
        return f"Track(track_id={self.track_id}, track_count={self.track_count}, track_cluster_list={self.track_cluster_list})"


class TrackList:
    def __init__(self, df):
        self.track_list = self.get_track_list(df)

    def get_final_clusters(self):

        final_cluster_dict = {}
        combine_dict = {}

        for track in self.track_list:
            final_cluster = track.get_majority_forming_cluster_id()
            final_cluster_dict[track.track_id] = final_cluster
            if type(final_cluster) == list:
                combine_dict[track.track_id] = final_cluster

        existing_sets_list: list[set] = []

        for track_id, cluster_list in combine_dict.items():
            cluster_set = set(cluster_list)

            if len(existing_sets_list) == 0:
                existing_sets_list.append(cluster_set)
                continue

            for existing_sets in existing_sets_list:
                if len(existing_sets.intersection(cluster_set)) > 0:
                    existing_sets.update(cluster_set)
                else:
                    existing_sets_list.append(cluster_set)

        # assign the first value of the set to all the values in the set
        final_combo_clusters = {}
        for cluster_set in existing_sets_list:
            if len(cluster_set) == 0:
                continue
            cluster_id = cluster_set.pop()
            final_combo_clusters[cluster_id] = cluster_id
            for cluster in cluster_set:
                final_combo_clusters[cluster] = cluster_id

        for track_id, cluster in final_cluster_dict.items():
            if type(cluster) == list:
                final_cluster_dict[track_id] = final_combo_clusters[cluster[0]]
            elif cluster in final_combo_clusters:
                final_cluster_dict[track_id] = final_combo_clusters[cluster]

        return final_cluster_dict

    def get_track(self, track_id):
        for track in self.track_list:
            if track.track_id == track_id:
                return track

    @staticmethod
    def get_track_list(df):

        df["TrackClusterClass"] = df.apply(
            lambda row: TrackCluster(
                track_cluster_cluster_id=row["Track_Cluster_Cluster_ID"],
                track_cluster_cluster_count=row["Track_Cluster_Cluster_Count"],
                track_cluster_percent=row["Track_Cluster_Percent"],
            ),
            axis=1,
        )

        df = (
            df.groupby("Track")
            .agg(
                {
                    "Track_Count": "first",
                    "TrackClusterClass": lambda x: list(x),
                }
            )
            .reset_index()
        )

        df["TrackClass"] = df.apply(
            lambda row: Track(
                track_id=row["Track"],
                track_count=row["Track_Count"],
                track_cluster_list=row["TrackClusterClass"],
            ),
            axis=1,
        )

        return df["TrackClass"].tolist()

    def __repr__(self) -> str:
        return f"TrackList(track_list_count={len(self.track_list)})"


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
    temp_df = temp_df.merge(track_cluster_count_df, on=["Track", "Track_Cluster_Cluster_ID"])

    temp_df = temp_df.drop_duplicates().reset_index(drop=True)

    #     Get percentage of track cluster count in total track count

    temp_df["Track_Cluster_Percent"] = temp_df.apply(
        lambda row: (row["Track_Cluster_Cluster_Count"] / row["Track_Count"]) * 100, axis=1
    )

    track_list = TrackList(temp_df)

    final_cluster_dict = track_list.get_final_clusters()

    temp_df["EnhancedClusters"] = temp_df["Track"].apply(lambda x: int(final_cluster_dict.get(x, -1)))

    temp_df = temp_df.drop(columns=["TrackClusterClass"], axis=1)

    return temp_df


def match_output(output_df, gt_col, pred_col, matched_col):

    pred_list = output_df[pred_col].unique()
    pred_match = {}
    pred_count = {}

    for pred in pred_list:
        if pred == "Unassigned":
            pred_match[pred] = "Unassigned"
        else:
            pred_match[pred] = output_df[output_df[pred_col] == pred][gt_col].value_counts().idxmax()
            pred_count[pred] = len(output_df[output_df[pred_col] == pred])

    temp_count = {}

    for pred, gt in pred_match.items():
        if gt in temp_count.keys():
            temp_count[gt].append(pred)
        else:
            temp_count[gt] = [pred]

    for gt, preds in temp_count.items():
        if len(preds) > 1:
            max_count = 0
            max_match = None
            for pred in preds:
                if pred_count[pred] > max_count:
                    if max_match is None:
                        max_match = pred
                        max_count = pred_count[pred]
                    else:
                        pred_match[max_match] = "Unassigned"
                        max_match = pred
                        max_count = pred_count[pred]
                else:
                    pred_match[pred] = "Unassigned"
            pred_match[max_match] = gt

    output_df[matched_col] = output_df.apply(
        lambda row: pred_match[row[pred_col]],
        axis=1,
    )

    return output_df


def compute_performance(output_df, gt_col, pred_col, matched_col):

    def _match(gt, pred, matched):
        if pred == "Unassigned":
            return "Missed"
        elif matched == "Unassigned" or matched != gt:
            return "Confusion"
        else:
            return "Correct"

    output_df["matched_result"] = output_df.apply(
        lambda row: _match(row[gt_col], row[pred_col], row[matched_col]),
        axis=1,
    )

    return output_df


file_name = "PiersMorgan_1_165_368"
df_gt = pd.read_excel("../del_later/gt_3_video.xlsx")
with open(f"/vol3/sunil/output/video_temp/{file_name}/pywork/encoding_df.pckl", "rb") as fil:
    encoding_df = pickle.load(fil)
temp_df = encoding_df.merge(df_gt[df_gt["Filename"] == file_name][["Track", "GT"]])


# Simple
temp_df["SimpleClusters"] = perform_clustering(temp_df["Encoding"].to_list(), silhouette_threshold=0, max_clusters=10)
matched_df = match_output(temp_df, gt_col="GT", pred_col="SimpleClusters", matched_col="Matched_Output")
matched_df = compute_performance(
    output_df=matched_df, gt_col="GT", pred_col="SimpleClusters", matched_col="Matched_Output"
)
print(matched_df[["GT", "matched_result", "SimpleClusters"]].groupby(["GT", "matched_result"]).count())


# Enhanced
# reducer = umap.UMAP(n_components=5, n_neighbors=30)
# umap_encoding = reducer.fit_transform(list(temp_df["Encoding"]))
# temp_df["Umap_Encoding"] = list(umap_encoding)
final_df_cents, track_centroids_df = get_subset(temp_df, "Encoding")
track_centroids_df["Track_Cluster_Cluster_ID"], cluster_centers = perform_clustering(
    list(track_centroids_df["Track_Cluster_Centroid"]),
    max_clusters=12,
    silhouette_threshold=0.2,
    return_centers=True,
    perform_umap=False,
)
final_df_cents = final_df_cents.merge(
    track_centroids_df[["Track", "Track_Cluster_ID", "Track_Cluster_Cluster_ID"]], on=["Track", "Track_Cluster_ID"]
)
final_df_cents = get_final_cluster_ids(final_df_cents)
final_df_cents = temp_df.merge(final_df_cents[["Track", "EnhancedClusters"]].drop_duplicates(), on="Track")
matched_df = match_output(final_df_cents, gt_col="GT", pred_col="EnhancedClusters", matched_col="Matched_Output")
matched_df = compute_performance(
    output_df=matched_df, gt_col="GT", pred_col="EnhancedClusters", matched_col="Matched_Output"
)
matched_df[["GT", "matched_result", "EnhancedClusters"]].groupby(["GT", "matched_result"]).count()
# print(matched_df["EnhancedClusters"].value_counts())
print(matched_df[["GT", "matched_result", "EnhancedClusters"]].groupby(["GT", "matched_result"]).count())
