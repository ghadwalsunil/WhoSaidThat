import sys
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def perform_clustering(encoding_list, min_clusters=2, max_clusters=6):

    optimal_cluster = -1
    max_silhouette = 0

    for num_clusters in range(min_clusters, max_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(encoding_list)

        silhouette_avg = silhouette_score(encoding_list, cluster_labels)
        if silhouette_avg > max_silhouette:
            max_silhouette = silhouette_avg
            optimal_cluster = num_clusters

    kmeans = KMeans(n_clusters=optimal_cluster, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(encoding_list)

    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Optimal number of clusters: %d \r\n" % (optimal_cluster)
    )

    return cluster_labels


def convert_to_ranges(lst, frame_rate):
    ranges = []
    start = lst[0]

    threshold = round(frame_rate / 10) * 3

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
