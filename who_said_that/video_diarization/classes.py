class TrackCluster:
    def __init__(
        self,
        track_cluster_cluster_id,
        track_cluster_cluster_count,
        track_cluster_percent,
    ):
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
