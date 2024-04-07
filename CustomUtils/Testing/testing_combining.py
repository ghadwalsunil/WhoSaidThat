import pickle
import pandas as pd
import sys
import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pyannote.core import Annotation, Segment
import pandas as pd
import pickle
from pydub import AudioSegment

from dotenv import load_dotenv
import os, subprocess, sys, time
from pyannote.core import Segment, Annotation
from pyannote.core.notebook import Notebook
import matplotlib.pyplot as plt
load_dotenv()
import pickle, json
import torch
from pyannote.audio import Pipeline

class VideoFile:
    def __init__(self, name, save_name, start=-1, end=-1, duration=-1):
        self.name = name
        self.save_name = save_name
        self.start = start
        self.end = end
        self.duration = duration

def perform_clustering(encoding_list, min_clusters=2, max_clusters=6, input_clusters=-1, silhouette_threshold = 0.25):

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



def find_centroid(points):
    # Calculate the mean along each dimension
    centroid = np.mean(points, axis=0)
    return centroid

def mini_track_group(track_id, temp_df):
    
    centroid_list = []
    
    for cluster_num in temp_df["Track_Cluster_ID"].unique():
        centroid = find_centroid(list(temp_df[temp_df["Track_Cluster_ID"] == cluster_num]["Encoding"]))
        centroid_list.append({"Track": track_id, "Track_Cluster_ID": cluster_num, "Track_Cluster_Centroid": centroid})

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
    
    df_filtered = df.groupby('Track').apply(remove_first_two_rows).reset_index(drop=True)
    
    track_centroid_list = []
    df_list = []
    
    for track_id in list(df_filtered["Track"].unique()):
        temp = df_filtered[df_filtered["Track"] == track_id].copy()
        temp["Track_Cluster_ID"] = perform_clustering(list(temp["Encoding"]), max_clusters=3)
        
        track_centroid_list.extend(mini_track_group(track_id, temp))
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
            if track_cluster.track_cluster_percent >= 80:
                return track_cluster.track_cluster_cluster_id
            else:
                break

        combined_percent = 0
        cluster_list = []

        for track_cluster in self.track_cluster_list:
            combined_percent += track_cluster.track_cluster_percent
            cluster_list.append(track_cluster.track_cluster_cluster_id)
            if combined_percent >= 80:
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
    
    temp_df["Final_Cluster"] = temp_df["Track"].apply(lambda x: int(final_cluster_dict.get(x, -1)))
    
    temp_df = temp_df.drop(columns=["TrackClusterClass"], axis=1)

    return temp_df

def convert_rttm_to_diarization(rttm_file, offset):
    
    # Read RTTM file into pandas DataFrame
    rttm_df = pd.read_csv(rttm_file, sep=' ', header=None,
                      names=['temp', 'file_name', 'channel', 'start', 'duration', 'NA_1', 'NA_2', 'speaker_label', 'NA_3', 'NA_4'])
    
    rttm_df.sort_values(by="start", inplace=True)

    diarize_dict = {}

    # Iterate over RTTM rows and add segments to Pyannote annotation
    for _, row in rttm_df.iterrows():
        start_time = round(row['start'] - offset, 2)
        end_time = round(start_time + row['duration'], 2)
        label = row['speaker_label']
        if label not in diarize_dict.keys():
            diarize_dict[label] = [(start_time, end_time)]
        else:
            diarize_dict[label].append((start_time, end_time))

    return diarize_dict

def convert_diarization_output_to_pyannote(diarize_output):
    annotation = Annotation()
    
    for speaker, timelines in diarize_output.items():
        for timeline_start, timeline_end in timelines:
            annotation[Segment(timeline_start, timeline_end)] = speaker

    return annotation

def convert_pyannote_to_diarization(pyannote_output):
    
    diarize_dict = {}
    for duration,_, speaker_key in pyannote_output.itertracks(yield_label=True):
        start_time = round(duration.start, 2)
        end_time = round(duration.end, 2)
        if speaker_key in diarize_dict.keys():
            diarize_dict[speaker_key].append((start_time,end_time))
        else:
            diarize_dict[speaker_key] = [(start_time,end_time)]
            
    return diarize_dict

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

def get_final_tracks(df, frameRate):
    final_tracks = {}
    df = df.sort_values(by=["Final_Cluster", "Frame"])
    for idx in df["Final_Cluster"].unique():
        speaker_key = "SPEAKER_{:02d}".format(idx)
        final_tracks[speaker_key] = df[df["Final_Cluster"] == idx]["Frame"].to_list()

    for key in final_tracks.keys():
        final_tracks[key] = convert_to_ranges(
            final_tracks[key], frameRate
        )

    return final_tracks

def perform_pyannote_diarization(pretrained_model, audio_path, min_cluster_size=12):
    
    pretrained_model.instantiate({
        "clustering" : {
            "min_cluster_size": min_cluster_size
        }
    })
    pretrained_model.parameters(instantiated=True)
    diarization = pretrained_model(audio_path)
    
    return diarization

class PyannoteAudioSegment:
    def __init__(
        self,
        segment_idx,
        audio_segment_start,
        audio_segment_end,
        ad_03_speaker,
        ad_06_speaker,
        ad_09_speaker,
        ad_12_speaker,
        has_overlap,
        vd_speaker=None,
    ):
        self.segment_idx = segment_idx
        self.audio_segment_start = audio_segment_start
        self.audio_segment_end = audio_segment_end
        self.vd_speaker = vd_speaker
        self.ad_03_speaker = ad_03_speaker
        self.ad_06_speaker = ad_06_speaker
        self.ad_09_speaker = ad_09_speaker
        self.ad_12_speaker = ad_12_speaker
        self.has_overlap = has_overlap

    def get_speaker(self, group_id):
        if group_id == "ad_03":
            return self.ad_03_speaker
        elif group_id == "ad_06":
            return self.ad_06_speaker
        elif group_id == "ad_09":
            return self.ad_09_speaker
        elif group_id == "ad_12":
            return self.ad_12_speaker
        elif group_id == "vd":
            return self.vd_speaker
        else:
            print(f"Invalid group id - {group_id}")
            return None

    def __lt__(self, other):
        return self.audio_segment_start < other.audio_segment_start

    def __gt__(self, other):
        return self.audio_segment_start > other.audio_segment_start

    def __eq__(self, other):
        return self.audio_segment_start == other.audio_segment_start

    def __le__(self, other):
        return self.audio_segment_start <= other.audio_segment_start

    def __ge__(self, other):
        return self.audio_segment_start >= other.audio_segment_start

    def __ne__(self, other):
        return self.audio_segment_start != other.audio_segment_start

    def __repr__(self) -> str:
        return f"PyannoteAudioSegment({self.audio_segment_start}, {self.audio_segment_end}, {self.ad_03_speaker}, {self.ad_06_speaker}, {self.ad_09_speaker}, {self.ad_12_speaker}, {self.vd_speaker})"


class MappingClass:
    def __init__(self):
        self.mapping_dict = {}

    def add_mapping(self, speaker_key_id, speaker_value_id, audio_segment_start, audio_segment_end):
        if speaker_value_id == "Unknown":
            return
        if speaker_key_id not in self.mapping_dict.keys():
            self.mapping_dict[speaker_key_id] = {speaker_value_id: audio_segment_end - audio_segment_start}
        else:
            if speaker_value_id not in self.mapping_dict[speaker_key_id].keys():
                self.mapping_dict[speaker_key_id][speaker_value_id] = audio_segment_end - audio_segment_start
            else:
                self.mapping_dict[speaker_key_id][speaker_value_id] += audio_segment_end - audio_segment_start

    def get_max_mapping(self, speaker_key_id):
        if speaker_key_id not in self.mapping_dict.keys():
            return None

        return max(self.mapping_dict[speaker_key_id], key=self.mapping_dict[speaker_key_id].get)


class DiarizationOutput:
    def __init__(
        self, audio_diarization_03, audio_diarization_06, audio_diarization_09, audio_diarization_12, video_diarization
    ):
        self.audio_diarization_03 = audio_diarization_03
        self.audio_diarization_06 = audio_diarization_06
        self.audio_diarization_09 = audio_diarization_09
        self.audio_diarization_12 = audio_diarization_12
        self.video_diarization = video_diarization
        self.audio_segments: list[PyannoteAudioSegment] = self.get_audio_segments()
        self.ad03_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_03")
        self.ad06_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_06")
        self.ad09_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_09")
        self.ad12_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_12")
        self.predict_unknowns()

    def get_corresponding_speaker(self, present_group_id, target_group_id, speaker_id):

        if present_group_id == "ad_03":
            groups_not_allowed = ["ad_03"]
        elif present_group_id == "ad_06":
            groups_not_allowed = ["ad_03", "ad_06"]
        elif present_group_id == "ad_09":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09"]
        elif present_group_id == "ad_12":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09", "ad_12"]
        elif present_group_id == "vd":
            groups_not_allowed = ["vd"]
        else:
            print(f"Invalid present group id - {present_group_id}")
            return None

        if target_group_id in groups_not_allowed:
            print(f"For present_group {present_group_id}, target_group should not be among {groups_not_allowed}")
            return None

        present_group = self.get_group_by_id(present_group_id)
        if speaker_id not in present_group.keys():
            print(f"Speaker_id {speaker_id} not in present group {present_group_id} keys - {present_group.keys()}")
            return None

        speaker_interval = present_group[speaker_id]
        target_group = self.get_group_by_id(target_group_id)

        return self.get_mapping(speaker_interval, target_group)

    def get_all_child_speakers(self, parent_group_id, child_group_id, parent_speaker_id):

        if parent_group_id == "ad_12":
            groups_not_allowed = ["ad_12"]
        elif parent_group_id == "ad_09":
            groups_not_allowed = ["ad_09", "ad_12"]
        elif parent_group_id == "ad_06":
            groups_not_allowed = ["ad_06", "ad_09", "ad_12"]
        elif parent_group_id == "ad_03":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09", "ad_12"]
        elif parent_group_id == "vd":
            groups_not_allowed = ["vd"]
        else:
            print(f"Invalid parent group id - {parent_group_id}")
            return None

        if child_group_id in groups_not_allowed:
            print(f"For parent_group {child_group_id}, child_group should not be among {groups_not_allowed}")
            return None

        parent_group = self.get_group_by_id(parent_group_id)
        if parent_speaker_id not in parent_group.keys():
            print(
                f"parent_speaker_id {parent_speaker_id} not in parent_group {parent_group_id} keys - {parent_group.keys()}"
            )
            return None

        child_group = self.get_group_by_id(child_group_id)

        child_speakers = []

        for child_speaker_id in child_group.keys():
            _parent_speaker_id = self.get_corresponding_speaker(
                present_group_id=child_group_id, target_group_id=parent_group_id, speaker_id=child_speaker_id
            )

            if _parent_speaker_id == parent_speaker_id:
                child_speakers.append(child_speaker_id)

        return child_speakers

    def get_native_speakers_in_parent_group(self, child_group_id, parent_group_id, child_speaker_id):

        if child_group_id == "ad_03":
            groups_not_allowed = ["ad_03"]
        elif child_group_id == "ad_06":
            groups_not_allowed = ["ad_03", "ad_06"]
        elif child_group_id == "ad_09":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09"]
        elif child_group_id == "ad_12":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09", "ad_12"]
        elif child_group_id == "vd":
            groups_not_allowed = ["vd"]
        else:
            print(f"Invalid child group id - {child_group_id}")
            return None

        if parent_group_id in groups_not_allowed:
            print(f"For child_group {child_group_id}, parent_group should not be among {groups_not_allowed}")
            return None

        child_group = self.get_group_by_id(child_group_id)
        if child_speaker_id not in child_group.keys():
            print(
                f"child_speaker_id {child_speaker_id} not in child_group {child_speaker_id} keys - {child_group.keys()}"
            )
            return None

        parent_group = self.get_group_by_id(parent_group_id)

        parent_speaker_id = self.get_corresponding_speaker(
            target_group_id=parent_group_id, present_group_id=child_group_id, speaker_id=child_speaker_id
        )

        return self.get_all_child_speakers(
            parent_group_id=parent_group_id, child_group_id=child_group_id, parent_speaker_id=parent_speaker_id
        )

    def get_audio_segments(self):
        audio_segment_list = []
        segment_idx = 0
        for speaker_id in self.audio_diarization_03.keys():
            for speech_segment_start, speech_segment_end in self.audio_diarization_03[speaker_id]:
                audio_segment = PyannoteAudioSegment(
                    segment_idx=segment_idx,
                    audio_segment_start=speech_segment_start,
                    audio_segment_end=speech_segment_end,
                    ad_03_speaker=speaker_id,
                    ad_06_speaker=self.get_corresponding_speaker(
                        present_group_id="ad_03", target_group_id="ad_06", speaker_id=speaker_id
                    ),
                    ad_09_speaker=self.get_corresponding_speaker(
                        present_group_id="ad_03", target_group_id="ad_09", speaker_id=speaker_id
                    ),
                    ad_12_speaker=self.get_corresponding_speaker(
                        present_group_id="ad_03", target_group_id="ad_12", speaker_id=speaker_id
                    ),
                    vd_speaker=self.get_mapping([(speech_segment_start, speech_segment_end)], self.video_diarization),
                    has_overlap=False,
                )
                audio_segment_list.append(audio_segment)

                segment_idx += 1

        audio_segment_list.sort()

        # Check whether each audio segment has overlap with other audio segments
        for i in range(len(audio_segment_list)):
            for j in range(i + 1, len(audio_segment_list)):
                if audio_segment_list[i].audio_segment_end > audio_segment_list[j].audio_segment_start:
                    audio_segment_list[i].has_overlap = True
                    audio_segment_list[j].has_overlap = True
                    audio_segment_list[i].vd_speaker = "Unknown"
                    audio_segment_list[j].vd_speaker = "Unknown"

        return audio_segment_list

    def perform_audio_video_mapping(self, audio_group_id):
        mapping_class = MappingClass()
        for audio_segment in self.audio_segments:
            if audio_group_id == "ad_03":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_03_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            elif audio_group_id == "ad_06":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_06_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            elif audio_group_id == "ad_09":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_09_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            elif audio_group_id == "ad_12":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_12_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            else:
                print(f"Invalid audio_group_id - {audio_group_id}")
                return None

        return mapping_class

    def predict_unknowns(self):
        for audio_segment in self.audio_segments:
            if audio_segment.vd_speaker == "Unknown":
                vd_speaker = self.ad03_video_mapping.get_max_mapping(audio_segment.ad_03_speaker)
                if vd_speaker is not None:
                    audio_segment.vd_speaker = vd_speaker
                    self.ad03_video_mapping.add_mapping(
                        speaker_key_id=audio_segment.ad_03_speaker,
                        speaker_value_id=vd_speaker,
                        audio_segment_start=audio_segment.audio_segment_start,
                        audio_segment_end=audio_segment.audio_segment_end,
                    )
                else:
                    # audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_03_speaker}"
                    vd_speaker = self.ad06_video_mapping.get_max_mapping(audio_segment.ad_06_speaker)
                    if vd_speaker is not None:
                        audio_segment.vd_speaker = vd_speaker
                        self.ad06_video_mapping.add_mapping(
                            speaker_key_id=audio_segment.ad_06_speaker,
                            speaker_value_id=vd_speaker,
                            audio_segment_start=audio_segment.audio_segment_start,
                            audio_segment_end=audio_segment.audio_segment_end,
                        )
                    else:
                        # audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_06_speaker}"
                        vd_speaker = self.ad09_video_mapping.get_max_mapping(audio_segment.ad_09_speaker)
                        if vd_speaker is not None:
                            audio_segment.vd_speaker = vd_speaker
                            self.ad09_video_mapping.add_mapping(
                                speaker_key_id=audio_segment.ad_09_speaker,
                                speaker_value_id=vd_speaker,
                                audio_segment_start=audio_segment.audio_segment_start,
                                audio_segment_end=audio_segment.audio_segment_end,
                            )
                        else:
                            # audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_09_speaker}"
                            vd_speaker = self.ad12_video_mapping.get_max_mapping(audio_segment.ad_12_speaker)
                            if vd_speaker is not None:
                                audio_segment.vd_speaker = vd_speaker
                                self.ad12_video_mapping.add_mapping(
                                    speaker_key_id=audio_segment.ad_12_speaker,
                                    speaker_value_id=vd_speaker,
                                    audio_segment_start=audio_segment.audio_segment_start,
                                    audio_segment_end=audio_segment.audio_segment_end,
                                )
                            else:
                                audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_12_speaker}"

    def get_diarization_output(self):
        diarize_output = {}
        for audio_segment in self.audio_segments:
            if audio_segment.vd_speaker not in diarize_output.keys():
                diarize_output[audio_segment.vd_speaker] = []
            diarize_output[audio_segment.vd_speaker].append(
                (audio_segment.audio_segment_start, audio_segment.audio_segment_end)
            )
        return diarize_output

    def get_group_by_id(self, group_id):
        if group_id == "ad_03":
            return self.audio_diarization_03
        if group_id == "ad_06":
            return self.audio_diarization_06
        if group_id == "ad_09":
            return self.audio_diarization_09
        if group_id == "ad_12":
            return self.audio_diarization_12
        if group_id == "vd":
            return self.video_diarization

    @staticmethod
    def get_mapping(speaker_interval, target_group):
        max_overlap = 0
        max_overlap_speaker = "Unknown"
        for speaker_id in target_group.keys():
            result_overlap, _ = find_overlap(
                speaker_interval,
                target_group[speaker_id],
            )
            if result_overlap > max_overlap:
                max_overlap = result_overlap
                max_overlap_speaker = speaker_id

        return max_overlap_speaker


def find_overlap(intervals1, intervals2):
    overlap = 0
    total_duration1 = 0
    total_duration2 = 0

    # Calculate the total duration of intervals in intervals1
    for start, end in intervals1:
        total_duration1 += end - start

    # Calculate the total duration of intervals in intervals2 and find the overlap
    for start, end in intervals2:
        total_duration2 += end - start
        for s1, e1 in intervals1:
            common_start = max(s1, start)
            common_end = min(e1, end)
            if common_start < common_end:
                overlap += common_end - common_start

    # Calculate the percentage of overlap with respect to intervals1
    if total_duration1 == 0:
        percentage_overlap1 = 0
    else:
        percentage_overlap1 = (overlap / total_duration1) * 100

    # Calculate the percentage of overlap with respect to intervals2
    if total_duration2 == 0:
        percentage_overlap2 = 0
    else:
        percentage_overlap2 = (overlap / total_duration2) * 100

    return percentage_overlap1, percentage_overlap2