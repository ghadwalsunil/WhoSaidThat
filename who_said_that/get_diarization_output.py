import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import pandas as pd

from who_said_that.evaluation.utils import compute_word_diarization_error_rate

temp = pickle.load(
    open("output/run_output/final_diarization_output_MY_DATASET.pckl", "rb")
)


def convert_rttm_to_diarization(rttm_file, offset):

    # Read RTTM file into pandas DataFrame
    rttm_df = pd.read_csv(
        rttm_file,
        sep=" ",
        header=None,
        names=[
            "temp",
            "file_name",
            "channel",
            "start",
            "duration",
            "NA_1",
            "NA_2",
            "speaker_label",
            "NA_3",
            "NA_4",
        ],
    )

    rttm_df.sort_values(by="start", inplace=True)

    diarize_dict = {}

    # Iterate over RTTM rows and add segments to Pyannote annotation
    for _, row in rttm_df.iterrows():
        start_time = round(row["start"] - offset, 2)
        end_time = round(start_time + row["duration"], 2)
        label = row["speaker_label"]
        if label not in diarize_dict.keys():
            diarize_dict[label] = [(start_time, end_time)]
        else:
            diarize_dict[label].append((start_time, end_time))

    return diarize_dict


rttm_dir = "../AVA-AVD/save/token/avaavd/rttms/"
ground_truth_dir = "/vol3/sunil/AVA-AVD/dataset/csv/"

final_results = []

for video_file in temp.keys():
    diarize_output = convert_rttm_to_diarization(
        os.path.join(rttm_dir, f"{video_file}.rttm"), 0
    )
    video_results = {"video_name": video_file}
    ground_truth = pd.read_csv(
        os.path.join(ground_truth_dir, f"{video_file}.csv"), sep="\t"
    )
    video_results.update(
        compute_word_diarization_error_rate(ground_truth, diarize_output)
    )
    final_results.append(video_results)

df = pd.DataFrame(final_results)

df.to_excel("output/run_output/AVR_NET_MY_DATASET.xlsx", index=False)
