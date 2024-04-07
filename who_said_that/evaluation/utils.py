from pyannote.core import Annotation, Segment
import pandas as pd


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


def get_word_to_speaker_mapping(word_start, word_end, final_output):
    final_speaker = None
    max_overlap = 0

    for speaker_key in final_output.keys():
        word_overlap, _ = find_overlap([(word_start, word_end)], final_output[speaker_key])
        if word_overlap > max_overlap:
            final_speaker = speaker_key
            max_overlap = word_overlap

    if final_speaker is None:
        final_speaker = "Unassigned"

    return final_speaker


def match_output(output_df):

    assigned_speaker_list = output_df["assigned_speaker"].unique()
    speaker_match = {}
    speaker_count = {}

    for assigned_speaker in assigned_speaker_list:
        if assigned_speaker == "Unassigned":
            speaker_match[assigned_speaker] = "Unassigned"
        else:
            speaker_match[assigned_speaker] = (
                output_df[output_df["assigned_speaker"] == assigned_speaker]["speaker"].value_counts().idxmax()
            )
            speaker_count[assigned_speaker] = len(output_df[output_df["assigned_speaker"] == assigned_speaker])

    temp_count = {}

    for speaker_pred, speaker_actual in speaker_match.items():
        if speaker_actual in temp_count.keys():
            temp_count[speaker_actual].append(speaker_pred)
        else:
            temp_count[speaker_actual] = [speaker_pred]

    for speaker_actual, assigned_speakers in temp_count.items():
        if len(assigned_speakers) > 1:
            max_count = 0
            max_speaker = None
            for speaker in assigned_speakers:
                if speaker_count[speaker] > max_count:
                    if max_speaker is None:
                        max_speaker = speaker
                        max_count = speaker_count[speaker]
                    else:
                        speaker_match[max_speaker] = "Unassigned"
                        max_speaker = speaker
                        max_count = speaker_count[speaker]
                else:
                    speaker_match[speaker] = "Unassigned"
            speaker_match[max_speaker] = speaker_actual

    output_df["matched_speaker"] = output_df.apply(
        lambda row: speaker_match[row["assigned_speaker"]],
        axis=1,
    )

    return output_df


def compute_performance(output_df):

    def speaker_match(speaker, assigned_speaker, matched_speaker):
        if assigned_speaker == "Unassigned":
            return "Missed"
        elif matched_speaker == "Unassigned" or matched_speaker != speaker:
            return "Confusion"
        else:
            return "Correct"

    output_df["matched_result"] = output_df.apply(
        lambda row: speaker_match(row["speaker"], row["assigned_speaker"], row["matched_speaker"]),
        axis=1,
    )

    return output_df["matched_result"].value_counts().to_dict()


def compute_word_diarization_error_rate(ground_truth, diarization_output):

    ground_truth["assigned_speaker"] = ground_truth.apply(
        lambda row: (
            get_word_to_speaker_mapping(
                row["word_start"],
                row["word_end"],
                diarization_output,
            )
            if row["word_end"] > row["word_start"]
            else get_word_to_speaker_mapping(
                row["word_start"],
                row["word_end"] + 0.01,
                diarization_output,
            )
        ),
        axis=1,
    )

    matched_df = match_output(ground_truth)
    return compute_performance(matched_df)


def convert_rttm_to_diarization(rttm_file, offset):

    # Read RTTM file into pandas DataFrame
    rttm_df = pd.read_csv(
        rttm_file,
        sep=" ",
        header=None,
        names=["temp", "file_name", "channel", "start", "duration", "NA_1", "NA_2", "speaker_label", "NA_3", "NA_4"],
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


def convert_diarization_output_to_pyannote(diarize_output):
    annotation = Annotation()

    for speaker, timelines in diarize_output.items():
        for timeline_start, timeline_end in timelines:
            annotation[Segment(timeline_start, timeline_end)] = speaker

    return annotation


def convert_pyannote_to_diarization(pyannote_output):

    diarize_dict = {}
    for duration, _, speaker_key in pyannote_output.itertracks(yield_label=True):
        start_time = round(duration.start, 2)
        end_time = round(duration.end, 2)
        if speaker_key in diarize_dict.keys():
            diarize_dict[speaker_key].append((start_time, end_time))
        else:
            diarize_dict[speaker_key] = [(start_time, end_time)]

    return diarize_dict
