import pandas as pd
from pyannote.core import Annotation, Segment
from who_said_that.utils.components import find_overlap
from who_said_that.evaluation import util_components


def match_output(output_df):

    assigned_speaker_list = output_df["assigned_speaker"].unique()
    speaker_match = {}
    speaker_count = {}

    for assigned_speaker in assigned_speaker_list:
        if assigned_speaker == "Unassigned":
            speaker_match[assigned_speaker] = "Unassigned"
        else:
            speaker_match[assigned_speaker] = (
                output_df[output_df["assigned_speaker"] == assigned_speaker]["speaker"]
                .value_counts()
                .idxmax()
            )
            speaker_count[assigned_speaker] = len(
                output_df[output_df["assigned_speaker"] == assigned_speaker]
            )

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


def compute_performance(output_df: pd.DataFrame):

    def speaker_match(speaker, assigned_speaker, matched_speaker):
        if assigned_speaker == "Unassigned":
            return "Missed"
        elif matched_speaker == "Unassigned" or matched_speaker != speaker:
            return "Confusion"
        else:
            return "Correct"

    output_df["matched_result"] = output_df.apply(
        lambda row: speaker_match(
            row["speaker"], row["assigned_speaker"], row["matched_speaker"]
        ),
        axis=1,
    )

    return output_df


def compute_word_diarization_error_rate(ground_truth: pd.DataFrame, diarization_output):

    ground_truth = ground_truth.copy()

    ground_truth["assigned_speaker"] = ground_truth.apply(
        lambda row: (
            util_components.get_word_to_speaker_mapping(
                row["word_start"],
                row["word_end"],
                diarization_output,
            )
            if row["word_end"] > row["word_start"]
            else util_components.get_word_to_speaker_mapping(
                row["word_start"],
                row["word_end"] + 0.01,
                diarization_output,
            )
        ),
        axis=1,
    )

    matched_df = match_output(ground_truth)
    return compute_performance(matched_df)


def compute_word_diarization_error_rate_combined(
    ground_truth: pd.DataFrame, audio_diarization_output, video_diarization_output
):

    final_audio_video_mapping, final_video_audio_mapping = (
        util_components.get_audio_video_mapping(
            final_video_output=video_diarization_output,
            final_audio_output=audio_diarization_output,
        )
    )

    ground_truth["assigned_speaker"] = ground_truth.apply(
        lambda row: util_components.get_speaker_label(
            row["word_start"],
            (
                row["word_end"]
                if row["word_end"] > row["word_start"]
                else row["word_end"] + 0.01
            ),
            video_diarization_output,
            audio_diarization_output,
            final_video_audio_mapping,
            final_audio_video_mapping,
        ),
        axis=1,
    )

    matched_df = match_output(ground_truth)
    return compute_performance(matched_df)


def get_additional_info(
    ground_truth: pd.DataFrame,
    audio_diarization_output,
    video_diarization_output,
    video_name,
    save_path,
):

    ground_truth = ground_truth.copy()

    ground_truth = ground_truth[["word", "word_start", "word_end", "speaker"]]

    for audio_speaker in audio_diarization_output.keys():
        ground_truth[f"audio_{audio_speaker}"] = ground_truth.apply(
            lambda row: find_overlap(
                [
                    (
                        row["word_start"],
                        (
                            row["word_end"]
                            if row["word_end"] > row["word_start"]
                            else row["word_end"] + 0.01
                        ),
                    )
                ],
                audio_diarization_output[audio_speaker],
            )[0],
            axis=1,
        )

    for video_speaker in video_diarization_output.keys():
        ground_truth[f"video_{video_speaker}"] = ground_truth.apply(
            lambda row: find_overlap(
                [
                    (
                        row["word_start"],
                        (
                            row["word_end"]
                            if row["word_end"] > row["word_start"]
                            else row["word_end"] + 0.01
                        ),
                    )
                ],
                video_diarization_output[video_speaker],
            )[0],
            axis=1,
        )

    ground_truth.to_excel(f"{save_path}/{video_name}_additional_info.xlsx", index=False)

    return ground_truth


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
