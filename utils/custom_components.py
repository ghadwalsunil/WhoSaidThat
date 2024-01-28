import os
import logging
from datetime import timedelta
import pandas as pd
import srt
import whisper

from stable_whisper import modify_model
from pyannote.core import Segment, Annotation
from pyannote.core import notebook
import matplotlib.pyplot as plt


def create_annotation_plot(speaker_timelines):
    custom_diarization = Annotation()

    for speaker_key in speaker_timelines.keys():
        for timeline in speaker_timelines[speaker_key]:
            custom_diarization[Segment(timeline[0], timeline[1])] = speaker_key

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plot the custom diarization result
    notebook.plot_annotation(custom_diarization, ax, legend=True)

    # Customize the plot (if needed)
    ax.set_xlabel("Time")
    ax.set_yticks([])  # To hide the y-axis

    # Save the figure
    #     fig.savefig('custom_diarization_plot.png', bbox_inches='tight')

    # Show the figure (optional)
    plt.show()


def get_whisper_model(model_name="base"):
    # initialize model
    logging.info(f"Initializing openai's '{model_name} 'model")
    if model_name in [
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large",
    ]:
        try:
            model = whisper.load_model(model_name)
            # Using the stable whisper to modifiy the model for better timestamps accuracy
            modify_model(model)
            logging.info("Model was successfully initialized")
        except:
            logging.error("Unable to initialize openai model")
            return None
    else:
        logging.error(
            "Model  not found; available models = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']"
        )
        return None

    return model


def get_whisper_result(file_path, model):
    logging.info(f"Generating transcription for file - {file_path}")

    decode_options = dict(language="en")
    transcribe_options = dict(task="transcribe", **decode_options)
    output = model.transcribe(file_path, **transcribe_options)
    output = model.align(file_path, output, language="en")
    return output


def generate_whisper_transcription(file_name, file_path, output):
    logging.info(f"Organizing transcription for file - {file_path}")

    transcriptions = {}

    for num, s in enumerate(output.segments):
        transcriptions[num] = []
        for word in s.words:
            transcriptions[num].append(
                {
                    "text": s.text.strip(),
                    "segment_start": s.start,
                    "segment_end": s.end,
                    "word": word.word.strip(),
                    "word_start": word.start,
                    "word_end": word.end,
                }
            )

    rows = []

    for key, words in transcriptions.items():
        for word in words:
            row = {
                "file_name": file_name,
                "segment_id": key,
                "segment_text": word["text"],
                "segment_start": word["segment_start"],
                "segment_end": word["segment_end"],
                "word": word["word"],
                "word_start": word["word_start"],
                "word_end": word["word_end"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    return df


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
    percentage_overlap1 = (overlap / total_duration1) * 100

    # Calculate the percentage of overlap with respect to intervals2
    percentage_overlap2 = (overlap / total_duration2) * 100

    return percentage_overlap1, percentage_overlap2


def get_mapping(diarization_result_1, diarization_result_2):
    result_mapping = {}
    unknown_speaker_count = 0
    for speaker_result_2 in diarization_result_2.keys():
        max_overlap = 0
        max_overlap_speaker = None
        for speaker_result_1 in diarization_result_1.keys():
            result_overlap, _ = find_overlap(
                diarization_result_1[speaker_result_1], diarization_result_2[speaker_result_2]
            )
            if result_overlap > max_overlap:
                max_overlap = result_overlap
                max_overlap_speaker = speaker_result_1

        if max_overlap_speaker is None:
            result_mapping[speaker_result_2] = f"Unknown_{unknown_speaker_count}"
            unknown_speaker_count += 1
        else:
            result_mapping[speaker_result_2] = max_overlap_speaker

    return result_mapping


def get_segment_to_speaker_mapping(segment_start, segment_end, speaker_mapping, audio_output):
    final_speaker = None
    max_overlap = 0

    for audio_speaker in audio_output.keys():
        segment_overlap, _ = find_overlap([(segment_start, segment_end)], audio_output[audio_speaker])
        if segment_overlap > max_overlap:
            final_speaker = audio_speaker
            max_overlap = segment_overlap

    if final_speaker is None:
        return "Unknown", max_overlap
    else:
        if final_speaker in speaker_mapping.keys():
            return speaker_mapping[final_speaker], max_overlap
        else:
            return final_speaker, max_overlap


def _get_word_to_speaker_mapping(word_start, word_end, diarization_output):
    for audio_speaker, speaker_timeline in diarization_output.items():
        segment_overlap, _ = find_overlap([(word_start, word_end)], speaker_timeline)
        if segment_overlap > 0.5:
            return audio_speaker

    return "Unknown"


def get_segments_and_speaker(df, diarization_output, video_name, output_type):
    temp_df = df[df["video_name"] == video_name].copy()
    # temp_df["word_end"] = temp_df.apply(
    #     lambda row: (row["word_end"] + 0.1) if row["word_end"] == row["word_start"] else row["word_end"], axis=1
    # )

    temp_df[f"{output_type}_assigned_speaker"] = temp_df.apply(
        lambda row: _get_word_to_speaker_mapping(row["word_start"], row["word_end"], diarization_output), axis=1
    )

    # temp_df = (
    #     temp_df[["segment_id", "speaker", "word", "word_start", "word_end"]]
    #     .groupby(["segment_id", "speaker"], as_index=False)
    #     .agg({"word": " ".join, "word_start": min, "word_end": max})
    # )

    # temp_df["segment_id"] = temp_df.index + 1

    # temp_df["video_name"] = video_name

    return temp_df


def match_output(output_df, output_type):
    video_list = output_df["video_name"].unique()
    assigned_speaker_match = {}

    for video_name in video_list:
        temp_df = output_df[output_df["video_name"] == video_name]

        assigned_speaker_list = temp_df[f"{output_type}_assigned_speaker"].unique()
        video_speaker_match = {}
        speaker_count = {}

        for assigned_speaker in assigned_speaker_list:
            if assigned_speaker == "Unknown":
                video_speaker_match[assigned_speaker] = "Unknown"
            else:
                video_speaker_match[assigned_speaker] = (
                    temp_df[temp_df[f"{output_type}_assigned_speaker"] == assigned_speaker]["speaker"]
                    .value_counts()
                    .idxmax()
                )
                speaker_count[assigned_speaker] = len(temp_df[temp_df[f"{output_type}_assigned_speaker"] == assigned_speaker])

        temp_count = {}

        for speaker_pred, speaker_actual in video_speaker_match.items():
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
                            video_speaker_match[max_speaker] = "Unknown"
                            max_speaker = speaker
                            max_count = speaker_count[speaker]
                    else:
                        video_speaker_match[speaker] = "Unknown"
                video_speaker_match[max_speaker] = speaker_actual


        assigned_speaker_match[video_name] = video_speaker_match

    output_df[f"{output_type}_matched_speaker"] = output_df.apply(
        lambda row: assigned_speaker_match[row["video_name"]][row[f"{output_type}_assigned_speaker"]], axis=1
    )

    def speaker_match(speaker, assigned_speaker):
        if speaker == assigned_speaker:
            return "Match_True"
        elif assigned_speaker == "Unknown":
            return "Unassigned"
        else:
            return "Match_False"

    output_df[f"{output_type}_matched_result"] = output_df.apply(
        lambda row: speaker_match(row["speaker"], row[f"{output_type}_matched_speaker"]), axis=1
    )

    return output_df


def compute_performance(output_df, output_type):
    output_df = (
        output_df.groupby(by=["video_name", f"{output_type}_matched_result"]).count()[["segment_id"]].reset_index()
    )
    output_df.rename(columns={"segment_id": "count"}, inplace=True)

    df_pivot = output_df.pivot(
        index="video_name", columns=f"{output_type}_matched_result", values="count"
    ).reset_index()

    if "Match_True" not in df_pivot.columns:
        df_pivot["Match_True"] = 0
    else:
        df_pivot["Match_True"] = df_pivot["Match_True"].fillna(0)

    if "Match_False" not in df_pivot.columns:
        df_pivot["Match_False"] = 0
    else:
        df_pivot["Match_False"] = df_pivot["Match_False"].fillna(0)

    if "Unassigned" not in df_pivot.columns:
        df_pivot["Unassigned"] = 0
    else:
        df_pivot["Unassigned"] = df_pivot["Unassigned"].fillna(0)

    # Calculate accuracy
    df_pivot[f"{output_type}_accuracy"] = (
        df_pivot["Match_True"] / (df_pivot["Match_True"] + df_pivot["Match_False"] + df_pivot["Unassigned"])
    ) * 100
    df_pivot[f"{output_type}_unassigned"] = (
        df_pivot["Unassigned"] / (df_pivot["Match_True"] + df_pivot["Match_False"] + df_pivot["Unassigned"])
    ) * 100

    # Rename columns
    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(
        columns={
            "Match_True": f"{output_type}_true_count",
            "Match_False": f"{output_type}_false_count",
            "Unassigned": f"{output_type}_unassigned_count",
        }
    )

    return df_pivot


def create_srt(df, video_name, video_dir, output_type):
    temp_df = df[df["video_name"] == video_name].copy()

    temp_df.reset_index(drop=True, inplace=True)

    temp_df = (
        temp_df[["segment_id", f"{output_type}_matched_speaker", "word", "word_start", "word_end"]]
        .groupby(["segment_id", f"{output_type}_matched_speaker"], as_index=False)
        .agg({"word": " ".join, "word_start": min, "word_end": max})
    )

    temp_df["segment_id"] = temp_df.index + 1

    srt_list = temp_df.apply(
        lambda row: srt.Subtitle(
            index=row["segment_id"],
            start=timedelta(seconds=row["word_start"] if row["word_start"] < 0.1 else row["word_start"] - 0.1),
            end=timedelta(seconds=row["word_end"] + 0.1),
            content=row[f"{output_type}_matched_speaker"] + " : " + row["word"],
        ),
        axis=1,
    ).to_list()

    srt_string = srt.compose(srt_list)
    with open(
        os.path.join(
            video_dir,
            f"{os.path.splitext(video_name)[0]}_{output_type}_predicted.srt",
        ),
        "w",
    ) as f:
        f.write(srt_string)


def get_word_to_speaker_mapping(word_start, word_end, final_output):
    final_speaker = None
    max_overlap = 0

    for speaker_key in final_output.keys():
        word_overlap, _ = find_overlap([(word_start, word_end)], final_output[speaker_key])
        if word_overlap > max_overlap:
            final_speaker = speaker_key
            max_overlap = word_overlap

    if final_speaker is None:
        final_speaker = "Unknown"

    return final_speaker


def get_speaker_label(
    video_name,
    word_start,
    word_end,
    final_video_output,
    final_audio_output,
    final_video_audio_mapping,
    final_audio_video_mapping,
):
    audio_speaker = get_word_to_speaker_mapping(word_start, word_end, final_audio_output[video_name])
    video_speaker = get_word_to_speaker_mapping(word_start, word_end, final_video_output[video_name])

    if video_speaker is "Unknown":
        if audio_speaker is "Unknown":
            return "Unknown"
        else:
            return final_audio_video_mapping[video_name][audio_speaker]
    else:
        return video_speaker
