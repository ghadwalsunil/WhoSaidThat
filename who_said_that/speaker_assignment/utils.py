import os, pickle

import pandas as pd

from who_said_that.speaker_assignment import util_components


def get_video_stats(
    final_df: pd.DataFrame,
    final_video_output: dict,
    final_audio_output: dict,
    video_files: list,
):

    final_df["video_name"] = final_df["file_name"].apply(lambda x: os.path.splitext(x)[0])
    df_videos = pd.DataFrame({"video_name": video_files})
    final_df = final_df.merge(df_videos, how="inner")
    final_df["word_end_modified"] = final_df.apply(
        lambda row: ((row["word_end"] + 0.01) if row["word_start"] == row["word_end"] else row["word_end"]),
        axis=1,
    )
    final_df["predicted_speakers_video"] = final_df["video_name"].apply(lambda x: len(final_video_output[x].keys()))
    final_df["predicted_speakers_audio"] = final_df["video_name"].apply(lambda x: len(final_audio_output[x].keys()))

    return final_df


def get_audio_video_mapping(final_video_output, final_audio_output, run_output_folder):
    final_audio_video_mapping = {}
    final_video_audio_mapping = {}

    for file_name, video_output in final_video_output.items():
        final_audio_video_mapping[file_name] = util_components.get_mapping(
            final_video_output[file_name], final_audio_output[file_name]
        )

    for file_name, video_output in final_video_output.items():
        final_video_audio_mapping[file_name] = util_components.get_mapping(
            final_audio_output[file_name], final_video_output[file_name]
        )

    with open(os.path.join(run_output_folder, "final_audio_video_mapping.pckl"), "wb") as f:
        pickle.dump(final_audio_video_mapping, f)

    with open(os.path.join(run_output_folder, "final_video_audio_mapping.pckl"), "wb") as f:
        pickle.dump(final_video_audio_mapping, f)

    return final_audio_video_mapping, final_video_audio_mapping


def assign_speakers(
    final_df,
    final_audio_output,
    final_video_output,
    run_output_folder,
    final_video_audio_mapping,
    final_audio_video_mapping,
):

    final_df["a_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_word_to_speaker_mapping(
            row["word_start"],
            row["word_end_modified"],
            final_audio_output[row["video_name"]],
        ),
        axis=1,
    )

    final_df["v_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_word_to_speaker_mapping(
            row["word_start"],
            row["word_end_modified"],
            final_video_output[row["video_name"]],
        ),
        axis=1,
    )

    final_df["av_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_speaker_label(
            row["video_name"],
            row["word_start"],
            row["word_end_modified"],
            final_video_output,
            final_audio_output,
            final_video_audio_mapping,
            final_audio_video_mapping,
        ),
        axis=1,
    )
    output_file_name = "result_audio_video_pre_match.xlsx"

    final_df.to_excel(os.path.join(run_output_folder, output_file_name), index=False)

    return final_df


def get_stats(final_df, run_output_folder):

    output_file_name = "result_audio_video.xlsx"

    final_df = util_components.match_output(final_df, "a")
    final_df = util_components.match_output(final_df, "v")
    final_df = util_components.match_output(final_df, "av")

    final_df.to_excel(os.path.join(run_output_folder, output_file_name), index=False)

    stats_a = util_components.compute_performance(final_df, "a")
    stats_v = util_components.compute_performance(final_df, "v")
    stats_av = util_components.compute_performance(final_df, "av")

    stats = stats_a.merge(stats_v, on="video_name", how="left").merge(stats_av, on="video_name", how="left")

    temp_df = final_df[["video_name", "predicted_speakers_audio", "predicted_speakers_video"]].drop_duplicates()

    stats = stats.merge(temp_df, on="video_name", how="left")

    stats.to_excel(os.path.join(run_output_folder, "stats_" + output_file_name), index=False)

    return final_df, stats
