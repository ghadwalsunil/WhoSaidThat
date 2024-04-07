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


def get_mapping(diarization_result_1, diarization_result_2):
    result_mapping = {}
    unknown_speaker_count = 0
    for speaker_result_2 in diarization_result_2.keys():
        max_overlap = 0
        max_overlap_speaker = None
        for speaker_result_1 in diarization_result_1.keys():
            result_overlap, _ = find_overlap(
                diarization_result_1[speaker_result_1],
                diarization_result_2[speaker_result_2],
            )
            if result_overlap > max_overlap:
                max_overlap = result_overlap
                max_overlap_speaker = speaker_result_1

        if max_overlap_speaker is None or max_overlap < 20:
            result_mapping[speaker_result_2] = f"Unknown_{unknown_speaker_count}"
            unknown_speaker_count += 1
        else:
            result_mapping[speaker_result_2] = max_overlap_speaker

    return result_mapping


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

    if video_speaker == "Unknown":
        if audio_speaker == "Unknown":
            return "Unknown"
        else:
            return final_audio_video_mapping[video_name][audio_speaker]
    else:
        return video_speaker


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
                speaker_count[assigned_speaker] = len(
                    temp_df[temp_df[f"{output_type}_assigned_speaker"] == assigned_speaker]
                )

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
        lambda row: assigned_speaker_match[row["video_name"]][row[f"{output_type}_assigned_speaker"]],
        axis=1,
    )

    return output_df


def compute_performance(output_df, output_type):

    def speaker_match(speaker, assigned_speaker):
        if speaker == assigned_speaker:
            return "Match_True"
        elif assigned_speaker == "Unknown":
            return "Unassigned"
        else:
            return "Match_False"

    output_df[f"{output_type}_matched_result"] = output_df.apply(
        lambda row: speaker_match(row["speaker"], row[f"{output_type}_matched_speaker"]),
        axis=1,
    )

    output_df = (
        output_df.groupby(by=["video_name", f"{output_type}_matched_result"]).count()[["segment_id"]].reset_index()
    )
    output_df.rename(columns={"segment_id": "count"}, inplace=True)

    df_pivot = output_df.pivot(
        index="video_name",
        columns=f"{output_type}_matched_result",
        values="count",
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
