import json
import os
from datetime import timedelta

import srt
from create_subtitles import util_components


def generate_subtitles(
    final_df, output_type=["srt", "js"], srt_output_path=None, js_output_path=None
):

    video_names = final_df["video_name"].unique()
    speaker_cols = {
        "av_matched_speaker": "av",
        "a_matched_speaker": "a",
        "v_matched_speaker": "v",
    }

    for video_name in video_names:
        temp_df = final_df[final_df["video_name"] == video_name].copy()
        temp_df.reset_index(drop=True, inplace=True)

        for speaker_column in speaker_cols.keys():
            segment_lst, word_lst = util_components.get_segments_and_words(
                temp_df, speaker_column
            )

            if "srt" in output_type:
                srt_list = []

                for segment in segment_lst:
                    if "speaker" in segment.keys():
                        segment_content = segment["speaker"] + " : " + segment["text"]
                    else:
                        segment_content = segment["text"]
                    srt_list.append(
                        srt.Subtitle(
                            index=segment["segment_id"],
                            start=timedelta(
                                seconds=(
                                    segment["segment_start"]
                                    if segment["segment_start"] < 0.1
                                    else segment["segment_start"] - 0.1
                                )
                            ),
                            end=timedelta(seconds=segment["segment_end"] + 0.1),
                            content=segment_content,
                        )
                    )

                srt_string = srt.compose(srt_list)
                with open(
                    os.path.join(
                        srt_output_path,
                        video_name + "_" + speaker_cols[speaker_column] + ".srt",
                    ),
                    "w",
                ) as f:
                    f.write(srt_string)

            if "js" in output_type:

                with open(
                    os.path.join(
                        js_output_path,
                        video_name
                        + "_"
                        + speaker_cols[speaker_column]
                        + "_"
                        + "subtitles.js",
                    ),
                    "w",
                ) as f:
                    f.write("const subtitles = ")
                    f.write(json.dumps(segment_lst, indent=4))
                    f.write(";\n")

                with open(
                    os.path.join(
                        js_output_path,
                        video_name
                        + "_"
                        + speaker_cols[speaker_column]
                        + "_"
                        + "wordTimestamps.js",
                    ),
                    "w",
                ) as f:
                    f.write("const wordTimestamps = ")
                    f.write(json.dumps(word_lst, indent=4))
                    f.write(";\n")
