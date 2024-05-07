from who_said_that.utils.components import find_overlap


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


def get_audio_video_mapping(final_video_output, final_audio_output):

    final_audio_video_mapping = get_mapping(final_video_output, final_audio_output)

    final_video_audio_mapping = get_mapping(final_audio_output, final_video_output)

    return final_audio_video_mapping, final_video_audio_mapping


def get_word_to_speaker_mapping(word_start, word_end, final_output):
    final_speaker = None
    max_overlap = 0

    for speaker_key in final_output.keys():
        word_overlap, _ = find_overlap(
            [(word_start, word_end)], final_output[speaker_key]
        )
        if word_overlap > max_overlap:
            final_speaker = speaker_key
            max_overlap = word_overlap

    if final_speaker is None:
        final_speaker = "Unassigned"

    return final_speaker


def get_speaker_label(
    word_start,
    word_end,
    final_video_output,
    final_audio_output,
    final_video_audio_mapping,
    final_audio_video_mapping,
):
    audio_speaker = get_word_to_speaker_mapping(
        word_start, word_end, final_audio_output
    )
    video_speaker = get_word_to_speaker_mapping(
        word_start, word_end, final_video_output
    )

    if video_speaker == "Unassigned":
        if audio_speaker == "Unassigned":
            return "Unassigned"
        else:
            return final_audio_video_mapping[audio_speaker]
    else:
        return video_speaker
