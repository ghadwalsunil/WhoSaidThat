def get_segments_and_words(
    df,
    speaker_column=None,
    max_length=50,
):
    """
    This function takes a dataframe with columns "segment_id", "word", "word_start", and "word_end" and returns a list of segments and a list of words. The function also takes an optional speaker_column parameter to group words by speaker.
    Args:
        df (pandas.DataFrame): A dataframe with columns "segment_id", "word", "word_start", and "word_end".
        speaker_column (str, optional): The name of the column that contains the speaker information. Defaults to None.
        max_length (int, optional): The maximum length of a segment. Defaults to 50.
    Returns:
        list: A list of segments.
        list: A list of words.
    """

    segment_id_list = df["segment_id"].drop_duplicates().tolist()

    if speaker_column is not None:
        if speaker_column not in df.columns:
            speaker_column = None

    final_segment_list = []
    final_word_list = []
    new_segment_id = 0

    for segment_id in segment_id_list:
        word_df = df[df["segment_id"] == segment_id]
        words = word_df["word"].tolist()
        word_starts = word_df["word_start"].tolist()
        word_ends = word_df["word_end"].tolist()
        if speaker_column is not None:
            speakers = word_df[speaker_column].tolist()

        current_segment = []
        previous_word_start = -1
        previous_word_end = -1
        current_word_list = []
        if speaker_column is not None:
            previous_speaker = None
        for idx, word in enumerate(words):
            if not current_segment:
                current_segment.append(word)
                previous_word_start = word_starts[idx]
                previous_word_end = word_ends[idx]
                word_dict = {
                    "word_id": idx,
                    "word": word,
                    "word_start": word_starts[idx],
                    "word_end": word_ends[idx],
                }
                if speaker_column is not None:
                    word_dict["speaker"] = speakers[idx]
                    previous_speaker = speakers[idx]
                current_word_list.append(word_dict)
            elif (speaker_column is not None and speakers[idx] != previous_speaker) or (
                len(" ".join(current_segment)) + len(word)
            ) > max_length:
                new_segment_id += 1
                segment_dict = {
                    "segment_id": new_segment_id,
                    "segment_start": previous_word_start,
                    "segment_end": previous_word_end,
                    "text": " ".join(current_segment),
                }
                if speaker_column is not None:
                    segment_dict["speaker"] = previous_speaker
                final_segment_list.append(segment_dict)
                final_word_list.append(current_word_list)
                current_segment = []
                current_segment.append(word)
                current_word_list = []
                current_word_list.append(
                    {
                        "word_id": idx,
                        "word": word,
                        "word_start": word_starts[idx],
                        "word_end": word_ends[idx],
                    }
                )
                previous_word_start = word_starts[idx]
                previous_word_end = word_ends[idx]
                if speaker_column is not None:
                    previous_speaker = speakers[idx]
            else:
                current_segment.append(word)
                previous_word_end = word_ends[idx]
                current_word_list.append(
                    {
                        "word_id": idx,
                        "word": word,
                        "word_start": word_starts[idx],
                        "word_end": word_ends[idx],
                    }
                )

        if current_segment:
            new_segment_id += 1
            segment_dict = {
                "segment_id": new_segment_id,
                "segment_start": previous_word_start,
                "segment_end": previous_word_end,
                "text": " ".join(current_segment),
            }
            if speaker_column is not None:
                segment_dict["speaker"] = previous_speaker
            final_segment_list.append(segment_dict)
            final_word_list.append(current_word_list)

    return final_segment_list, final_word_list
