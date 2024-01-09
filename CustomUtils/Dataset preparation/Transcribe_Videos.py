from datetime import datetime, timedelta

import pandas as pd
import srt
import whisper
import logging
import os
from stable_whisper import modify_model
logger = logging.getLogger("transcribe")

def get_openai_model(model_name="base"):
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

def generate_openai_transcription(file_name, file_path, model):
    logging.info(f"Generating transcription for file - {file_path}")

    decode_options = dict(language="en")
    transcribe_options = dict(task="transcribe", **decode_options)
    output = model.transcribe(file_path, **transcribe_options)

    transcriptions = {}

    for num, s in enumerate(output.segments):
        transcriptions[num] = []
        for word in s.words:
            transcriptions[num].append(
                {
                    "text": s.text,
                    "segment_start": s.start,
                    "segment_end": s.end,
                    "word": word.word,
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


model_name_openai = "base.en"
video_dir = "../Dataset/Videos/"
model = get_openai_model(model_name_openai)