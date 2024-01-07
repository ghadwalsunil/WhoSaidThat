import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import srt
import whisper

import config

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


def generate_openai_transcription(file_path, model):
    logging.info(f"Generating transcription for file - {file_path}")

    if config.TRANSCRIPTION_LANGUAGE is None:
        output = model.transcribe(file_path)
    else:
        decode_options = dict(language=config.TRANSCRIPTION_LANGUAGE)
        transcribe_options = dict(task="transcribe", **decode_options)
        output = model.transcribe(file_path, **transcribe_options)

    transcription = []
    transcription_list = []

    for num, s in enumerate(output.segments):
        if config.CREATE_SRT:
            transcription.append(
                srt.Subtitle(
                    index=num + 1,
                    start=timedelta(seconds=s.start),
                    end=timedelta(seconds=s.end + 0.1),
                    content=s.text.strip(),
                )
            )
        if config.CREATE_EXCEL:
            transcription_list.append([s.text.strip(), s.start, s.end])

    return transcription, transcription_list


def generate_transcriptions_from_files(
    model_type="openai",
    model_name_openai="base",
    source_folder_path="",
):
    if not os.path.exists(source_folder_path):
        logging.error("Audio/Video folder path does not exist")
        return None

    if model_type == "openai":
        model = get_openai_model(model_name_openai)
    else:
        logging.error("Unable to get desired model")

    # Get audio/video files from the folder
    files = [
        f
        for f in os.listdir(source_folder_path)
        if os.path.isfile(os.path.join(source_folder_path, f)) and os.path.splitext(f)[1] in config.SUPPORTED_FROMATS
    ]

    if len(files) == 0:
        logging.warn("No files to read in folder in the source folder")
        return None

    # Generate transcriptions
    transcriptions = {}
    t_list = {}
    for f in files:
        transcriptions[f], t_list[f] = generate_openai_transcription(os.path.join(source_folder_path, f), model)

    if not os.path.isdir(config.TRANSCRIPTION_OUTPUT_PATH):
        os.mkdir(config.TRANSCRIPTION_OUTPUT_PATH)

    for key in transcriptions.keys():
        if config.CREATE_EXCEL:
            df = pd.DataFrame(t_list[key], columns=["Text", "Start time", "End time"])
            df.to_excel(
                os.path.join(
                    config.TRANSCRIPTION_OUTPUT_PATH,
                    key + "_" + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".xlsx",
                ),
                index=False,
            )
        if config.CREATE_SRT:
            srt_string = srt.compose(transcriptions[key])
            with open(
                os.path.join(
                    config.TRANSCRIPTION_OUTPUT_PATH,
                    key.split(".")[0] + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".srt",
                ),
                "w",
            ) as f:
                f.write(srt_string)

    logging.info("Transcription completed")

    return transcriptions


model_type = config.MODEL_TYPE
model_name_openai = config.MODEL_NAME_OPENAI
source_folder_path = config.SOURCE_FOLDER_PATH

if (
    generate_transcriptions_from_files(
        model_type=model_type,
        model_name_openai=model_name_openai,
        source_folder_path=source_folder_path,
    )
    is None
):
    print("Failed to generate transcriptions")
