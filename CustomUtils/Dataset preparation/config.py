import logging
import os

MODEL_TYPE = "openai"  # ['openai']
MODEL_NAME_OPENAI = (
    "tiny.en"  # ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']
)

SOURCE_FOLDER_PATH = os.path.join("Clips")  # path of the folder containing audio/video files
TRANSCRIPTION_OUTPUT_PATH = os.path.join("Clips", "transcription_output")  # path to store the transcriptions

# Supported Audio/Video formats
SUPPORTED_FROMATS = [".webm", ".wav", ".mp4", ".mkv"]

# Transcription Language code, examples ['en', 'de']
# Set None for auto detection
TRANSCRIPTION_LANGUAGE = "en"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CREATE_SRT = True
CREATE_EXCEL = False