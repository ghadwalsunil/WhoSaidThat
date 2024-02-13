# Input paramters
VIDEO_FOLDER = "Dataset/Videos"
VIDEO_FILES = [
    "MagnusCarlson_542_599",
    "NDT_India_19_88",
    "StarTalk_CMBR_92_152",
    "StarTalk_FlyingVehicles_300_340",
    "StarTalk_Sleep_1980_2041",
]
TRANSCRIPTION_DF_PATH = "Dataset/All_Transcriptions_WithSpeakers.xlsx"

# Output parameters
VIDEO_OUTPUT_FOLDER = "output/video_temp"
RUN_OUTPUT_FOLDER = "output/run_output"
PYAVI_FOLDER_NAME = "pyavi"
PYFRAMES_FOLDER_NAME = "pyframes"
PYWORK_FOLDER_NAME = "pywork"
PYCROP_FOLDER_NAME = "pycrop"
SRT_OUTPUT_FOLDER = "output/run_output/srt"
JS_OUTPUT_FOLDER = "output/run_output/js"
PLOT_OUTPUT_FOLDER = "output/run_output/plots"

# TalkNet parameters
TALKNET_PRETRAIN_MODEL_PATH = "pretrain_TalkSet.model"

# WhisperNet parameters
WHISPER_MODEL_NAME = "medium.en"


# Pretrained pipeline parameters
PRETRAINED_PIPELINE_NAME = "pyannote/speaker-diarization-3.0"
