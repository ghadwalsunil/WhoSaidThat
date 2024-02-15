# Input paramters
VIDEO_FOLDER = "Dataset/Videos"
VIDEO_FILES = [
    # "StarTalk_Sleep_1602_1639"
    # "StarTalk_Cosmic_1050_1130",
    # "StarTalk_Cosmic_2600_2683",
    # "StarTalk_Cosmic_780_850",
    # "StarTalk_Farming_1700_1800",
    # "StarTalk_FlyingVehicles_780_811",
    # "StarTalk_Questions_690_750",
    # "StarTalk_Sleep_1602_1639",
    # "StarTalk_Sleep_1980_2041",
    # "StarTalk_Sleep_2470_2551",
    # "StarTalk_Sleep_748_796",
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
TALKNET_PRETRAIN_MODEL_PATH = "who_said_that/utils/pretrain_TalkSet.model"

# S3FD parameters
S3FD_PATH_WEIGHT = "who_said_that/utils/sfd_face.pth"

# WhisperNet parameters
WHISPER_MODEL_NAME = "medium.en"


# Pretrained pipeline parameters
PRETRAINED_PIPELINE_NAME = "pyannote/speaker-diarization-3.0"
# PRETRAINED_PIPELINE_PARAMS = {
#     "segmentation": {
#         "min_duration_off": 0.0,
#     },
#     "clustering": {
#         "method": "centroid",
#         "min_cluster_size": 12,
#         "threshold": 0.6,
#     },
# }
# PRETRAINED_PIPELINE_PARAMS = {"clustering": {"min_cluster_size": 3}}

PRETRAINED_PIPELINE_PARAMS = None
