import os

TRANSCRIPTION_DF_PATH = "Dataset/All_Transcriptions_WithSpeakers.xlsx"

# Output parameters
PARENT_DIR = "/home/sunil/projects/Stuff/Combined/WhoSaidThat"
# PARENT_DIR = "/vol3/sunil/"
VIDEO_OUTPUT_FOLDER = os.path.join(PARENT_DIR, "output", "video_temp")
RUN_OUTPUT_FOLDER = os.path.join(PARENT_DIR, "output", "run_output")
PYAVI_FOLDER_NAME = "pyavi"
PYWAV_FOLDER_NAME = "pywav"
PYFRAMES_FOLDER_NAME = "pyframes"
PYWORK_FOLDER_NAME = "pywork"
PYCROP_FOLDER_NAME = "pycrop"
SRT_OUTPUT_FOLDER = os.path.join(RUN_OUTPUT_FOLDER, "srt")
JS_OUTPUT_FOLDER = os.path.join(RUN_OUTPUT_FOLDER, "js")
PLOT_OUTPUT_FOLDER = os.path.join(RUN_OUTPUT_FOLDER, "plots")
SPEAKER_STATS_FOLDER = os.path.join(RUN_OUTPUT_FOLDER, "speaker_stats")

# TalkNet parameters
TALKNET_PRETRAIN_MODEL_PATH = "who_said_that/utils/pretrain_TalkSet.model"
# who_said_that/utils/pretrain_TalkSet.model
# who_said_that/utils/msdwild.pretrained.model
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
