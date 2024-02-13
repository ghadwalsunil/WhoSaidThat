import os
import sys

import params
from assign_speakers.assign_speakers import AssignSpeakers
from audio_diarization.audio_diarization import AudioDiarization
from create_subtitles.create_subtitles import CreateSubtitles
from dotenv import load_dotenv
from models.models import Models
from preprocess.preprocess import Preprocess
from talkNetASD.talkNetASD import TalkNetASD
from video_diarization.video_diarization import VideoDiarization

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup parameters
models = Models(
    load_talknet_model=True,
    load_pretrained_pipeline=True,
    load_whisper_model=True,
)

talknet_model = models.get_talknet_model(params.TALKNET_PRETRAIN_MODEL_PATH)
pretrained_pipeline = models.get_pretrained_pipeline(params.PRETRAINED_PIPELINE_NAME)
# # whisper_model = models.get_whisper_model(model_name=params.WHISPER_MODEL_NAME)

# Preprocess
preprocess = Preprocess(
    video_files=params.VIDEO_FILES,
    video_folder=params.VIDEO_FOLDER,
    run_output_folder=params.RUN_OUTPUT_FOLDER,
    video_output_folder=params.VIDEO_OUTPUT_FOLDER,
)

preprocess.perform_preprocessing()

# # TalkNetASD
talkNetASD = TalkNetASD(
    video_files=params.VIDEO_FILES,
    video_folder=params.VIDEO_FOLDER,
    run_output_folder=params.RUN_OUTPUT_FOLDER,
    video_output_folder=params.VIDEO_OUTPUT_FOLDER,
    talkNetModel=talknet_model,
    generate_visualization=False,
)

talkNetASD.perform_talkNetASD()

# Video Diarization
videoDiarization = VideoDiarization(
    video_files=params.VIDEO_FILES,
    video_folder=params.VIDEO_FOLDER,
    run_output_folder=params.RUN_OUTPUT_FOLDER,
    video_output_folder=params.VIDEO_OUTPUT_FOLDER,
)

videoDiarization.perform_video_diarization()

# # Audio Diarization
audioDiarization = AudioDiarization(
    video_files=params.VIDEO_FILES,
    video_folder=params.VIDEO_FOLDER,
    run_output_folder=params.RUN_OUTPUT_FOLDER,
    video_output_folder=params.VIDEO_OUTPUT_FOLDER,
    pretrained_pipeline=pretrained_pipeline,
)

audioDiarization.perform_audio_diarization()

# Assign speakers to transcriptions
assign_speakers = AssignSpeakers(
    transcriptions_path=params.TRANSCRIPTION_DF_PATH,
    run_output_path=params.RUN_OUTPUT_FOLDER,
    video_files=params.VIDEO_FILES,
    video_folder=params.VIDEO_FOLDER,
)

assign_speakers.perform_speaker_assignment()

# # Create srt and js files
create_subtitles = CreateSubtitles(
    run_output_path=params.RUN_OUTPUT_FOLDER,
    video_files=params.VIDEO_FILES,
    video_folder=params.VIDEO_FOLDER,
)
create_subtitles.create_subtitles()
