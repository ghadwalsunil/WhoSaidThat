import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from who_said_that import params
from who_said_that.audio_diarization.diarize_audio import AudioDiarization
from who_said_that.models.get_models import Models
from who_said_that.preprocess.preprocess_input import Preprocess
from who_said_that.speaker_assignment.assign_speakers import AssignSpeakers
from who_said_that.subtitle_creation.create_subtitles import CreateSubtitles
from who_said_that.talkNetASD.perform_talkNetASD import TalkNetASD
from who_said_that.video_diarization.diarize_video import VideoDiarization

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Setup parameters
models = Models(
    load_talknet_model=True,
    load_pretrained_pipeline=True,
    load_whisper_model=True,
)

talknet_model = models.get_talknet_model(params.TALKNET_PRETRAIN_MODEL_PATH)
pretrained_pipeline = models.get_pretrained_pipeline(params.PRETRAINED_PIPELINE_NAME)
whisper_model = models.get_whisper_model(model_name=params.WHISPER_MODEL_NAME)

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

# Audio Diarization
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

# Create srt and js files
create_subtitles = CreateSubtitles(
    run_output_path=params.RUN_OUTPUT_FOLDER,
    video_files=params.VIDEO_FILES,
    video_folder=params.VIDEO_FOLDER,
)
create_subtitles.create_subtitles()
