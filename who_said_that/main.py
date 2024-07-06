import os
import pickle
import sys
from datetime import datetime
from shutil import rmtree

import pandas as pd
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from who_said_that import params, video_list
from who_said_that.audio_diarization.diarize_audio import AudioDiarization
from who_said_that.classes import DiarizationOutput
from who_said_that.evaluation.evaluate_results import DiarizationEvaluateResults
from who_said_that.models.get_models import Models
from who_said_that.preprocess.preprocess_input import Preprocess

from who_said_that.speaker_assignment.assign_speakers import AssignSpeakers

# from who_said_that.subtitle_creation.create_subtitles import CreateSubtitles
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
pretrained_pipeline_12 = models.get_pretrained_pipeline(params.PRETRAINED_PIPELINE_NAME, min_cluster_size=12)
# whisper_model = models.get_whisper_model(model_name=params.WHISPER_MODEL_NAME)

if not os.path.exists(params.RUN_OUTPUT_FOLDER):
    os.mkdir(params.RUN_OUTPUT_FOLDER)
    os.mkdir(params.SRT_OUTPUT_FOLDER)
    os.mkdir(params.JS_OUTPUT_FOLDER)
    os.mkdir(params.PLOT_OUTPUT_FOLDER)
    os.mkdir(params.SPEAKER_STATS_FOLDER)

final_output = {}
video_durations = {}

# run_timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
# run_timestamp = "TEST_NDT"
run_timestamp = "Test"
# final_output = pickle.load(
#     open(os.path.join(params.RUN_OUTPUT_FOLDER, f"final_diarization_output_{run_timestamp}.pckl"), "rb")
# )

for video_file in video_list.VIDEO_FILES:
    final_output[video_file.save_name] = {}
    preprocess = Preprocess(
        video_file=video_file,
        run_output_folder=params.RUN_OUTPUT_FOLDER,
        video_output_folder=params.VIDEO_OUTPUT_FOLDER,
    )

    if not preprocess.perform_preprocessing():
        continue
    audioFilePath = os.path.join(
        params.VIDEO_OUTPUT_FOLDER,
        video_file.save_name,
        params.PYWAV_FOLDER_NAME,
        "audio.wav",
    )
    video_durations[video_file.save_name] = len(AudioSegment.from_file(audioFilePath)) / 1000

    talkNetASD = TalkNetASD(
        video_file=video_file,
        run_output_folder=params.RUN_OUTPUT_FOLDER,
        video_output_folder=params.VIDEO_OUTPUT_FOLDER,
        talkNetModel=talknet_model,
        generate_visualization=not (video_file.del_intermediate_files),
        saveMarkedFrames=not (video_file.del_intermediate_files),
    )

    talkNetASD.perform_talkNetASD()

    videoDiarization = VideoDiarization(
        video_file=video_file,
        video_output_folder=params.VIDEO_OUTPUT_FOLDER,
    )

    final_output[video_file.save_name]["video_simple"] = videoDiarization.perform_video_diarization(
        videoDuration=video_durations[video_file.save_name]
    )

    audioDiarization = AudioDiarization(
        video_file=video_file,
        video_output_folder=params.VIDEO_OUTPUT_FOLDER,
    )

    final_output[video_file.save_name]["audio_12"] = audioDiarization.perform_audio_diarization(
        pretrained_pipeline=pretrained_pipeline_12,
        pipeline_name="12",
        videoDuration=video_durations[video_file.save_name],
    )

    if video_file.del_intermediate_files:
        if os.path.exists(
            os.path.join(
                params.VIDEO_OUTPUT_FOLDER,
                video_file.save_name,
                params.PYFRAMES_FOLDER_NAME,
            )
        ):
            rmtree(
                os.path.join(
                    params.VIDEO_OUTPUT_FOLDER,
                    video_file.save_name,
                    params.PYFRAMES_FOLDER_NAME,
                )
            )
        if os.path.exists(
            os.path.join(
                params.VIDEO_OUTPUT_FOLDER,
                video_file.save_name,
                params.PYCROP_FOLDER_NAME,
            )
        ):
            rmtree(
                os.path.join(
                    params.VIDEO_OUTPUT_FOLDER,
                    video_file.save_name,
                    params.PYCROP_FOLDER_NAME,
                )
            )
        if os.path.exists(
            os.path.join(
                params.VIDEO_OUTPUT_FOLDER,
                video_file.save_name,
                params.PYAVI_FOLDER_NAME,
            )
        ):
            rmtree(
                os.path.join(
                    params.VIDEO_OUTPUT_FOLDER,
                    video_file.save_name,
                    params.PYAVI_FOLDER_NAME,
                )
            )


with open(
    os.path.join(params.RUN_OUTPUT_FOLDER, f"final_diarization_output_{run_timestamp}.pckl"),
    "wb",
) as f:
    pickle.dump(final_output, f)

# Combine results
# final_output = pickle.load(
#     open(
#         os.path.join(params.RUN_OUTPUT_FOLDER, f"final_diarization_output_{run_timestamp}.pckl"),
#         "rb",
#     )
# )

# for video_file in video_list.VIDEO_FILES:
#     audioFilePath = os.path.join(
#         params.VIDEO_OUTPUT_FOLDER,
#         video_file.save_name,
#         params.PYWAV_FOLDER_NAME,
#         "audio.wav",
#     )
#     video_durations[video_file.save_name] = len(AudioSegment.from_file(audioFilePath)) / 1000
#     if video_file.save_name not in final_output.keys() or len(final_output[video_file.save_name]) < 6:
#         sys.stderr.write(f"Video {video_file.save_name} not found in final_output\n")
#         continue
#     combined_output_simple = DiarizationOutput(
#         audio_diarization_03=final_output[video_file.save_name]["audio_03"],
#         audio_diarization_06=final_output[video_file.save_name]["audio_06"],
#         audio_diarization_09=final_output[video_file.save_name]["audio_09"],
#         audio_diarization_12=final_output[video_file.save_name]["audio_12"],
#         video_diarization=final_output[video_file.save_name]["video_simple"],
#     )
#     final_output[video_file.save_name]["combined_simple"] = combined_output_simple.get_diarization_output(
#         videoDuration=video_durations[video_file.save_name],
#         video_save_name=video_file.save_name,
#         plot_name="combined_simple",
#     )

#     combined_output_enhanced = DiarizationOutput(
#         audio_diarization_03=final_output[video_file.save_name]["audio_03"],
#         audio_diarization_06=final_output[video_file.save_name]["audio_06"],
#         audio_diarization_09=final_output[video_file.save_name]["audio_09"],
#         audio_diarization_12=final_output[video_file.save_name]["audio_12"],
#         video_diarization=final_output[video_file.save_name]["video_enhanced"],
#     )
#     final_output[video_file.save_name]["combined_enhanced"] = combined_output_enhanced.get_diarization_output(
#         videoDuration=video_durations[video_file.save_name],
#         video_save_name=video_file.save_name,
#         plot_name="combined_enhanced",
#     )

# with open(
#     os.path.join(
#         params.RUN_OUTPUT_FOLDER,
#         f"final_diarization_output_w_combined_{run_timestamp}.pckl",
#     ),
#     "wb",
# ) as f:
#     pickle.dump(final_output, f)

# Evaluate results

final_evaluation_results_der = []
final_evaluation_results_wder = []
stats_df_list = []
assigned_speaker_df_list = []

final_output = pickle.load(
    open(
        os.path.join(
            params.RUN_OUTPUT_FOLDER,
            f"final_diarization_output_{run_timestamp}.pckl",
        ),
        "rb",
    )
)

for video_file in video_list.VIDEO_FILES:
    audioFilePath = os.path.join(
        params.VIDEO_OUTPUT_FOLDER,
        video_file.save_name,
        params.PYWAV_FOLDER_NAME,
        "audio.wav",
    )
    video_durations[video_file.save_name] = len(AudioSegment.from_file(audioFilePath)) / 1000
    if video_file.save_name not in final_output.keys() or len(final_output[video_file.save_name]) < 2:
        sys.stderr.write(f"Video {video_file.save_name} not found in final_output\n")
        continue
    evaluate_results = DiarizationEvaluateResults(
        video_file=video_file,
        video_diarization_output=final_output[video_file.save_name],
    )
    if video_file.ground_truth_type == "der":
        _result = evaluate_results.perform_evaluation(videoDuration=video_durations[video_file.save_name])
        if _result is not None:
            final_evaluation_results_der.append(_result)
    elif video_file.ground_truth_type == "wder":
        _result, _assigned_speakers = evaluate_results.perform_evaluation(
            videoDuration=video_durations[video_file.save_name]
        )
        if _result is not None:
            final_evaluation_results_wder.extend(_result)
            assigned_speaker_df_list.append(_assigned_speakers)

        # Assign speakers to transcriptions
        # assign_speakers = AssignSpeakers(
        #     video_file=video_file,
        #     final_video_output=final_output[video_file.save_name]["video_simple"],
        #     final_audio_output=final_output[video_file.save_name]["audio_12"],
        # )

        # final_df, stats_df = assign_speakers.perform_speaker_assignment()

        # stats_df_list.append(stats_df)

if len(stats_df_list) > 0:
    final_stats_df = pd.concat(stats_df_list)
    final_stats_df.to_excel(
        os.path.join(params.RUN_OUTPUT_FOLDER, f"temp_final_stats_{run_timestamp}.xlsx"),
        index=False,
    )
    # print(final_stats_df)

if len(assigned_speaker_df_list) > 0:
    assigned_speaker_df = pd.concat(assigned_speaker_df_list)
    assigned_speaker_df.to_excel(
        os.path.join(params.RUN_OUTPUT_FOLDER, f"temp_assigned_speakers_{run_timestamp}.xlsx"),
        index=False,
    )
    # print(assigned_speaker_df)


if len(final_evaluation_results_der) > 0:
    results_df = pd.DataFrame(final_evaluation_results_der)
    results_df.to_excel(
        os.path.join(
            params.RUN_OUTPUT_FOLDER,
            f"final_evaluation_results_der_{run_timestamp}.xlsx",
        ),
        index=False,
    )
    # print(results_df)

if len(final_evaluation_results_wder) > 0:
    results_df = pd.DataFrame(final_evaluation_results_wder)
    results_df.fillna(0, inplace=True)
    if "Correct" not in results_df.columns:
        results_df["Correct"] = 0
    if "Confusion" not in results_df.columns:
        results_df["Confusion"] = 0
    if "Missed" not in results_df.columns:
        results_df["Missed"] = 0
    results_df["Total"] = results_df["Correct"] + results_df["Confusion"] + results_df["Missed"]
    results_df["Accuracy"] = results_df["Correct"] / results_df["Total"]
    results_df["Error_Rate"] = (results_df["Confusion"] + results_df["Missed"]) / results_df["Total"]
    results_df.to_excel(
        os.path.join(
            params.RUN_OUTPUT_FOLDER,
            f"final_evaluation_results_wder_{run_timestamp}.xlsx",
        ),
        index=False,
    )
    # print(results_df)

# Create srt and js files
# create_subtitles = CreateSubtitles(
#     run_output_path=params.RUN_OUTPUT_FOLDER,
#     video_files=params.VIDEO_FILES,
#     video_folder=params.VIDEO_FOLDER,
# )
# create_subtitles.create_subtitles()
