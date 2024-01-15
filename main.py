import os, subprocess, glob
from shutil import rmtree
import cv2
import demoTalkNetMod
import logging
import os
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import srt
import whisper

from stable_whisper import modify_model
from pyannote.core import Segment, Annotation
from pyannote.core import notebook
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

load_dotenv()
import torch
from pyannote.audio import Pipeline

logger = logging.getLogger("transcribe")

pretrained_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0", use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
)
pretrained_pipeline.to(torch.device("cuda"))
# sys.path.append(os.path.abspath("TalkNet-ASD/"))


def create_annotation_plot(speaker_timelines):
    custom_diarization = Annotation()

    for speaker_key in speaker_timelines.keys():
        for timeline in speaker_timelines[speaker_key]:
            custom_diarization[Segment(timeline[0], timeline[1])] = speaker_key

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plot the custom diarization result
    notebook.plot_annotation(custom_diarization, ax, legend=True)

    # Customize the plot (if needed)
    ax.set_xlabel("Time")
    ax.set_yticks([])  # To hide the y-axis

    # Save the figure
    #     fig.savefig('custom_diarization_plot.png', bbox_inches='tight')

    # Show the figure (optional)
    plt.show()


def get_whisper_model(model_name="base"):
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


def get_whisper_result(file_path, model):
    logging.info(f"Generating transcription for file - {file_path}")

    decode_options = dict(language="en")
    transcribe_options = dict(task="transcribe", **decode_options)
    output = model.transcribe(file_path, **transcribe_options)
    output = model.align(file_path, output, language="en")
    return output


def generate_whisper_transcription(file_name, file_path, output):
    logging.info(f"Organizing transcription for file - {file_path}")

    transcriptions = {}

    for num, s in enumerate(output.segments):
        transcriptions[num] = []
        for word in s.words:
            transcriptions[num].append(
                {
                    "text": s.text.strip(),
                    "segment_start": s.start,
                    "segment_end": s.end,
                    "word": word.word.strip(),
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


def find_overlap(intervals1, intervals2):
    overlap = 0
    total_duration1 = 0
    total_duration2 = 0

    # Calculate the total duration of intervals in intervals1
    for start, end in intervals1:
        total_duration1 += end - start

    # Calculate the total duration of intervals in intervals2 and find the overlap
    for start, end in intervals2:
        total_duration2 += end - start
        for s1, e1 in intervals1:
            common_start = max(s1, start)
            common_end = min(e1, end)
            if common_start < common_end:
                overlap += common_end - common_start

    # Calculate the percentage of overlap with respect to intervals1
    percentage_overlap1 = (overlap / total_duration1) * 100

    # Calculate the percentage of overlap with respect to intervals2
    percentage_overlap2 = (overlap / total_duration2) * 100

    return percentage_overlap1, percentage_overlap2


def get_video_to_audio_mapping(video_output, audio_output):
    video_audio_mapping = {}
    for audio_speaker in audio_output.keys():
        for video_speaker in video_output.keys():
            video_overlap, _ = find_overlap(video_output[video_speaker], audio_output[audio_speaker])
            if video_overlap > 80:
                video_audio_mapping[audio_speaker] = video_speaker
                break
    return video_audio_mapping


def get_segment_to_speaker_mapping(segment_start, segment_end, speaker_mapping, audio_output):
    final_speaker = None
    max_overlap = 0

    for audio_speaker in audio_output.keys():
        segment_overlap, _ = find_overlap([(segment_start, segment_end)], audio_output[audio_speaker])
        if segment_overlap > max_overlap:
            final_speaker = audio_speaker
            max_overlap = segment_overlap

    if final_speaker is None:
        return "Unknown", max_overlap
    else:
        if final_speaker in speaker_mapping.keys():
            return speaker_mapping[final_speaker], max_overlap
        else:
            return final_speaker, max_overlap


def get_word_to_speaker_mapping(word_start, word_end, diarization_output):
    for audio_speaker, speaker_timeline in diarization_output.items():
        segment_overlap, _ = find_overlap([(word_start, word_end)], speaker_timeline)
        if segment_overlap > 0.5:
            return audio_speaker

    return "Unknown"


def get_segments_and_speaker(df, diarization_output, video_name):
    temp_df = df[df["video_name"] == video_name].copy()
    temp_df["word_end"] = temp_df.apply(
        lambda row: (row["word_end"] + 0.1) if row["word_end"] == row["word_start"] else row["word_end"], axis=1
    )

    temp_df["assigned_speaker"] = temp_df.apply(
        lambda row: get_word_to_speaker_mapping(row["word_start"], row["word_end"], diarization_output), axis=1
    )

    # temp_df = (
    #     temp_df[["segment_id", "speaker", "word", "word_start", "word_end"]]
    #     .groupby(["segment_id", "speaker"], as_index=False)
    #     .agg({"word": " ".join, "word_start": min, "word_end": max})
    # )

    # temp_df["segment_id"] = temp_df.index + 1

    # temp_df["video_name"] = video_name

    return temp_df


class Args:
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)


args_dict = {
    "videoFolder": "Dataset/Temp",
    "outputFolder": "output",
    "pretrainModel": "pretrain_TalkSet.model",
    "nDataLoaderThread": 10,
    "facedetScale": 0.25,
    "minTrack": 10,
    "numFailedDet": 10,
    "minFaceSize": 1,
    "cropScale": 0.40,
    "start": 0,
    "duration": 0,
    "evalCol": False,
    "colSavePath": "/data08/col",
}

args = Args(args_dict)

if os.path.isfile(args.pretrainModel) == False:  # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s" % (Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)


model_name_openai = "medium.en"
model = get_whisper_model(model_name_openai)

args.videoFiles = [
    os.path.splitext(f)[0]
    for f in os.listdir(args.videoFolder)
    if os.path.isfile(os.path.join(args.videoFolder, f))
    and os.path.splitext(os.path.join(args.videoFolder, f))[1] in [".mp4"]
]

# final_video_output = {}
# for video_file in args.videoFiles:
#     print(video_file)
#     args.videoPath = glob.glob(os.path.join(args.videoFolder, video_file + ".*"))[0]
#     args.savePath = os.path.join(args.outputFolder, video_file)
#     args.audioPath = os.path.join(args.savePath, "pyavi", "audio.wav")
#     video = cv2.VideoCapture(args.videoPath)
#     args.frameRate = video.get(cv2.CAP_PROP_FPS)
#     video.release()
#     print(args.frameRate)
#     vidTracks, scores, args = demoTalkNetMod.preprocess(args)

#     df = demoTalkNetMod.get_track_face_encodings(vidTracks, scores, args)

#     # 001 - eps=0.4, min_samples = 4
#     # PL - eps=0.5, min_samples = 200
#     # Video3 - eps=0.5, min_samples = 100
#     # Video2 - eps=0.5, min_samples = 100
#     # Choose DBSCAN parameters
#     eps = 0.5
#     min_samples = 100

#     # Apply DBSCAN clustering
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     clusters = dbscan.fit_predict(df["Encoding"].to_list())
#     unique, counts = np.unique(clusters, return_counts=True)
#     print(np.asarray((unique, counts)).T)

#     df["Clusters"] = clusters

#     video_output = demoTalkNetMod.get_final_tracks(df, args)

#     final_video_output[video_file] = video_output


final_audio_output = {}
for video_file in args.videoFiles:
    print(video_file)
    args.videoPath = glob.glob(os.path.join(args.videoFolder, video_file + ".*"))[0]
    args.savePath = os.path.join(args.outputFolder, video_file)
    args.audioPath = os.path.join(args.savePath, "pyavi", "audio.wav")

    diarization = pretrained_pipeline(args.audioPath)

    audio_output = {}
    for duration, _, speaker_key in diarization.itertracks(yield_label=True):
        if speaker_key in audio_output.keys():
            audio_output[speaker_key].append((duration.start, duration.end))
        else:
            audio_output[speaker_key] = [(duration.start, duration.end)]

    final_audio_output[video_file] = audio_output

# Transcription
# Currently reading existing output

df = pd.read_excel("Dataset/Videos/Transcriptions_WithSpeakers.xlsx")
df["video_name"] = df["file_name"].apply(lambda x: os.path.splitext(x)[0])


final_df_list = []
for video_name, video_output in final_audio_output.items():
    final_df_list.append(get_segments_and_speaker(df, video_output, video_name))

final_df = pd.concat(final_df_list)

final_df.to_csv(
    os.path.join(args.outputFolder, "output" + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".csv"), index=False
)
