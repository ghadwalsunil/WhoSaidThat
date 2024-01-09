from dotenv import load_dotenv
import os
from pyannote.core import Segment, Annotation
from pyannote.core import notebook
import matplotlib.pyplot as plt

load_dotenv()

from pyannote.audio import Pipeline


pretrained_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0", use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
)
audio_dir = "../../Dataset/Audios/"

import torch

pretrained_pipeline.to(torch.device("cuda"))


def create_annotation_plot(speaker_timelines, filename):
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
    fig.savefig(os.path.join(audio_dir, f"{filename}.png"), bbox_inches="tight")


f = "NDT_India_19_88.mp3"
diarization = pretrained_pipeline(os.path.join(audio_dir, f))
audio_output = {}
for duration, _, speaker_key in diarization.itertracks(yield_label=True):
    if speaker_key in audio_output.keys():
        audio_output[speaker_key].append((duration.start, duration.end))
    else:
        audio_output[speaker_key] = [(duration.start, duration.end)]
create_annotation_plot(audio_output, os.path.splitext(f)[0])
