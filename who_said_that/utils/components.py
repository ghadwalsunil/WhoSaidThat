import os
import cv2

import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment
from pyannote.core.notebook import Notebook
import pandas as pd
import numpy as np


def create_annotation_plot(
    diarization_output,
    save_path,
    video_name,
    video_duration,
    plot_name="diarization",
):
    custom_diarization = Annotation()

    for speaker_key in diarization_output.keys():
        for timeline in diarization_output[speaker_key]:
            custom_diarization[Segment(timeline[0], timeline[1])] = speaker_key

    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 2))

    # Plot the custom diarization result
    nb = Notebook()
    nb.plot_annotation(custom_diarization, ax, legend=True)

    # Customize the plot
    ax.set_xlabel("Time")
    ax.set_yticks([])  # To hide the y-axis
    ax.set_xlim(0, video_duration)
    ax.set_xticks(np.arange(0, int(video_duration), int(video_duration/20)))

    # Save the figure
    saveFileName = os.path.join(save_path, f"{video_name}_{plot_name}.png")
    fig.savefig(saveFileName, bbox_inches="tight")
    # Close the figure
    ax.clear()
    plt.close(fig)
