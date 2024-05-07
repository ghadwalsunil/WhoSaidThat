import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.core.notebook import Notebook


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
    ax.set_xticks(np.arange(0, int(video_duration), int(video_duration / 20)))

    # Save the figure
    saveFileName = os.path.join(save_path, f"{video_name}_{plot_name}.png")
    fig.savefig(saveFileName, bbox_inches="tight")
    # Close the figure
    ax.clear()
    plt.close(fig)


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
    if total_duration1 == 0:
        percentage_overlap1 = 0
    else:
        percentage_overlap1 = (overlap / total_duration1) * 100

    # Calculate the percentage of overlap with respect to intervals2
    if total_duration2 == 0:
        percentage_overlap2 = 0
    else:
        percentage_overlap2 = (overlap / total_duration2) * 100

    return percentage_overlap1, percentage_overlap2
