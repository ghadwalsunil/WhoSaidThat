import os
import pickle

import pandas as pd
import params
from create_subtitles import utils


class CreateSubtitles:
    def __init__(self, run_output_path: str, video_files: list, video_folder: str):
        if not video_files:
            self.video_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(video_folder)
                if os.path.isfile(os.path.join(video_folder, f))
                and os.path.splitext(os.path.join(video_folder, f))[1] in [".mp4"]
            ]
        else:
            self.video_files = video_files
        self.video_folder = video_folder
        self.run_output_path = run_output_path
        self.final_output_df = pd.read_excel(
            os.path.join(self.run_output_path, "result_audio_video.xlsx")
        )

    def create_subtitles(self):

        utils.generate_subtitles(
            self.final_output_df,
            output_type=["srt", "js"],
            srt_output_path=params.SRT_OUTPUT_FOLDER,
            js_output_path=params.JS_OUTPUT_FOLDER,
        )
