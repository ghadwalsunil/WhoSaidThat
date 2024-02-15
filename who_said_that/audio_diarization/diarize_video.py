import os
import pickle
import sys
import time

import params
from pydub import AudioSegment

from utils import components


class AudioDiarization:
    def __init__(
        self,
        video_files,
        video_folder,
        run_output_folder,
        video_output_folder,
        pretrained_pipeline,
    ):
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
        self.run_output_folder = run_output_folder
        self.video_output_folder = video_output_folder
        self.pretrained_pipeline = pretrained_pipeline

    def perform_audio_diarization(self):

        final_audio_output = {}
        for video_file in self.video_files:
            sys.stderr.write(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + " Processing video %s \r\n" % (video_file)
            )
            savePath = os.path.join(self.video_output_folder, video_file)
            pyaviPath = os.path.join(savePath, params.PYAVI_FOLDER_NAME)
            audioFilePath = os.path.join(pyaviPath, "audio.wav")
            videoDuration = len(AudioSegment.from_file(audioFilePath)) / 1000

            diarization = self.pretrained_pipeline(audioFilePath)

            audio_output = {}
            for duration, _, speaker_key in diarization.itertracks(yield_label=True):
                if speaker_key in audio_output.keys():
                    audio_output[speaker_key].append(
                        (round(duration.start, 2), round(duration.end, 2))
                    )
                else:
                    audio_output[speaker_key] = [
                        (round(duration.start, 2), round(duration.end, 2))
                    ]

            components.create_annotation_plot(
                diarization_output=audio_output,
                save_path=params.PLOT_OUTPUT_FOLDER,
                video_name=video_file,
                plot_name="audio",
                video_duration=videoDuration,
            )

            final_audio_output[video_file] = audio_output

        with open(
            os.path.join(self.run_output_folder, "audio_diarization_output.pckl"), "wb"
        ) as fil:
            pickle.dump(final_audio_output, fil)

        return final_audio_output
