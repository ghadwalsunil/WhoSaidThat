import os
import subprocess
import sys
from typing import Optional

import torch
import whisper
from pyannote.audio import Pipeline

from who_said_that import params
from who_said_that.models import utils
from who_said_that.models.talkNet import talkNet


class Models:
    def __init__(
        self,
        load_talknet_model: bool,
        load_pretrained_pipeline: bool,
        load_whisper_model: bool,
    ):
        self.load_talknet_model = load_talknet_model
        self.load_whisper_model = load_whisper_model
        self.load_pretrained_pipeline = load_pretrained_pipeline

    def get_talknet_model(self, pretrainedModelPath: str) -> Optional[talkNet]:
        if self.load_talknet_model:
            if os.path.isfile(pretrainedModelPath) == False:
                sys.stderr.write(
                    "Downloading the pretrained model %s \r\n" % pretrainedModelPath
                )
                Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
                cmd = "gdown --id %s -O %s" % (Link, pretrainedModelPath)
                subprocess.call(cmd, shell=True, stdout=None)

            s = talkNet()
            s.loadParameters(pretrainedModelPath)
            sys.stderr.write(
                "Model %s loaded from previous state! \r\n" % pretrainedModelPath
            )
            s.eval()

            return s
        else:
            return None

    def get_pretrained_pipeline(
        self, pretrained_pipeline_name: str = "pyannote/speaker-diarization-3.0"
    ) -> Optional[Pipeline]:
        if self.load_pretrained_pipeline:
            sys.stderr.write(
                "Downloading the pretrained model pyannote/speaker-diarization-3.0 \r\n"
            )
            pretrained_pipeline = Pipeline.from_pretrained(
                pretrained_pipeline_name, use_auth_token=os.getenv
            )
            pretrained_pipeline.to(torch.device("cuda"))
            if params.PRETRAINED_PIPELINE_PARAMS is not None:
                pretrained_pipeline.instantiate(params.PRETRAINED_PIPELINE_PARAMS)
                pretrained_pipeline.parameters(instantiated=True)
            sys.stderr.write(
                "Successfully loaded the pretrained model pyannote/speaker-diarization-3.0 \r\n"
            )
            return pretrained_pipeline
        else:
            return None

    def get_whisper_model(
        self, model_name: str = "medium.en"
    ) -> Optional[whisper.Whisper]:
        if self.load_whisper_model:
            return utils.get_whisper_model(model_name)
        else:
            return None
