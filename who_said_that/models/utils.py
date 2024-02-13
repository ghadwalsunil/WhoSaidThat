import sys
from typing import Optional

import whisper
from stable_whisper import modify_model

VALID_MODELS = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large",
]


def get_whisper_model(
    model_name: str = "medium.en",
) -> Optional[whisper.Whisper]:
    # initialize model
    sys.stderr.write("Initializing the Whisper model %s \r\n" % model_name)
    if model_name in VALID_MODELS:
        try:
            model = whisper.load_model(model_name)
            # Using the stable whisper to modifiy the model for better timestamps accuracy
            modify_model(model)
            sys.stderr.write(
                "Whisper model %s successfully initialized! \r\n" % model_name
            )
        except:
            sys.stderr.write("Unable to initialize Whisper model \r\n")
            return None
    else:
        sys.stderr.write(
            "Whisper Model  not found; available models = %s \r\n" % VALID_MODELS
        )
        return None

    return model
