import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
print(torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    print("GPU is available.")
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU found. Using CPU.")