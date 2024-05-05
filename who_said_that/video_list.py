import os
from typing import Literal

# Input paramters
# VIDEO_FOLDER = "/vol3/sunil/AVA-AVD/dataset/videos"
VIDEO_FOLDER = "/vol3/sunil/FinalDataset"
MSD_VIDEO_FOLDER = "/vol3/sunil/MSDWild"
AVA_VIDEO_FOLDER = "/vol3/sunil/AVAAVDDataset"



class VideoFile:
    def __init__(
        self,
        name,
        video_folder,
        save_name,
        start=-1,
        end=-1,
        duration=-1,
        ground_truth_type="der",
        ground_truth_file=None,
        del_intermediate_files=True,
    ):
        self.name = name
        self.video_folder = video_folder
        self.save_name = save_name
        self.start = start
        self.end = end
        self.duration = duration
        self.ground_truth_type: Literal["der", "wder"] = ground_truth_type
        self.ground_truth_file = ground_truth_file
        self.del_intermediate_files = del_intermediate_files


video_names = [
    # "NDT_India_19_88",
    # "PiersMorgan_1_165_368",
    #    "PiersMorgan_1_0_165",
    #    "ESPN_1"
]

# VIDEO_FILES = [
#     VideoFile(
#         name=video_name,
#         video_folder=os.path.join(VIDEO_FOLDER, "videos"),
#         save_name=f"zzz_{video_name}",
#         ground_truth_type="wder",
#         ground_truth_file=os.path.join(VIDEO_FOLDER, "csv", f"{video_name}.csv"),
#         del_intermediate_files=False,
#     )
#     for video_name in video_names
# ]

VIDEO_FILES = [
    VideoFile(
        name="1j20qq1JyX4",
        save_name="zzz_1j20qq1JyX4",
        video_folder=os.path.join(AVA_VIDEO_FOLDER, "videos"),
        ground_truth_type="der",
        ground_truth_file=os.path.join(AVA_VIDEO_FOLDER, "rttms", "1j20qq1JyX4_c_01.rttm"),
        del_intermediate_files=False,
        start=899.993,
        duration=300
    )
]

# VIDEO_FILES = [
#     VideoFile(name=os.path.splitext(f)[0], save_name=os.path.splitext(f)[0])
#     for f in os.listdir(VIDEO_FOLDER)
#     if os.path.isfile(os.path.join(VIDEO_FOLDER, f)) and os.path.splitext(os.path.join(VIDEO_FOLDER, f))[1] in [".mp4", ".mkv"]
# ]

# VIDEO_FILES = [
#     VideoFile(
#         name=os.path.splitext(f)[0],
#         video_folder=os.path.join(MSD_VIDEO_FOLDER, "videos"),
#         save_name=os.path.splitext(f)[0],
#         ground_truth_type="der",
#         ground_truth_file=os.path.join(MSD_VIDEO_FOLDER, "rttms", f"{os.path.splitext(f)[0]}.rttm"),
#     )
#     for f in os.listdir(os.path.join(MSD_VIDEO_FOLDER, "videos"))
#     if os.path.isfile(os.path.join(MSD_VIDEO_FOLDER, "videos", f))
#     and os.path.splitext(f)[1] in [".mp4", ".mkv"]
# ]
