import os
from typing import Literal

# Input paramters
VIDEO_FOLDER = "/vol3/sunil/AVA-AVD/dataset/videos"


class VideoFile:
    def __init__(self, name, save_name, start=-1, end=-1, duration=-1, ground_truth_type="der", ground_truth_file=None):
        self.name = name
        self.save_name = save_name
        self.start = start
        self.end = end
        self.duration = duration
        self.ground_truth_type: Literal["der", "wder"] = ground_truth_type
        self.ground_truth_file = ground_truth_file


# VIDEO_FILES = [
#     VideoFile(
#         name=os.path.splitext(f)[0],
#         save_name=os.path.splitext(f)[0],
#         ground_truth_type="wder",
#         ground_truth_file=os.path.join("/vol3/sunil/AVA-AVD/dataset/csv", os.path.splitext(f)[0] + ".csv"),
#     )
#     for f in os.listdir(VIDEO_FOLDER)
#     if os.path.isfile(os.path.join(VIDEO_FOLDER, f)) and os.path.splitext(os.path.join(VIDEO_FOLDER, f))[1] in [".mp4"]
# ]

# VIDEO_FILES = [
#     VideoFile(
#         name="StarTalk_Mars_1680_1780",
#         save_name="zzz_StarTalk_Mars_1680_1780",
#         ground_truth_type="wder",
#         ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_1680_1780.csv",
#     ),
# ]


# VIDEO_FILES = [
#     VideoFile(name="00001", save_name="00001"),
#     VideoFile(name="00001", save_name="00002"),
# ]

VIDEO_FILES = [
    # VideoFile(
    #     name="2qQs3Y9OJX0",
    #     save_name="2qQs3Y9OJX0_c_01",
    #     start=900.488,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/2qQs3Y9OJX0_c_01.rttm",
    # ),
    # VideoFile(
    #     name="2qQs3Y9OJX0",
    #     save_name="2qQs3Y9OJX0_c_02",
    #     start=1200.115,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/2qQs3Y9OJX0_c_02.rttm",
    # ),
    VideoFile(
        name="2qQs3Y9OJX0",
        save_name="2qQs3Y9OJX0_c_03",
        start=1500.002,
        duration=300,
        ground_truth_type="der",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/2qQs3Y9OJX0_c_03.rttm",
    ),
    # VideoFile(
    #     name="1j20qq1JyX4",
    #     save_name="1j20qq1JyX4_c_01",
    #     start=899.993,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/1j20qq1JyX4_c_01.rttm",
    # ),
    # VideoFile(
    #     name="1j20qq1JyX4",
    #     save_name="1j20qq1JyX4_c_02",
    #     start=1200.193,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/1j20qq1JyX4_c_02.rttm",
    # ),
    # VideoFile(
    #     name="1j20qq1JyX4",
    #     save_name="1j20qq1JyX4_c_03",
    #     start=1500.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/1j20qq1JyX4_c_03.rttm",
    # ),
    # VideoFile(
    #     name="4ZpjKfu6Cl8",
    #     save_name="4ZpjKfu6Cl8_c_01",
    #     start=900.003,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/4ZpjKfu6Cl8_c_01.rttm",
    # ),
    # VideoFile(
    #     name="4ZpjKfu6Cl8",
    #     save_name="4ZpjKfu6Cl8_c_02",
    #     start=1200.003,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/4ZpjKfu6Cl8_c_02.rttm",
    # ),
    # VideoFile(
    #     name="4ZpjKfu6Cl8",
    #     save_name="4ZpjKfu6Cl8_c_03",
    #     start=1500.097,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/4ZpjKfu6Cl8_c_03.rttm",
    # ),
    # VideoFile(
    #     name="5milLu-6bWI",
    #     save_name="5milLu-6bWI_c_01",
    #     start=904.06,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/5milLu-6bWI_c_01.rttm",
    # ),
    # VideoFile(
    #     name="5milLu-6bWI",
    #     save_name="5milLu-6bWI_c_02",
    #     start=1200.003,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/5milLu-6bWI_c_02.rttm",
    # ),
    # VideoFile(
    #     name="5milLu-6bWI",
    #     save_name="5milLu-6bWI_c_03",
    #     start=1500.003,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/5milLu-6bWI_c_03.rttm",
    # ),
    # VideoFile(
    #     name="7YpF6DntOYw",
    #     save_name="7YpF6DntOYw_c_01",
    #     start=907.132,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/7YpF6DntOYw_c_01.rttm",
    # ),
    # VideoFile(
    #     name="7YpF6DntOYw",
    #     save_name="7YpF6DntOYw_c_02",
    #     start=1239.28,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/7YpF6DntOYw_c_02.rttm",
    # ),
    VideoFile(
        name="7YpF6DntOYw",
        save_name="7YpF6DntOYw_c_03",
        start=1505.43,
        duration=300,
        ground_truth_type="der",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/7YpF6DntOYw_c_03.rttm",
    ),
    # VideoFile(
    #     name="BCiuXAuCKAU",
    #     save_name="BCiuXAuCKAU_c_01",
    #     start=904.705,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/BCiuXAuCKAU_c_01.rttm",
    # ),
    # VideoFile(
    #     name="BCiuXAuCKAU",
    #     save_name="BCiuXAuCKAU_c_02",
    #     start=1202.653,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/BCiuXAuCKAU_c_02.rttm",
    # ),
    # VideoFile(
    #     name="BCiuXAuCKAU",
    #     save_name="BCiuXAuCKAU_c_03",
    #     start=1500.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/BCiuXAuCKAU_c_03.rttm",
    # ),
    # VideoFile(
    #     name="HKjR70GCRPE",
    #     save_name="HKjR70GCRPE_c_01",
    #     start=900.0,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/HKjR70GCRPE_c_01.rttm",
    # ),
    VideoFile(
        name="HKjR70GCRPE",
        save_name="HKjR70GCRPE_c_02",
        start=1200.001,
        duration=300,
        ground_truth_type="der",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/HKjR70GCRPE_c_02.rttm",
    ),
    # VideoFile(
    #     name="HKjR70GCRPE",
    #     save_name="HKjR70GCRPE_c_03",
    #     start=1500.002,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/HKjR70GCRPE_c_03.rttm",
    # ),
    # VideoFile(
    #     name="IKdBLciu_-A",
    #     save_name="IKdBLciu_-A_c_01",
    #     start=900.0,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/IKdBLciu_-A_c_01.rttm",
    # ),
    # VideoFile(
    #     name="IKdBLciu_-A",
    #     save_name="IKdBLciu_-A_c_02",
    #     start=1200.56,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/IKdBLciu_-A_c_02.rttm",
    # ),
    VideoFile(
        name="IKdBLciu_-A",
        save_name="IKdBLciu_-A_c_03",
        start=1500.138,
        duration=300,
        ground_truth_type="der",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/IKdBLciu_-A_c_03.rttm",
    ),
    # VideoFile(
    #     name="KHHgQ_Pe4cI",
    #     save_name="KHHgQ_Pe4cI_c_01",
    #     start=913.81,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/KHHgQ_Pe4cI_c_01.rttm",
    # ),
    # VideoFile(
    #     name="KHHgQ_Pe4cI",
    #     save_name="KHHgQ_Pe4cI_c_02",
    #     start=1223.14,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/KHHgQ_Pe4cI_c_02.rttm",
    # ),
    # VideoFile(
    #     name="KHHgQ_Pe4cI",
    #     save_name="KHHgQ_Pe4cI_c_03",
    #     start=1500.63,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/KHHgQ_Pe4cI_c_03.rttm",
    # ),
    # VideoFile(
    #     name="PmElx9ZVByw",
    #     save_name="PmElx9ZVByw_c_01",
    #     start=902.0,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/PmElx9ZVByw_c_01.rttm",
    # ),
    VideoFile(
        name="PmElx9ZVByw",
        save_name="PmElx9ZVByw_c_02",
        start=1200.467,
        duration=300,
        ground_truth_type="der",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/PmElx9ZVByw_c_02.rttm",
    ),
    # VideoFile(
    #     name="PmElx9ZVByw",
    #     save_name="PmElx9ZVByw_c_03",
    #     start=1500.412,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/PmElx9ZVByw_c_03.rttm",
    # ),
    # VideoFile(
    #     name="a5mEmM6w_ks",
    #     save_name="a5mEmM6w_ks_c_01",
    #     start=902.35,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/a5mEmM6w_ks_c_01.rttm",
    # ),
    # VideoFile(
    #     name="a5mEmM6w_ks",
    #     save_name="a5mEmM6w_ks_c_02",
    #     start=1200.023,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/a5mEmM6w_ks_c_02.rttm",
    # ),
    # VideoFile(
    #     name="a5mEmM6w_ks",
    #     save_name="a5mEmM6w_ks_c_03",
    #     start=1510.264,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/a5mEmM6w_ks_c_03.rttm",
    # ),
    # VideoFile(
    #     name="kMy-6RtoOVU",
    #     save_name="kMy-6RtoOVU_c_01",
    #     start=900.023,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/kMy-6RtoOVU_c_01.rttm",
    # ),
    # VideoFile(
    #     name="kMy-6RtoOVU",
    #     save_name="kMy-6RtoOVU_c_02",
    #     start=1200.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/kMy-6RtoOVU_c_02.rttm",
    # ),
    # VideoFile(
    #     name="kMy-6RtoOVU",
    #     save_name="kMy-6RtoOVU_c_03",
    #     start=1500.003,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/kMy-6RtoOVU_c_03.rttm",
    # ),
    # VideoFile(
    #     name="qrkff49p4E4",
    #     save_name="qrkff49p4E4_c_01",
    #     start=900.0,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/qrkff49p4E4_c_01.rttm",
    # ),
    # VideoFile(
    #     name="qrkff49p4E4",
    #     save_name="qrkff49p4E4_c_02",
    #     start=1200.07,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/qrkff49p4E4_c_02.rttm",
    # ),
    # VideoFile(
    #     name="qrkff49p4E4",
    #     save_name="qrkff49p4E4_c_03",
    #     start=1500.0,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/qrkff49p4E4_c_03.rttm",
    # ),
    # VideoFile(
    #     name="zC5Fh2tTS1U",
    #     save_name="zC5Fh2tTS1U_c_01",
    #     start=919.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/zC5Fh2tTS1U_c_01.rttm",
    # ),
    # VideoFile(
    #     name="zC5Fh2tTS1U",
    #     save_name="zC5Fh2tTS1U_c_02",
    #     start=1200.002,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/zC5Fh2tTS1U_c_02.rttm",
    # ),
    # VideoFile(
    #     name="zC5Fh2tTS1U",
    #     save_name="zC5Fh2tTS1U_c_03",
    #     start=1500.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/zC5Fh2tTS1U_c_03.rttm",
    # ),
    # VideoFile(
    #     name="zR725veL-DI",
    #     save_name="zR725veL-DI_c_01",
    #     start=900.412,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/zR725veL-DI_c_01.rttm",
    # ),
    # VideoFile(
    #     name="zR725veL-DI",
    #     save_name="zR725veL-DI_c_02",
    #     start=1200.0,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/zR725veL-DI_c_02.rttm",
    # ),
    # VideoFile(
    #     name="zR725veL-DI",
    #     save_name="zR725veL-DI_c_03",
    #     start=1500.002,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/zR725veL-DI_c_03.rttm",
    # ),
    # VideoFile(
    #     name="IzvOYVMltkI",
    #     save_name="IzvOYVMltkI_c_01",
    #     start=900.375,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/IzvOYVMltkI_c_01.rttm",
    # ),
    # VideoFile(
    #     name="IzvOYVMltkI",
    #     save_name="IzvOYVMltkI_c_02",
    #     start=1200.002,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/IzvOYVMltkI_c_02.rttm",
    # ),
    # VideoFile(
    #     name="IzvOYVMltkI",
    #     save_name="IzvOYVMltkI_c_03",
    #     start=1500.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/IzvOYVMltkI_c_03.rttm",
    # ),
    # VideoFile(
    #     name="UrsCy6qIGoo",
    #     save_name="UrsCy6qIGoo_c_01",
    #     start=900.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/UrsCy6qIGoo_c_01.rttm",
    # ),
    # VideoFile(
    #     name="UrsCy6qIGoo",
    #     save_name="UrsCy6qIGoo_c_02",
    #     start=1200.115,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/UrsCy6qIGoo_c_02.rttm",
    # ),
    # VideoFile(
    #     name="UrsCy6qIGoo",
    #     save_name="UrsCy6qIGoo_c_03",
    #     start=1500.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/UrsCy6qIGoo_c_03.rttm",
    # ),
    # VideoFile(
    #     name="yn9WN9lsHRE",
    #     save_name="yn9WN9lsHRE_c_01",
    #     start=900.003,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/yn9WN9lsHRE_c_01.rttm",
    # ),
    # VideoFile(
    #     name="yn9WN9lsHRE",
    #     save_name="yn9WN9lsHRE_c_02",
    #     start=1200.001,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/yn9WN9lsHRE_c_02.rttm",
    # ),
    # VideoFile(
    #     name="yn9WN9lsHRE",
    #     save_name="yn9WN9lsHRE_c_03",
    #     start=1500.004,
    #     duration=300,
    #     ground_truth_type="der",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/rttms/yn9WN9lsHRE_c_03.rttm",
    # ),
    # VideoFile(
    #     name="MagnusCarlson_542_599",
    #     save_name="MagnusCarlson_542_599",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/MagnusCarlson_542_599.csv",
    # ),
    # VideoFile(
    #     name="NDT_India_19_88",
    #     save_name="NDT_India_19_88",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/NDT_India_19_88.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_CMBR_190_225",
    #     save_name="StarTalk_CMBR_190_225",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_CMBR_190_225.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_CMBR_270_308",
    #     save_name="StarTalk_CMBR_270_308",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_CMBR_270_308.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_CMBR_319_356",
    #     save_name="StarTalk_CMBR_319_356",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_CMBR_319_356.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_CMBR_92_152",
    #     save_name="StarTalk_CMBR_92_152",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_CMBR_92_152.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Consciousness_1075_1175",
    #     save_name="StarTalk_Consciousness_1075_1175",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Consciousness_1075_1175.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Consciousness_1799_1887",
    #     save_name="StarTalk_Consciousness_1799_1887",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Consciousness_1799_1887.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Consciousness_2190_2254",
    #     save_name="StarTalk_Consciousness_2190_2254",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Consciousness_2190_2254.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Consciousness_2254_2314",
    #     save_name="StarTalk_Consciousness_2254_2314",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Consciousness_2254_2314.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Consciousness_2314_2387",
    #     save_name="StarTalk_Consciousness_2314_2387",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Consciousness_2314_2387.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Consciousness_56_180",
    #     save_name="StarTalk_Consciousness_56_180",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Consciousness_56_180.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Consciousness_683_784",
    #     save_name="StarTalk_Consciousness_683_784",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Consciousness_683_784.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_1050_1130",
    #     save_name="StarTalk_Cosmic_1050_1130",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_1050_1130.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_1135_1200",
    #     save_name="StarTalk_Cosmic_1135_1200",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_1135_1200.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_1350_1442",
    #     save_name="StarTalk_Cosmic_1350_1442",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_1350_1442.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_1550_1620",
    #     save_name="StarTalk_Cosmic_1550_1620",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_1550_1620.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_1820_1900",
    #     save_name="StarTalk_Cosmic_1820_1900",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_1820_1900.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_200_290",
    #     save_name="StarTalk_Cosmic_200_290",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_200_290.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_2225_2300",
    #     save_name="StarTalk_Cosmic_2225_2300",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_2225_2300.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_2600_2683",
    #     save_name="StarTalk_Cosmic_2600_2683",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_2600_2683.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_440_532",
    #     save_name="StarTalk_Cosmic_440_532",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_440_532.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_600_700",
    #     save_name="StarTalk_Cosmic_600_700",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_600_700.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Cosmic_780_850",
    #     save_name="StarTalk_Cosmic_780_850",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Cosmic_780_850.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Farming_0_98",
    #     save_name="StarTalk_Farming_0_98",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Farming_0_98.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Farming_1059_1180",
    #     save_name="StarTalk_Farming_1059_1180",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Farming_1059_1180.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Farming_1700_1800",
    #     save_name="StarTalk_Farming_1700_1800",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Farming_1700_1800.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Farming_2405_2500",
    #     save_name="StarTalk_Farming_2405_2500",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Farming_2405_2500.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Farming_2550_2645",
    #     save_name="StarTalk_Farming_2550_2645",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Farming_2550_2645.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Farming_307_387",
    #     save_name="StarTalk_Farming_307_387",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Farming_307_387.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_1001_1043",
    #     save_name="StarTalk_FlyingVehicles_1001_1043",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_1001_1043.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_1980_2040",
    #     save_name="StarTalk_FlyingVehicles_1980_2040",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_1980_2040.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_2446_2508",
    #     save_name="StarTalk_FlyingVehicles_2446_2508",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_2446_2508.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_2670_2710",
    #     save_name="StarTalk_FlyingVehicles_2670_2710",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_2670_2710.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_300_340",
    #     save_name="StarTalk_FlyingVehicles_300_340",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_300_340.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_674_719",
    #     save_name="StarTalk_FlyingVehicles_674_719",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_674_719.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_780_811",
    #     save_name="StarTalk_FlyingVehicles_780_811",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_780_811.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_FlyingVehicles_949_1000",
    #     save_name="StarTalk_FlyingVehicles_949_1000",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_FlyingVehicles_949_1000.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Mars_1026_1086",
    #     save_name="StarTalk_Mars_1026_1086",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_1026_1086.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Mars_1109_1175",
    #     save_name="StarTalk_Mars_1109_1175",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_1109_1175.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Mars_1345_1426",
    #     save_name="StarTalk_Mars_1345_1426",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_1345_1426.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Mars_1430_1500",
    #     save_name="StarTalk_Mars_1430_1500",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_1430_1500.csv",
    # ),
    VideoFile(
        name="StarTalk_Mars_1680_1780",
        save_name="StarTalk_Mars_1680_1780",
        ground_truth_type="wder",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_1680_1780.csv",
    ),
    # VideoFile(
    #     name="StarTalk_Mars_1810_1890",
    #     save_name="StarTalk_Mars_1810_1890",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_1810_1890.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Mars_201_5_302",
    #     save_name="StarTalk_Mars_201_5_302",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_201_5_302.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Mars_2020_2095",
    #     save_name="StarTalk_Mars_2020_2095",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_2020_2095.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Mars_2210_2270",
    #     save_name="StarTalk_Mars_2210_2270",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_2210_2270.csv",
    # ),
    VideoFile(
        name="StarTalk_Mars_2315_2376",
        save_name="StarTalk_Mars_2315_2376",
        ground_truth_type="wder",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_2315_2376.csv",
    ),
    # VideoFile(
    #     name="StarTalk_Mars_380_450",
    #     save_name="StarTalk_Mars_380_450",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_380_450.csv",
    # ),
    VideoFile(
        name="StarTalk_Mars_927_1025",
        save_name="StarTalk_Mars_927_1025",
        ground_truth_type="wder",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Mars_927_1025.csv",
    ),
    # VideoFile(
    #     name="StarTalk_Questions_1250_1350",
    #     save_name="StarTalk_Questions_1250_1350",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Questions_1250_1350.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Questions_1660_1710",
    #     save_name="StarTalk_Questions_1660_1710",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Questions_1660_1710.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Questions_690_750",
    #     save_name="StarTalk_Questions_690_750",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Questions_690_750.csv",
    # ),
    VideoFile(
        name="StarTalk_Questions_831_5_924",
        save_name="StarTalk_Questions_831_5_924",
        ground_truth_type="wder",
        ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Questions_831_5_924.csv",
    ),
    # VideoFile(
    #     name="StarTalk_Sleep_1152_1211",
    #     save_name="StarTalk_Sleep_1152_1211",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_1152_1211.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Sleep_1602_1639",
    #     save_name="StarTalk_Sleep_1602_1639",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_1602_1639.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Sleep_1980_2041",
    #     save_name="StarTalk_Sleep_1980_2041",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_1980_2041.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Sleep_2099_2160",
    #     save_name="StarTalk_Sleep_2099_2160",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_2099_2160.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Sleep_2379_2443",
    #     save_name="StarTalk_Sleep_2379_2443",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_2379_2443.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Sleep_2470_2551",
    #     save_name="StarTalk_Sleep_2470_2551",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_2470_2551.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Sleep_382_450",
    #     save_name="StarTalk_Sleep_382_450",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_382_450.csv",
    # ),
    # VideoFile(
    #     name="StarTalk_Sleep_748_796",
    #     save_name="StarTalk_Sleep_748_796",
    #     ground_truth_type="wder",
    #     ground_truth_file="/vol3/sunil/AVA-AVD/dataset/csv/StarTalk_Sleep_748_796.csv",
    # ),
]