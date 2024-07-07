import pickle
import pandas as pd
temp = pickle.load(open("../WhoSaidThat/output/run_output/final_diarization_output_MY_DATASET.pckl", "rb"))

for video_name in temp.keys():
    rttm_list = [["SPEAKER", video_name, 1, 0, 2, "<NA>", "<NA>", "spk02", "<NA>", "<NA>"]]

    df = pd.DataFrame(rttm_list, columns=["Type", "File", "Channel", "Start", "Duration", "NA_1", "NA_2", "Speaker", "NA_3", "NA_4"])
    df.to_csv("rttms/" + video_name + ".rttm", sep=" ", header=False, index=False)