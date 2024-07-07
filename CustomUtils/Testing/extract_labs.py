import pickle
import pandas as pd
temp = pickle.load(open("../WhoSaidThat/output/run_output/final_diarization_output_MY_DATASET.pckl", "rb"))
for video_name in temp.keys():
    lab_list = []
    for speaker, intervals in temp[video_name]["audio_03"].items():
        for interval in intervals:
            lab_list.append([interval[0], interval[1], "speaker"])
    df = pd.DataFrame(lab_list, columns=["start", "end", "label"])
    df.sort_values(by=["start"], inplace=True)
    df.to_csv("labs/" + video_name + ".lab", sep=" ", header=False, index=False)