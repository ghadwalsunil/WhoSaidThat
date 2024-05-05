from who_said_that.evaluation import utils
from who_said_that.utils import components
from who_said_that import params


class PyannoteAudioSegment:
    def __init__(
        self,
        segment_idx,
        audio_segment_start,
        audio_segment_end,
        ad_03_speaker,
        ad_06_speaker,
        ad_09_speaker,
        ad_12_speaker,
        has_overlap,
        vd_speaker=None,
    ):
        self.segment_idx = segment_idx
        self.audio_segment_start = audio_segment_start
        self.audio_segment_end = audio_segment_end
        self.vd_speaker = vd_speaker
        self.ad_03_speaker = ad_03_speaker
        self.ad_06_speaker = ad_06_speaker
        self.ad_09_speaker = ad_09_speaker
        self.ad_12_speaker = ad_12_speaker
        self.has_overlap = has_overlap

    def get_speaker(self, group_id):
        if group_id == "ad_03":
            return self.ad_03_speaker
        elif group_id == "ad_06":
            return self.ad_06_speaker
        elif group_id == "ad_09":
            return self.ad_09_speaker
        elif group_id == "ad_12":
            return self.ad_12_speaker
        elif group_id == "vd":
            return self.vd_speaker
        else:
            print(f"Invalid group id - {group_id}")
            return None

    def __lt__(self, other):
        return self.audio_segment_start < other.audio_segment_start

    def __gt__(self, other):
        return self.audio_segment_start > other.audio_segment_start

    def __eq__(self, other):
        return self.audio_segment_start == other.audio_segment_start

    def __le__(self, other):
        return self.audio_segment_start <= other.audio_segment_start

    def __ge__(self, other):
        return self.audio_segment_start >= other.audio_segment_start

    def __ne__(self, other):
        return self.audio_segment_start != other.audio_segment_start

    def __repr__(self) -> str:
        return f"PyannoteAudioSegment({self.audio_segment_start}, {self.audio_segment_end}, {self.ad_03_speaker}, {self.ad_06_speaker}, {self.ad_09_speaker}, {self.ad_12_speaker}, {self.vd_speaker})"


class MappingClass:
    def __init__(self):
        self.mapping_dict = {}

    def add_mapping(self, speaker_key_id, speaker_value_id, audio_segment_start, audio_segment_end):
        if speaker_value_id == "Unknown":
            return
        if speaker_key_id not in self.mapping_dict.keys():
            self.mapping_dict[speaker_key_id] = {speaker_value_id: audio_segment_end - audio_segment_start}
        else:
            if speaker_value_id not in self.mapping_dict[speaker_key_id].keys():
                self.mapping_dict[speaker_key_id][speaker_value_id] = audio_segment_end - audio_segment_start
            else:
                self.mapping_dict[speaker_key_id][speaker_value_id] += audio_segment_end - audio_segment_start

    def get_max_mapping(self, speaker_key_id):
        if speaker_key_id not in self.mapping_dict.keys():
            return None

        return max(self.mapping_dict[speaker_key_id], key=self.mapping_dict[speaker_key_id].get)


class DiarizationOutput:
    def __init__(
        self, audio_diarization_03, audio_diarization_06, audio_diarization_09, audio_diarization_12, video_diarization
    ):
        self.audio_diarization_03 = audio_diarization_03
        self.audio_diarization_06 = audio_diarization_06
        self.audio_diarization_09 = audio_diarization_09
        self.audio_diarization_12 = audio_diarization_12
        self.video_diarization = video_diarization
        self.audio_segments: list[PyannoteAudioSegment] = self.get_audio_segments()
        self.ad03_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_03")
        self.ad06_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_06")
        self.ad09_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_09")
        self.ad12_video_mapping: MappingClass = self.perform_audio_video_mapping("ad_12")
        self.predict_unknowns()

    def get_corresponding_speaker(self, present_group_id, target_group_id, speaker_id):

        if present_group_id == "ad_03":
            groups_not_allowed = ["ad_03"]
        elif present_group_id == "ad_06":
            groups_not_allowed = ["ad_03", "ad_06"]
        elif present_group_id == "ad_09":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09"]
        elif present_group_id == "ad_12":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09", "ad_12"]
        elif present_group_id == "vd":
            groups_not_allowed = ["vd"]
        else:
            print(f"Invalid present group id - {present_group_id}")
            return None

        if target_group_id in groups_not_allowed:
            print(f"For present_group {present_group_id}, target_group should not be among {groups_not_allowed}")
            return None

        present_group = self.get_group_by_id(present_group_id)
        if speaker_id not in present_group.keys():
            print(f"Speaker_id {speaker_id} not in present group {present_group_id} keys - {present_group.keys()}")
            return None

        speaker_interval = present_group[speaker_id]
        target_group = self.get_group_by_id(target_group_id)

        return self.get_mapping(speaker_interval, target_group)

    def get_all_child_speakers(self, parent_group_id, child_group_id, parent_speaker_id):

        if parent_group_id == "ad_12":
            groups_not_allowed = ["ad_12"]
        elif parent_group_id == "ad_09":
            groups_not_allowed = ["ad_09", "ad_12"]
        elif parent_group_id == "ad_06":
            groups_not_allowed = ["ad_06", "ad_09", "ad_12"]
        elif parent_group_id == "ad_03":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09", "ad_12"]
        elif parent_group_id == "vd":
            groups_not_allowed = ["vd"]
        else:
            print(f"Invalid parent group id - {parent_group_id}")
            return None

        if child_group_id in groups_not_allowed:
            print(f"For parent_group {child_group_id}, child_group should not be among {groups_not_allowed}")
            return None

        parent_group = self.get_group_by_id(parent_group_id)
        if parent_speaker_id not in parent_group.keys():
            print(
                f"parent_speaker_id {parent_speaker_id} not in parent_group {parent_group_id} keys - {parent_group.keys()}"
            )
            return None

        child_group = self.get_group_by_id(child_group_id)

        child_speakers = []

        for child_speaker_id in child_group.keys():
            _parent_speaker_id = self.get_corresponding_speaker(
                present_group_id=child_group_id, target_group_id=parent_group_id, speaker_id=child_speaker_id
            )

            if _parent_speaker_id == parent_speaker_id:
                child_speakers.append(child_speaker_id)

        return child_speakers

    def get_native_speakers_in_parent_group(self, child_group_id, parent_group_id, child_speaker_id):

        if child_group_id == "ad_03":
            groups_not_allowed = ["ad_03"]
        elif child_group_id == "ad_06":
            groups_not_allowed = ["ad_03", "ad_06"]
        elif child_group_id == "ad_09":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09"]
        elif child_group_id == "ad_12":
            groups_not_allowed = ["ad_03", "ad_06", "ad_09", "ad_12"]
        elif child_group_id == "vd":
            groups_not_allowed = ["vd"]
        else:
            print(f"Invalid child group id - {child_group_id}")
            return None

        if parent_group_id in groups_not_allowed:
            print(f"For child_group {child_group_id}, parent_group should not be among {groups_not_allowed}")
            return None

        child_group = self.get_group_by_id(child_group_id)
        if child_speaker_id not in child_group.keys():
            print(
                f"child_speaker_id {child_speaker_id} not in child_group {child_speaker_id} keys - {child_group.keys()}"
            )
            return None

        parent_group = self.get_group_by_id(parent_group_id)

        parent_speaker_id = self.get_corresponding_speaker(
            target_group_id=parent_group_id, present_group_id=child_group_id, speaker_id=child_speaker_id
        )

        return self.get_all_child_speakers(
            parent_group_id=parent_group_id, child_group_id=child_group_id, parent_speaker_id=parent_speaker_id
        )

    def get_audio_segments(self):
        audio_segment_list = []
        segment_idx = 0
        for speaker_id in self.audio_diarization_03.keys():
            for speech_segment_start, speech_segment_end in self.audio_diarization_03[speaker_id]:
                audio_segment = PyannoteAudioSegment(
                    segment_idx=segment_idx,
                    audio_segment_start=speech_segment_start,
                    audio_segment_end=speech_segment_end,
                    ad_03_speaker=speaker_id,
                    ad_06_speaker=self.get_corresponding_speaker(
                        present_group_id="ad_03", target_group_id="ad_06", speaker_id=speaker_id
                    ),
                    ad_09_speaker=self.get_corresponding_speaker(
                        present_group_id="ad_03", target_group_id="ad_09", speaker_id=speaker_id
                    ),
                    ad_12_speaker=self.get_corresponding_speaker(
                        present_group_id="ad_03", target_group_id="ad_12", speaker_id=speaker_id
                    ),
                    vd_speaker=self.get_mapping([(speech_segment_start, speech_segment_end)], self.video_diarization),
                    has_overlap=False,
                )
                audio_segment_list.append(audio_segment)

                segment_idx += 1

        audio_segment_list.sort()

        # Check whether each audio segment has overlap with other audio segments
        for i in range(len(audio_segment_list)):
            for j in range(i + 1, len(audio_segment_list)):
                if audio_segment_list[i].audio_segment_end > audio_segment_list[j].audio_segment_start:
                    audio_segment_list[i].has_overlap = True
                    audio_segment_list[j].has_overlap = True
                    audio_segment_list[i].vd_speaker = "Unknown"
                    audio_segment_list[j].vd_speaker = "Unknown"

        return audio_segment_list

    def perform_audio_video_mapping(self, audio_group_id):
        mapping_class = MappingClass()
        for audio_segment in self.audio_segments:
            if audio_group_id == "ad_03":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_03_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            elif audio_group_id == "ad_06":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_06_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            elif audio_group_id == "ad_09":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_09_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            elif audio_group_id == "ad_12":
                mapping_class.add_mapping(
                    speaker_key_id=audio_segment.ad_12_speaker,
                    speaker_value_id=audio_segment.vd_speaker,
                    audio_segment_start=audio_segment.audio_segment_start,
                    audio_segment_end=audio_segment.audio_segment_end,
                )
            else:
                print(f"Invalid audio_group_id - {audio_group_id}")
                return None

        return mapping_class

    def predict_unknowns(self):
        for audio_segment in self.audio_segments:
            if audio_segment.vd_speaker == "Unknown":
                vd_speaker = self.ad03_video_mapping.get_max_mapping(audio_segment.ad_03_speaker)
                if vd_speaker is not None:
                    audio_segment.vd_speaker = vd_speaker
                    self.ad03_video_mapping.add_mapping(
                        speaker_key_id=audio_segment.ad_03_speaker,
                        speaker_value_id=vd_speaker,
                        audio_segment_start=audio_segment.audio_segment_start,
                        audio_segment_end=audio_segment.audio_segment_end,
                    )
                else:
                    # audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_03_speaker}"
                    vd_speaker = self.ad06_video_mapping.get_max_mapping(audio_segment.ad_06_speaker)
                    if vd_speaker is not None:
                        audio_segment.vd_speaker = vd_speaker
                        self.ad06_video_mapping.add_mapping(
                            speaker_key_id=audio_segment.ad_06_speaker,
                            speaker_value_id=vd_speaker,
                            audio_segment_start=audio_segment.audio_segment_start,
                            audio_segment_end=audio_segment.audio_segment_end,
                        )
                    else:
                        # audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_06_speaker}"
                        vd_speaker = self.ad09_video_mapping.get_max_mapping(audio_segment.ad_09_speaker)
                        if vd_speaker is not None:
                            audio_segment.vd_speaker = vd_speaker
                            self.ad09_video_mapping.add_mapping(
                                speaker_key_id=audio_segment.ad_09_speaker,
                                speaker_value_id=vd_speaker,
                                audio_segment_start=audio_segment.audio_segment_start,
                                audio_segment_end=audio_segment.audio_segment_end,
                            )
                        else:
                            # audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_09_speaker}"
                            vd_speaker = self.ad12_video_mapping.get_max_mapping(audio_segment.ad_12_speaker)
                            if vd_speaker is not None:
                                audio_segment.vd_speaker = vd_speaker
                                self.ad12_video_mapping.add_mapping(
                                    speaker_key_id=audio_segment.ad_12_speaker,
                                    speaker_value_id=vd_speaker,
                                    audio_segment_start=audio_segment.audio_segment_start,
                                    audio_segment_end=audio_segment.audio_segment_end,
                                )
                            else:
                                audio_segment.vd_speaker = f"Unknown_{audio_segment.ad_12_speaker}"

    def get_diarization_output(self, videoDuration, video_save_name, plot_name):
        diarize_output = {}
        for audio_segment in self.audio_segments:
            if audio_segment.vd_speaker not in diarize_output.keys():
                diarize_output[audio_segment.vd_speaker] = []
            diarize_output[audio_segment.vd_speaker].append(
                (audio_segment.audio_segment_start, audio_segment.audio_segment_end)
            )

        components.create_annotation_plot(
            diarization_output=diarize_output,
            save_path=params.PLOT_OUTPUT_FOLDER,
            video_name=video_save_name,
            video_duration=videoDuration,
            plot_name=plot_name,
        )
        return diarize_output

    def get_group_by_id(self, group_id):
        if group_id == "ad_03":
            return self.audio_diarization_03
        if group_id == "ad_06":
            return self.audio_diarization_06
        if group_id == "ad_09":
            return self.audio_diarization_09
        if group_id == "ad_12":
            return self.audio_diarization_12
        if group_id == "vd":
            return self.video_diarization

    @staticmethod
    def get_mapping(speaker_interval, target_group):
        max_overlap = 0
        max_overlap_speaker = "Unknown"
        for speaker_id in target_group.keys():
            result_overlap, _ = utils.find_overlap(
                speaker_interval,
                target_group[speaker_id],
            )
            if result_overlap > max_overlap:
                max_overlap = result_overlap
                max_overlap_speaker = speaker_id

        return max_overlap_speaker
