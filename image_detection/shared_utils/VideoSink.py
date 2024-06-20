import supervision as sv
import numpy as np
import cv2


class CustomVideoSink(sv.VideoSink):
    def __init__(self, target_path, video_info, codec="avc1"):
        super().__init__(target_path, video_info, codec)

        self.__fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )

    def write_frame(self, frame: np.ndarray):
        self.writer.write(frame)

    def release_video(self):
        self.writer.release()

