from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from datetime import datetime
from collections import deque


MAX_FRAME_COUNT_BUFFER_SIZE = 120
FRAME_COUNT_THRESHOLD_FOR_SAVING = 50


class CustomVideoSing(sv.VideoSink):
    def __init__(self, target_path, video_info, codec="mp4v"):
        super().__init__(target_path, video_info, codec)
        try:
            self.__fourcc = cv2.VideoWriter_fourcc(*codec)
        except TypeError as e:
            print(str(e) + ". Defaulting to mp4v...")
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


class ImageDetector:
    def __init__(self):
        self.video_path = "long_zoo.mp4"
        self.model = YOLO("yolov8x.pt")
        self.model.to('cuda')

        # self.interested_classes = {"car", "dog", "carrot", "bird", "sheep"}
        self.interested_classes = {"car", "bird"}
        self.id_to_name = self.model.model.names
        self.name_to_id = {self.model.model.names[i]: i for i in self.model.model.names}
        self.interested_classes_ids = [self.name_to_id[name] for name in self.interested_classes]

        self.video_info = sv.VideoInfo.from_video_path(self.video_path)

        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
        self.label_annotator = sv.LabelAnnotator()
        self.byte_track = sv.ByteTrack(frame_rate=self.video_info.fps, lost_track_buffer=100)

        self.frame_buffer = deque(maxlen=FRAME_COUNT_THRESHOLD_FOR_SAVING)

    def predict_on_video(self):
        cap = cv2.VideoCapture(self.video_path)


        curren_prediction_frame_count = 0
        is_saving_frames = False
        saved_buffer = False

        i=0
        while cap.isOpened():
            i+=1

            ret, frame = cap.read()
            if not ret:
                break

            frame = self.preprocess_frame(frame)
            results = self.model(frame, verbose=False)[0]

            detections = self.get_detections_from_results(results)
            simpler_detections = self.get_more_human_detections(detections)
            annotated_frame = self.add_annotation_to_frame(frame, detections)

            self.frame_buffer.append(annotated_frame)

            cv2.imshow("f", annotated_frame)

            if simpler_detections:
                curren_prediction_frame_count = min(curren_prediction_frame_count + 1, MAX_FRAME_COUNT_BUFFER_SIZE)
                if not is_saving_frames and curren_prediction_frame_count >= FRAME_COUNT_THRESHOLD_FOR_SAVING:
                    print("STARTED SAVING FRAMES")
                    is_saving_frames = True
                    saved_buffer = False
                    file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    wideo_sink = CustomVideoSing(target_path=f"{file_name}.mp4", video_info=self.video_info)
            else:
                curren_prediction_frame_count = max(curren_prediction_frame_count - 1, 0)
                if is_saving_frames and curren_prediction_frame_count == 0:
                    print("STOPPED SAVING FRAMES")
                    is_saving_frames = False
                    wideo_sink.release_video()

            if is_saving_frames:
                if not saved_buffer:
                    saved_buffer = True
                    for i in range(len(self.frame_buffer)):
                        wideo_sink.write_frame(self.frame_buffer[i])

                wideo_sink.write_frame(annotated_frame)

            if i%10 == 0:
                print(curren_prediction_frame_count)

            if cv2.waitKey(20) == ord('q'):
                break

        wideo_sink.release_video()
        cap.release()
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame):
        """#TODO think about preprocessing"""
        return sv.resize_image(frame, 1)

    def get_detections_from_results(self, results):
        detections = sv.Detections.from_ultralytics(results)
        detections = self.filter_detections(detections)
        detections = self.byte_track.update_with_detections(detections=detections)#

        return detections

    def filter_detections(self, detections):
        """#TODO implement: detections = detections[np.isin(detections.class_id, self.interested_classes_ids)]"""
        return detections[np.isin(detections.class_id, self.interested_classes_ids)]

    def get_more_human_detections(self, detections):
        res = []
        for i in range(len(detections)):
            res.append({"confidence": detections.confidence[i],
                        "class_id": detections.class_id[i],
                        "class_name": self.id_to_name[detections.class_id[i]],
                        "tracker_id": detections.tracker_id[i]})
        return res

    def add_annotation_to_frame(self, frame, detections):
        labels = self.create_labels_from_detections(detections)

        annotated_frame = frame.copy()

        annotated_frame = self.bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.putText(annotated_frame, f"Detected objects: {len(labels)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated_frame

    def create_labels_from_detections(self, detections):
        return [(f"#{detections.tracker_id[i]} {detections.class_id[i]} {self.id_to_name[detections.class_id[i]]} "
                 f"{'{:.2f}'.format(detections.confidence[i])}") for i in range(len(detections.tracker_id))]


if __name__ == "__main__":
    detector = ImageDetector()
    detector.predict_on_video()
