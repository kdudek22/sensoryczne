from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from datetime import datetime
from collections import deque
import multiprocessing
import shutil
import requests
import torch
from shared_utils.broker import BrokerClient
from shared_utils.logging_config import logger
import json
import os
import time
from shared_utils.VideoSink import CustomVideoSink

MAX_FRAME_COUNT_BUFFER_SIZE = 120
FRAME_COUNT_THRESHOLD_FOR_SAVING = 50


class ImageDetector:

    def __init__(self, model_path="best2.pt"):
        if not torch.cuda.is_available():
            raise RuntimeError("Cuda needs to be available")
        self.video_path = "input_videos/sarna.mp4"

        self.model_path: str = model_path
        self.model = self.initialize_model()

        self.class_name_to_id_map: dict[str, int] = {self.model.model.names[i]: i for i in self.model.model.names}
        self.id_to_class_name_map: dict[int, str] = self.model.model.names

        self.classes_to_detect: set[str] = self.request_for_classes_to_detect()
        self.classes_to_detect_ids: set[int] = set([self.class_name_to_id_map[name] for name in self.classes_to_detect])

        # supervision helpers for tracking and annotating frames
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
        self.label_annotator = sv.LabelAnnotator()
        self.byte_track = sv.ByteTrack(frame_rate=10, lost_track_buffer=100)

        # ques used to store images for detection, and frame rate evaluation
        self.frame_buffer: deque = deque(maxlen=FRAME_COUNT_THRESHOLD_FOR_SAVING)
        self.fps_buffer: deque = deque(maxlen=MAX_FRAME_COUNT_BUFFER_SIZE)

        self.last_time: float | None = None
        self.frame_prediction_and_confidence: list = []

        self.broker_send_message_callback = None

    def initialize_model(self):
        model = YOLO(self.model_path)
        model.to("cuda")
        return model

    def predict_on_video(self):
        cap = cv2.VideoCapture(self.video_path)

        consecutive_prediction_frame_count: int = 0
        is_currently_saving_frames: bool = False
        saved_buffer: bool = False
        wideo_sink = None
        file_name: str | None = None
        self.last_time = time.time()

        logger.info(f"Starting detection, classes_to_detect: {self.classes_to_detect}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.preprocess_frame(frame)
            results = self.model(frame, verbose=False)[0]

            detections = self.get_detections_from_results(results)
            formatted_detections = self.format_detections(detections)
            annotated_frame = self.add_annotation_to_frame(frame, detections)

            self.frame_buffer.append(annotated_frame)

            cv2.imshow("predictions", annotated_frame)

            if formatted_detections:
                consecutive_prediction_frame_count = min(consecutive_prediction_frame_count + 1, MAX_FRAME_COUNT_BUFFER_SIZE)
                if not is_currently_saving_frames and consecutive_prediction_frame_count >= FRAME_COUNT_THRESHOLD_FOR_SAVING:
                    """We are starting to save frames because we have been detecting objects for enough frames"""
                    logger.info("STARTED SAVING FRAMES")
                    self.send_started_detecting_message_to_broker()
                    is_currently_saving_frames = True
                    saved_buffer = False
                    file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
                    wideo_sink = CustomVideoSink(target_path=file_name, video_info=self.get_video_info(frame))
            else:
                consecutive_prediction_frame_count = max(consecutive_prediction_frame_count - 1, 0)
                if is_currently_saving_frames and consecutive_prediction_frame_count == 0:
                    """We are stopping saving frames because we have not been detecting objects for enough frames"""
                    logger.info("STOPPED SAVING FRAMES")
                    self.send_ended_detecting_message_to_broker()
                    is_currently_saving_frames = False
                    wideo_sink.release_video()
                    os.makedirs("results", exist_ok=True)
                    shutil.move(file_name, "results")
                    self.start_process_to_send_the_video(file_name)
                    self.frame_prediction_and_confidence = []

            if is_currently_saving_frames:
                if not saved_buffer:
                    """When we start saving for the first time, we store a buffer of frames that needs to be included"""
                    saved_buffer = True
                    for i in range(len(self.frame_buffer)):
                        wideo_sink.write_frame(self.frame_buffer[i])

                if formatted_detections:
                    self.frame_prediction_and_confidence.append(formatted_detections[0])
                wideo_sink.write_frame(annotated_frame)

            self.fps_buffer.append(self.calculate_fps())
            self.last_time = time.time()
            if cv2.waitKey(20) == ord('q'):
                break

        if wideo_sink:
            wideo_sink.release_video()

        cap.release()
        cv2.destroyAllWindows()

    def calculate_fps(self):
        """Based on the time it takes us to process consequent frames, we have a fps measure that is used for saving"""
        return 1/(time.time() - self.last_time)

    def preprocess_frame(self, frame):
        """#TODO think about preprocessing"""
        return frame

    def get_detections_from_results(self, results):
        """This is used to retrieve the detections from the model"""
        detections = sv.Detections.from_ultralytics(results)
        detections = self.filter_detections(detections)
        detections = self.byte_track.update_with_detections(detections=detections)
        return detections

    def filter_detections(self, detections):
        """Filters detections - only the ones present in the classes to detect are left"""
        return detections[np.isin(detections.class_id, [self.class_name_to_id_map[c] for c in self.classes_to_detect])]

    def format_detections(self, detections):
        """This pretty much just formats the detections to an easier to work with format"""
        res = []
        for i in range(len(detections)):
            res.append({"confidence": detections.confidence[i],
                        "class_id": detections.class_id[i],
                        "class_name": self.id_to_class_name_map[detections.class_id[i]],
                        "tracker_id": detections.tracker_id[i]})

        res.sort(key=lambda x: x["confidence"], reverse=True)
        return res

    def add_annotation_to_frame(self, frame, detections):
        """We add a detected objects, and  fps info to the frame"""
        labels = self.create_labels_from_detections(detections)
        annotated_frame = frame.copy()
        annotated_frame = self.bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        cv2.putText(annotated_frame, f"Detected objects: {len(labels)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"fps {self.calculate_fps()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated_frame

    def create_labels_from_detections(self, detections):
        """This outputs labels for objects, with the class name, confidence, and the tracker id"""
        return [(f"#{detections.tracker_id[i]} {detections.class_id[i]}"
                 f" {self.id_to_class_name_map[detections.class_id[i]]}"
                 f"{'{:.2f}'.format(detections.confidence[i])}") for i in range(len(detections.tracker_id))]

    def start_process_to_send_the_video(self, file_name):
        """This spawns a new process that will upload the recorded video to the api"""
        file_path = f"results/{file_name}"
        best_label = self.get_video_label_from_predictions()
        process = multiprocessing.Process(target=send_file_to_api, args=(file_path, best_label))
        process.start()

    def update_classes_to_detect(self, message: dict):
        """This is used by the update the classes by the broker"""
        if "classes_to_detect" not in message:
            logger.info("Message missing classes to detect")
            return

        old_classes = self.classes_to_detect
        self.classes_to_detect = set(message['classes_to_detect'])
        logger.info(f"Updated classes to detect: before {old_classes}, after: {self.classes_to_detect}")

    def send_started_detecting_message_to_broker(self):
        """When we start saving frames(detecting) we send a message to the broker indicating the fact"""
        logger.info("Sending detection message to broker")
        message = json.dumps({"detecting": True})
        self.broker_send_message_callback(message)

    def send_ended_detecting_message_to_broker(self):
        """When we stop detecting - we send a message to the broker"""
        logger.info("Sending ended detection message to broker")
        message = json.dumps({"detecting": False})
        self.broker_send_message_callback(message)

    def request_for_classes_to_detect(self):
        """On startup, we send a request to the api to get the configured classes to detect"""
        logger.info("Sending a request to the api for the classes to detect")
        response = requests.get(f"http://{server_address}:8000/api/classes")
        data: list[dict] = json.loads(response.text)
        res = []
        for entry in data:
            if entry["name"] in self.class_name_to_id_map:
                if entry["is_active"]:
                    res.append(entry["name"])
            else:
                logger.warn(f"Provided class does not exist in the model: {entry['name']}")
        if not res:
            logger.warn("There are no classes to detect")
        # return {"human", "dog", "horse", "hen", "deer"}
        return set(res)

    def get_video_info(self, frame):
        """This is used to get the video info when starting to save the frames"""
        width, height, fps = frame.shape[1], frame.shape[0], int(sum(self.fps_buffer)/len(self.fps_buffer))
        logger.info(f"Video info: h-{height}, w-{width}, fps-{fps} ")
        return sv.VideoInfo(width=width, height=height, fps=fps)

    def get_video_label_from_predictions(self):
        """Returns the best weighted average label from the prediction frames"""
        res = {}
        for prediction in self.frame_prediction_and_confidence:
            if prediction["class_name"] in res:
                res[prediction["class_name"]] += prediction["confidence"]
            else:
                res[prediction["class_name"]] = prediction["confidence"]

        return sorted(res.items(), key=lambda p: p[1], reverse=True)[0][0]


def send_file_to_api(file_path, label):
    logger.info(f"Sending video to api, label:  {label}")
    url = f"http://{os.environ.get('server_address')}:8000/api/videos/"

    body = {"detection": label}
    response = requests.post(url, data=body, files={"video": open(file_path, "rb")})

    logger.info("Video received by API, status code:" + str(response.status_code))

    logger.info(f"Removing file: {file_path}")
    os.remove(file_path)


if __name__ == "__main__":
    DEBUG_MODE = False

    detection_settings_topic = "test/detection_settings"
    detections_topic = "test/detections"
    server_address = "34.116.207.218"

    os.environ["server_address"] = server_address

    broker = BrokerClient(server_address, detection_settings_topic, detections_topic)

    detector = ImageDetector()
    broker.message_received_callback = detector.update_classes_to_detect
    detector.broker_send_message_callback = broker.publish

    broker.start_in_thread()
    detector.predict_on_video()

    broker.client.disconnect()
