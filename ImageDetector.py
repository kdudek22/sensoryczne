import cv2
from ultralytics import YOLO
import os
from collections import deque
from datetime import datetime
import multiprocessing
from singleton_decorator import singleton


SAVING_FRAME_COUNT_THRESHOLD = 30
MAX_FRAME_BUFFER_COUNT = 60


@singleton
class ImageDetector:

    def __init__(self, video_source):
        print("CREATES SHIT")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport"
        self.model = YOLO("yolov8n.pt")
        self.model.to('cuda')
        self.video_source = video_source
        self.frame_buffer = deque(maxlen=MAX_FRAME_BUFFER_COUNT)
        self.saving_predictions = None
        self.current_folder_name = None
        self.classes_to_predict = ["car"]

    def predict_on_video(self):
        print(f"Starting prediction on video: {self.video_source}")
        print(f"looking for: {','.join(self.classes_to_predict)}")

        predicted_frame_count = 0

        cap = cv2.VideoCapture(self.video_source)
        i = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_buffer.append(frame)

            results = self.model.predict(source=frame, show=True, verbose=False)

            predicted_labels = []
            for res in results:
                for box in res.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf[0])
                    predicted_labels.append(self.model.names[class_id] + " " + str(confidence))

            if self.predictions_contain_interested_class(predicted_labels):
                predicted_frame_count = min(predicted_frame_count + 1, MAX_FRAME_BUFFER_COUNT)

                if self.should_start_saving_frames(predicted_frame_count):
                    self.create_folder_and_set_saving_status()
                    i = 0

            else:
                predicted_frame_count = max(predicted_frame_count - 1, 0)

                if self.should_create_video_from_images_in_folder(predicted_frame_count):
                    self.saving_predictions = False
                    p = multiprocessing.Process(target=create_video_from_images, args=(self.current_folder_name,))
                    p.start()
                    # TODO here we should send the video

            if self.saving_predictions:
                if i == 0:
                    self.save_images_from_buffer_to_folder()
                    i += len(self.frame_buffer)

                else:
                    cv2.imwrite(f"./{self.current_folder_name}/{i}.jpg", frame)
                    i += 1

            if cv2.waitKey(20) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def predictions_contain_interested_class(self, predicted_labels):
        return any(label.startswith(l_class) for label in predicted_labels for l_class in self.classes_to_predict)

    def should_start_saving_frames(self, predicted_frame_count):
        """When the frame count with predictions is above the given threshold, and we are not currently saving images"""
        return predicted_frame_count > SAVING_FRAME_COUNT_THRESHOLD and not self.saving_predictions

    def create_folder_and_set_saving_status(self):
        """Creates a folder named after the current datetime"""
        self.saving_predictions = True
        self.current_folder_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self.current_folder_name)

    def should_create_video_from_images_in_folder(self, predicted_frame_count):
        return predicted_frame_count == 0 and self.saving_predictions

    def save_images_from_buffer_to_folder(self):
        """We keep a buffer of given size of images, in case we decide to start capturing, we add the buffer first"""
        i = 0
        for old_frame in self.frame_buffer:
            cv2.imwrite(f"./{self.current_folder_name}/{i}.jpg", old_frame)
            i += 1

    def update_classes_to_predict(self, new_classes_to_predict):
        print("updating classes to predict")
        self.classes_to_predict = new_classes_to_predict


def create_video_from_images(video_folder_name, video_name=None):
    print("Creating the video...")
    if video_name is None:
        video_name = video_folder_name + ".avi"
    images = [img for img in os.listdir(video_folder_name) if img.endswith(".jpg")]

    images.sort(key=lambda name: int(name.split('.')[0]))

    frame = cv2.imread(os.path.join(video_folder_name, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 15, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(video_folder_name, image)))

    video.release()


