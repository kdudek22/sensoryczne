import cv2
from ultralytics import YOLO
import os
from collections import deque
from datetime import datetime
import multiprocessing
from singleton_decorator import singleton
import shutil
import math
import cvzone

SAVING_FRAME_COUNT_THRESHOLD = 30
MAX_FRAME_BUFFER_COUNT = 45


@singleton
class ImageDetector:

    def __init__(self, video_source="car.mp4", classes_to_save={"car",}):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport"
        self.model = YOLO("yolov8x.pt")
        self.model.to('cuda')
        self.video_source = video_source
        self.frame_buffer = deque(maxlen=MAX_FRAME_BUFFER_COUNT)
        self.saving_predictions = None
        self.current_folder_name = None
        self.classes_to_save = classes_to_save

    def predict_on_video(self):
        """This is the main loop. Does all the interesting things"""
        print(f"Starting prediction on video: {self.video_source}")
        print(f"looking for: {','.join(self.classes_to_save)}")

        cap = cv2.VideoCapture(self.video_source)

        predicted_frame_count = 0
        i = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.preprocess_frame(frame)
            self.frame_buffer.append(frame)

            results = self.model.predict(source=frame, show=True, verbose=False)
            predicted_labels = self.get_current_frame_predictions(results)

            if self.predictions_contain_class_to_save(predicted_labels):
                predicted_frame_count = min(predicted_frame_count + 1, MAX_FRAME_BUFFER_COUNT)
                if self.should_start_saving_frames(predicted_frame_count):
                    print("SAVING FRAMES")
                    self.create_folder_and_set_saving_status()
                    i = 0

            else:
                predicted_frame_count = max(predicted_frame_count - 1, 0)
                if self.should_create_video_from_images_in_folder(predicted_frame_count):
                    print("STOPPED SAVING FRAMES")
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

    def preprocess_frame(self, frame):
        """#TODO here we should think about the resolution and performance issues, and resize the image accordingly"""
        x, y, _ = frame.shape
        frame = cv2.resize(frame, (int(y / 2), int(x / 2)), interpolation=cv2.INTER_LINEAR)
        return frame

    def get_current_frame_predictions(self, results):
        labels = []
        for res in results:
            for box in res.boxes:
                labels.append(self.get_class_and_confidence_from_box(box))

        return labels

    def get_class_and_confidence_from_box(self, box):
        class_id = int(box.cls)
        confidence = float(box.conf[0])
        return self.model.names[class_id] + " " + str(confidence)

    def predictions_contain_class_to_save(self, predicted_labels):
        return any(label.startswith(l_class) for label in predicted_labels for l_class in self.classes_to_save)

    def should_start_saving_frames(self, predicted_frame_count):
        """When the frame count with predictions is above the given threshold, and we are not currently saving images"""
        return predicted_frame_count > SAVING_FRAME_COUNT_THRESHOLD and not self.saving_predictions

    def create_folder_and_set_saving_status(self):
        """Creates a folder named after the current datetime, and sets the saving_prediction status to True, indicating
        that we are capturing frames"""
        self.saving_predictions = True
        self.current_folder_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self.current_folder_name)

    def should_create_video_from_images_in_folder(self, predicted_frame_count):
        """If we stopped seeing classes we were looking for in the images, and we were saving frames. This indicates
        that we should take the created images, and make a video out of them"""
        return predicted_frame_count == 0 and self.saving_predictions

    def save_images_from_buffer_to_folder(self):
        """We keep a buffer of given size of images, in case we decide to start capturing, we add the buffer first
        This is done to make the recording more smooth, as an object needs to be registered for
        SAVING_FRAME_COUNT_THRESHOLD frames before we start saving frames. The buffer makes sure, the first frames
        we registered the object in, are also saved"""
        i = 0
        for old_frame in self.frame_buffer:
            cv2.imwrite(f"./{self.current_folder_name}/{i}.jpg", old_frame)
            i += 1

    def update_classes_to_predict(self, new_classes_to_predict):
        """#TODO this will be used to update the classes were looking for"""
        print(f"Updating classes, now looking for: {new_classes_to_predict}")
        self.classes_to_save = new_classes_to_predict


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

    shutil.rmtree(video_folder_name)


