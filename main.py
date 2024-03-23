from broker_client import BrokerClient
from image_detector import ImageDetector


if __name__ == '__main__':
    # rtsp_url = "rtsp://192.168.0.213:8080/h264_opus.sdp"

    mqtt_client = BrokerClient("127.0.0.1", "test_topic")
    mqtt_client.start_in_thread()

    detector = ImageDetector("input_videos/long_zoo.mp4", classes_to_save={"car"})
    detector.predict_on_video()



