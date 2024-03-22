from EventPooler import EventPoller
from ImageDetector import ImageDetector


if __name__ == '__main__':
    rtsp_url = "rtsp://192.168.0.213:8080/h264_opus.sdp"

    # poller = EventPoller("http://127.0.0.1:8000/api/events/")
    # poller.start_poller_thread()

    detector = ImageDetector("long_zoo.mp4", classes_to_save={"car"})
    detector.predict_on_video()



