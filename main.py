import time
from threading import Thread
import logging
from EventPooler import EventPoller
from ImageDetector import ImageDetector



if __name__ == '__main__':
    detector = ImageDetector("car.mp4")


    # detector2 = ImageDetector()
    # detector.predict_on_video()
    # x = EventPoller(5)
    # t1 = Thread(target=x.start_pool)
    # t1.start()

    # while True:
    #     logger.debug(api_response)
    #     time.sleep(10)

