import multiprocessing
import requests
import time
from ImageDetector import ImageDetector
from threading import Thread


class EventPoller:
    def __init__(self, api_url, poll_wait_time=10):
        self.api_url = api_url
        self.pool_wait_time = poll_wait_time
        self.pool = False

    def start_poller_thread(self):
        print(f"Starting to pool the api @{self.api_url}")
        p = Thread(target=self.poll_api)
        p.start()

    def poll_api(self):
        self.pool = True
        time.sleep(10)

        while self.pool:
            try:
                req = requests.get(self.api_url)
                res = req.json()
                api_response = res
                self.update_classes_to_predict(api_response)

            except Exception as e:
                print(f"Error while polling: {e}")

            time.sleep(self.pool_wait_time)

    def update_classes_to_predict(self, new_classes_to_predict):
        detector = ImageDetector()

        if detector.classes_to_save != new_classes_to_predict:
            detector.update_classes_to_predict(new_classes_to_predict)
