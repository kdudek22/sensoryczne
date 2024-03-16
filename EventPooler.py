import requests
import time
from ImageDetector import ImageDetector


class EventPoller:
    def __init__(self, pool_wait_time=30):
        self.pool_wait_time = pool_wait_time
        self.pool = False
        self.classes_to_predict = ["car"]

    def start_pool(self):
        self.pool = True

        while self.pool:
            try:
                req = requests.get("http://127.0.0.1:8000/api/events/")
                res = req.json()
                api_response = res
                self.update_classes_to_predict(api_response)

            except Exception as e:
                print(f"ERROR IN REQUEST: {e}")

            time.sleep(self.pool_wait_time)

    def update_classes_to_predict(self, new_classes_to_predict):
        if self.classes_to_predict != new_classes_to_predict:
            self.classes_to_predict = new_classes_to_predict

            ImageDetector().update_classes_to_predict(self.classes_to_predict)

