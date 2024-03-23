import paho.mqtt.client as mqtt
import threading
import json
from image_detector import ImageDetector


class BrokerClient:
    def __init__(self, broker_address, topic):
        self.broker_address = broker_address
        self.broker_topic = topic

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        print("CLIENT - Connected with result code " + str(rc))
        client.subscribe(self.broker_topic)

    def on_message(self, client, userdata, msg):
        print("CLIENT - Received message on topic: " + msg.topic + " - Message " + str(msg.payload.decode()))
        try:
            msg_json = json.loads(msg.payload.decode())
        except Exception as e:
            print("Error decoding message " + str(e))
            return

        image_detector = ImageDetector()
        if "classes_to_save" in msg_json:
            if image_detector.classes_to_save != msg_json["classes_to_save"]:
                image_detector.update_classes_to_save(msg_json["classes_to_save"])

    def connect(self):
        self.client.connect(self.broker_address, 1883, 60)

    def start_in_thread(self):
        self.connect()

        thread = threading.Thread(target=self.start_listening)
        thread.start()

    def start_listening(self):
        self.client.loop_forever()

    def publish(self, message):
        """#TODO think if we want to send messages back?"""
        pass


# if __name__ == "__main__":
#     mqtt_client = BrokerClient("127.0.0.1", "test_topic")
#     mqtt_client.start_in_thread()
