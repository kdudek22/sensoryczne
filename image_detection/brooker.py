import paho.mqtt.client as mqtt
import threading
import json


class BrokerClient:
    def __init__(self, broker_address, topic, on_message):
        self.broker_address = broker_address
        self.broker_topic = topic

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.message_callback = on_message

    def on_connect(self, client, userdata, flags, rc):
        print("CLIENT - Connected with result code " + str(rc))
        client.subscribe(self.broker_topic)

    def on_message(self, client, userdata, msg):
        print("CLIENT - Received message on topic: " + msg.topic + " - Message " + str(msg.payload.decode()))

        msg_json = json.loads(msg.payload.decode())

        self.message_callback(msg_json)

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
