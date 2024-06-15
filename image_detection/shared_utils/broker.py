import paho.mqtt.client as mqtt
import threading
import json
from .logging_config import logger


class BrokerClient:
    def __init__(self, broker_address, read_topic, publish_topic, message_received_callback=None):
        self.broker_address = broker_address
        self.broker_topic = read_topic
        self.broker_publish_topic = publish_topic

        self.client = mqtt.Client()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.message_received_callback = message_received_callback

    def on_connect(self, client, userdata, flags, rc):
        logger.info("Connected with result code " + str(rc))
        client.subscribe(self.broker_topic)

    def on_message(self, client, userdata, msg):
        logger.info("Received message on topic: " + msg.topic + " - Message " + str(msg.payload.decode()))
        try:
            json_message = json.loads(msg.payload.decode())
        except:
            logger.error("Could not decode broker message")
            return

        self.message_received_callback(json_message)

    def connect(self):
        self.client.connect(self.broker_address, 1883, 60)

    def start_in_thread(self):
        self.connect()

        thread = threading.Thread(target=self.start_listening)
        thread.start()

    def start_listening(self):
        self.client.loop_start()

    def publish(self, message):
        self.client.publish(self.broker_publish_topic, message)
