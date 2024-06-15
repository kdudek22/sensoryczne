from shared_utils.broker import BrokerClient
from shared_utils.logging_config import logger

def on_message_received(message):
    print(message)


if __name__ == "__main__":
    broker = BrokerClient("localhost", "test/detections", None,
                          message_received_callback=on_message_received)
    broker.start_in_thread()
