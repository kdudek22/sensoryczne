import RPi.GPIO as GPIO
import time
from shared_utils.broker import BrokerClient
from shared_utils.logging_config import logger

# Use the BCM numbering
GPIO.setmode(GPIO.BCM)

# Set up GPIO 17 as an output
GPIO.setup(17, GPIO.OUT)

# # Turn on GPIO 17
# GPIO.output(17, GPIO.HIGH)
#
# # Wait for a bit to see the result
# time.sleep(5)
#
# # Clean up GPIO settings
# GPIO.cleanup()


def turn_on():
    logger.info("Turning on")
    GPIO.output(17, GPIO.HIGH)


def turn_off():
    logger.info("Turning off")
    GPIO.cleanup()


def on_message_received(message: dict):
    if "detecting" not in message:
        logger.error("Could not get parameter from message")
        return

    if message["detecting"]:
        turn_on()
    else:
        turn_off()


if __name__ == "__main__":
    detection_settings_topic = "test/detection_settings"
    detections_topic = "test/detections"

    server_address = "34.116.207.218"

    broker = BrokerClient(server_address, detections_topic, None, message_received_callback=on_message_received)
    broker.start_in_thread()
