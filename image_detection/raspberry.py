import RPi.GPIO as GPIO
import time

# Use the BCM numbering
GPIO.setmode(GPIO.BCM)

# Set up GPIO 17 as an output
GPIO.setup(17, GPIO.OUT)

# Turn on GPIO 17
GPIO.output(17, GPIO.HIGH)

# Wait for a bit to see the result
time.sleep(5)

# Clean up GPIO settings
GPIO.cleanup()