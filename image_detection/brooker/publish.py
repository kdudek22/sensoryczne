import paho.mqtt.publish as publish

# Define the message payload and topic
message = '{"classes_to_save": ["bird", "dog"], "confidence_level": 0.85}'
topic = "test_topic"

# Publish the message
publish.single(topic, message, hostname="127.0.0.1")