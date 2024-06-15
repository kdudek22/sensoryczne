import paho.mqtt.client as mqtt

# Define callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribe to the topic you want
    client.subscribe("topic/to/publish")

def on_message(client, userdata, msg):
    print("Received message on topic: "+msg.topic+" - Message: "+str(msg.payload.decode()))

# Create a client instance
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

# Assign callbacks to client
client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker
client.connect("127.0.0.1", 1883, 60)

# Start the loop to listen for messages
client.loop_forever()