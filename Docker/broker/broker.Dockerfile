# Use the official Eclipse Mosquitto image from Docker Hub
FROM eclipse-mosquitto:latest

# Copy the custom configuration file into the container
COPY ./mosquitto.conf /mosquitto/config/mosquitto.conf

# Expose the MQTT port (1883)
EXPOSE 1883

# Command to run the MQTT broker
CMD ["mosquitto", "-c", "/mosquitto/config/mosquitto.conf"]

#docker run -p 1883:1883  my-mosquitto