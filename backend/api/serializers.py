from rest_framework import serializers
from .models import VideoModel, ClassToPredict
import os


class VideoSerializer(serializers.ModelSerializer):
    id = serializers.CharField(read_only=True)
    date = serializers.CharField(read_only=True)
    detection = serializers.CharField()
    video_url = serializers.SerializerMethodField(read_only=True)
    video = serializers.FileField(write_only=True)

    class Meta:
        model = VideoModel
        fields = ["id", "date", "detection", "video_url", "video"]

    def get_video_url(self, obj):
        return f"http://{os.environ.get('SERVER_ADDRESS', '127.0.0.1')}:8000/media/{obj.video}"


class ClassToPredictSerializer(serializers.ModelSerializer):
    id = serializers.CharField(read_only=True)
    name = serializers.CharField(read_only=True)
    is_active = serializers.BooleanField()

    class Meta:
        model = ClassToPredict
        fields = ["id", "name", "is_active"]
