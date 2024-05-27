from rest_framework import serializers
from .models import VideoModel


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
        return f"http://127.0.0.1:8000/media/{obj.video}"
