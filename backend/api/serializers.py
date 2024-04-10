from rest_framework import serializers
from .models import VideoModel


class VideoSerializer(serializers.Serializer):
    id = serializers.CharField()
    date = serializers.CharField()
    detection = serializers.CharField()
    video_url = serializers.SerializerMethodField()

    class Meta:
        model = VideoModel
        fields = ["id", "date", "detection", "video_url"]

    def get_video_url(self, obj):
        return f"http://127.0.0.1:8000/media/{obj.video}"
