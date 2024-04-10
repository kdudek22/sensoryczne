from django.http import HttpResponse
from .serializers import VideoSerializer
from .models import VideoModel
from rest_framework.response import Response
from rest_framework.views import APIView


class VideoView(APIView):
    def get(self, request):
        videos = VideoModel.objects.all()
        serialized_data = VideoSerializer(videos, many=True).data
        return Response(serialized_data)
