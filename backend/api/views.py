from django.http import HttpResponse
from .serializers import VideoSerializer
from .models import VideoModel
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import ListCreateAPIView


class VideoView(ListCreateAPIView):
    queryset = VideoModel.objects.all()
    serializer_class = VideoSerializer
