from .serializers import VideoSerializer, ClassToPredictSerializer
from .models import VideoModel, ClassToPredict
from rest_framework.generics import ListCreateAPIView, ListAPIView,RetrieveUpdateDestroyAPIView


class VideoView(ListCreateAPIView):
    queryset = VideoModel.objects.all()
    serializer_class = VideoSerializer


class ClassesToPredictListView(ListAPIView):
    queryset = ClassToPredict.objects.all()
    serializer_class = ClassToPredictSerializer


class ClassesToPredictView(RetrieveUpdateDestroyAPIView):
    serializer_class = ClassToPredictSerializer
    queryset = ClassToPredict.objects.all()
