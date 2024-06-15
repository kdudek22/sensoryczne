from .serializers import VideoSerializer, ClassToPredictSerializer
from .models import VideoModel, ClassToPredict
from rest_framework.generics import ListCreateAPIView, ListAPIView,RetrieveUpdateDestroyAPIView, GenericAPIView
from rest_framework.response import Response


class VideoView(ListCreateAPIView):
    queryset = VideoModel.objects.all().order_by("-date")
    serializer_class = VideoSerializer


class ClassesToPredictListView(ListAPIView, GenericAPIView):
    queryset = ClassToPredict.objects.all()
    serializer_class = ClassToPredictSerializer

    def post(self, request, *args, **kwargs):
        for entry in request.data:
            obj = ClassToPredict.objects.get(name=entry["name"])
            obj.is_active = entry["is_active"]
            obj.save()

        return Response(self.serializer_class(ClassToPredict.objects.all(), many=True).data)


class ClassesToPredictView(RetrieveUpdateDestroyAPIView):
    serializer_class = ClassToPredictSerializer
    queryset = ClassToPredict.objects.all()
