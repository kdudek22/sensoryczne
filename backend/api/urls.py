from django.urls import path
from .views import VideoView, ClassesToPredictListView, ClassesToPredictView

urlpatterns = [
    path("videos/", VideoView.as_view()),
    path("classes/", ClassesToPredictListView.as_view()),
    path("classes/<int:pk>", ClassesToPredictView.as_view()),
]
