from django.urls import path
from .views import VideoView

urlpatterns = [
    path("videos/", VideoView.as_view())
]
