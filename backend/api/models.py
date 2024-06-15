from django.db import models


class VideoModel(models.Model):
    video = models.FileField(upload_to="videos")
    date = models.DateTimeField(auto_now_add=True)
    detection = models.CharField(max_length=255)


class ClassToPredict(models.Model):
    name = models.CharField(max_length=255)
    is_active = models.BooleanField(default=False)
