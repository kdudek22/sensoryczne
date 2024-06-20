from django.contrib import admin
from .models import VideoModel, ClassToPredict


@admin.register(VideoModel)
class VideoAdmin(admin.ModelAdmin):
    list_display = [field.name for field in VideoModel._meta.fields]


@admin.register(ClassToPredict)
class ClassToPredictAdmin(admin.ModelAdmin):
    list_display = [field.name for field in ClassToPredict._meta.fields]