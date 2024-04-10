from .settings import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'yolo_db',
        'USER': 'root',
        'PASSWORD': 'root',
        'HOST': 'database',
        'PORT': '3306',
    }
}
