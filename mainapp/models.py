from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class AudioFile(models.Model):
    audio = models.FileField(upload_to="uploads/audio/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.audio.name
