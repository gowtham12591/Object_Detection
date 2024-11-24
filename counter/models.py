from django.db import models


class ObjectCount(models.Model):
    object_class = models.CharField(max_length=255, unique=True)
    count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.object_class}: {self.count}"
