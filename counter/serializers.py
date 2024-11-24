from rest_framework import serializers

class ObjectCountSerializer(serializers.Serializer):
    object_class = serializers.CharField()
    count = serializers.IntegerField()