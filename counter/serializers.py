from rest_framework import serializers

# class to create a serializer for different objects
class ObjectCountSerializer(serializers.Serializer):
    object_class = serializers.CharField()
    count = serializers.IntegerField()