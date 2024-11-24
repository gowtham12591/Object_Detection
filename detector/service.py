from counter.models import ObjectCount

# Service to handle database updates and retrieval
class ObjectCountService:
    def update_counts(self, predictions):
        """Update counts in the database."""
        counts = {}
        for prediction in predictions:
            class_name = prediction['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1

        for object_class, count in counts.items():
            obj, created = ObjectCount.objects.get_or_create(object_class=object_class)
            obj.count += count
            obj.save()

    def get_counts(self):
        """Retrieve counts from the database."""
        return ObjectCount.objects.all()