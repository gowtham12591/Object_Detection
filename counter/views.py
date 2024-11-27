# Import required libraries
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from detector.service import ObjectCountService
from detector.detector import ObjectDetectorTf, ObjectDetectorPyt
from .serializers import ObjectCountSerializer
import json
import os

# Get the current working directory
current_dir = os.getcwd()
# Load the model path along with its weights
model_path = os.path.join(current_dir, 'models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
ssd_weights_path = os.path.join(current_dir, 'models/ssd_model_pyt/ssd300_vgg16_coco-b556d3b4.pth')
vgg_weights_path = os.path.join(current_dir, 'models/vgg_model_pyt/vgg16_features-amdegroot-88682ab5.pth')

# Create the API getting the object count for the given image
class ObjectDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    # Define the post function
    def post(self, request):
        # Load the data from the request
        data = json.loads(request.body)
        image_path = data.get("image")
        threshold = data.get("threshold")
        framework = str(data.get("framework")).lower()

        # Identify the framework and then load the model accordingly
        if framework == 'tensorflow':
            detector = ObjectDetectorTf(model_path=model_path)
        elif framework == 'pytorch':
            detector = ObjectDetectorPyt(ssd_weights_path, vgg_weights_path)
        else:
            return Response('Please share appropriate framework (tensorflow or pytorch)')
        
        # predict the objects with the loaded model and given threshold
        predictions = detector.predict(image_path, threshold=threshold)
        print('-'*50)
        print(predictions)

        # Update database with the count
        service = ObjectCountService()
        service.update_counts(predictions)

        # # Get counts of entire database entries
        # counts = service.get_counts()
        # serializer = ObjectCountSerializer(counts, many=True)

        # Prepare current prediction response with the current detection
        current_counts = []
        for prediction in predictions:
            class_name = prediction['class_name']
            count = prediction.get('count', 1)  # Assume count is 1 if not provided
            current_counts.append({'object_class': class_name, 'count': count})

        #Serialize and return the current counts
        serializer = ObjectCountSerializer(current_counts, many=True)

        return Response(serializer.data)