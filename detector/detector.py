# Load the pretrained model for object detection, also load the image along with its respective preprocessing 

import tensorflow as tf
import numpy as np
import cv2
from detector.config import COCO_CLASSES
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F

# certificate bypass
import urllib3
urllib3.disable_warnings()

import os
os.environ["CURL_CA_BUNDLE"] = ""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# -------------- Tensoirflow Framework --------------------
class ObjectDetectorTf:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be loaded. Check the file path.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def predict(self, image_path, threshold=0.5):
        image = self.load_image(image_path)
        input_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]
        detections = self.model(input_tensor)

        results = []
        for i in range(int(detections['num_detections'][0])):
            class_id = int(detections['detection_classes'][0][i])
            score = float(detections['detection_scores'][0][i])
            if score >= threshold:
                class_name = COCO_CLASSES.get(class_id, f"unknown_{class_id}")
                results.append({"class_id": class_id, "class_name": class_name, "score": score})
        print('-'*50)
        print(results)
        return results

# -------------- Pytorch Framework ---------------
class ObjectDetectorPyt:
    def __init__(self, ssd_weights_path, vgg_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SSD model without downloading weights
        self.model = ssd300_vgg16(weights=None)
        
        # Load the SSD weights from the local file
        ssd_weights = torch.load(ssd_weights_path, weights_only=True)
        self.model.load_state_dict(ssd_weights, strict=False)
        
        # Load the VGG backbone weights from the local file
        vgg_weights = torch.load(vgg_weights_path, weights_only=True)
        self.model.backbone.load_state_dict(vgg_weights, strict=False)

        self.model.eval()

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be loaded. Check the file path.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def preprocess_image(self, image):
        image_tensor = F.to_tensor(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def predict(self, image_path, threshold=0.5):
        image = self.load_image(image_path)
        input_tensor = self.preprocess_image(image)

        with torch.no_grad():
            predictions = self.model(input_tensor)[0]

        results = []
        for i in range(len(predictions["scores"])):
            score = float(predictions["scores"][i])
            if score >= threshold:
                class_id = int(predictions["labels"][i])
                class_name = COCO_CLASSES.get(class_id, f"unknown_{class_id}")
                bbox = predictions["boxes"][i].cpu().numpy().tolist()
                results.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "bbox": bbox,
                })

        print('-' * 50)
        print(results)
        return results