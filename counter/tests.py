from django.test import TestCase
import pytest
from django.urls import reverse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@pytest.mark.django_db(databases=['default'])
def test_execute_objectdetection(client):
    image_path = os.path.join(BASE_DIR, 'tests', 'input', 'test.jpg')
    payload = {
    "image": image_path,
    "threshold": 0.6,
    "framework": "tensorflow"
    }

    url = reverse('object-detection')
    response = client.post(url, data=payload, content_type="application/json")
    print(response.json())
    assert response.status_code == 200