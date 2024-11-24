Object Detection and Counter:

- The goal of the repo is to detect the object from the given image using pre-trained deep learning models and then share the count of objects in the image alogn with the respective class.

Instruction to configure the project
- create virtual environment and activate it
- Install the dependenciesfrom requirements file
- create django project and application (object_detection, counter)
- Now download the model weights for running the tensorflow & pytorch model from the below links
	http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
	https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth
	https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth
- Now create a models folder and then add the downloaded files inside the folder
- Now install postgresql in your system and configure it.
- To run this project locally just run the command 
	python manage.py runserver
- To use docker compose install docker, docker-compose
** Note: Before using docker-compose change the HOST in settings file to 'db' for local it can be 'localhost'
- Now run the below commands,
	docker-compose build
	docker-compose up -d
	docker-compose exec app python manage.py makemigrations
	docker-compose exec app python manage.py migrate
	docker-compose exec app python manage.py createsuperuser
	docker-compose down
- To test it either use curl command or install postmand and then pass your endpoint
	endpoint local : http://127.0.0.1:8000/api/detect
	endpoint docker_compose: http://127.0.0.1:8001/api/detect
	contract:
	{
    "image": "resources/images/cat.jpg",
    "threshold": 0.6,
    "framework": "tensorflow"    
}
- you can replace the framework with pytorch incase you want to use that framework


