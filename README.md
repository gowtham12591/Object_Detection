


- create virtual environment and activate it
- create django project and application (object_detection, counter)



docker-compose build
docker-compose up -d

docker ps

docker-compose exec app python manage.py makemigrations
docker-compose exec app python manage.py migrate
docker-compose exec app python manage.py createsuperuser

docker-compose down

	•	Docker Compose ensures consistent, portable, and scalable environments for your entire application stack, making it easy to deploy and maintain in different environments.
	•	Nginx acts as a robust reverse proxy and load balancer, enhancing performance, security, and scalability for production deployments.
	•	Together, they enable a production-ready deployment setup that is far superior to local testing in terms of reliability, performance, and maintainability.