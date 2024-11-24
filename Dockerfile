# Base image with specific Python version
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project
COPY . /app/

# Create staticfiles directory
RUN mkdir -p /app/staticfiles

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose the application port
EXPOSE 8000