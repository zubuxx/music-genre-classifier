#!/bin/bash

# Stop all containers running the 'genre-class-app' image
docker ps | grep 'genre-class-app' | awk '{print $1}' | xargs -r docker stop

# Remove all containers using the 'genre-class-app' image
docker ps -a | grep 'genre-class-app' | awk '{print $1}' | xargs -r docker rm

# Remove the 'genre-class-app' image
docker rmi genre-class-app

# Build an image from the Dockerfile
docker build -t genre-class-app .

# Run the application in a container on port 5000
docker run -d -p 5000:5000 genre-class-app
