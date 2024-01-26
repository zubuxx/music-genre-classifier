#!/bin/bash

docker ps | grep 'genre-class-app' | awk '{print $1}' | xargs -r docker stop


# Usuń wszystkie kontenery korzystające z obrazu genre-class-app
docker ps -a | grep 'genre-class-app' | awk '{print $1}' | xargs -r docker rm

# Usuń obraz genre-class-app
docker rmi genre-class-app

# Zbuduj obraz z Dockerfile
docker build -t genre-class-app .

# Uruchom aplikację w kontenerze na porcie 5000
docker run  -p 5000:5000 genre-class-app
