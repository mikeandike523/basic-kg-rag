#!/bin/bash

cd mysql-dev-server

docker-compose down -v

docker-compose up -d

cd ..

cd qdrant-dev-server

docker-compose down -v

docker-compose up -d

cd ..

cd arangodb-dev-server

docker-compose down -v

docker-compose up -d

cd ..