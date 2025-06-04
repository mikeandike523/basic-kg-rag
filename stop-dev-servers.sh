#!/bin/bash

cd mysql-dev-server

docker-compose down -v

cd ..

cd qdrant-dev-server

docker-compose down -v

cd ..

cd arangodb-dev-server

docker-compose down -v

cd ..