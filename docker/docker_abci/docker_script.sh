#!/bin/bash

# Variables (customize as needed)
CONTAINER_NAME=kazuya_scbulk
IMAGE_NAME=naivete5656/scbulk
USER="hoge"
PASSWORD="fuga"
DOCKER_DIR="./docker/docker_abci"
WORKDIR="/workdir"

# Ensure a subcommand is provided
if [ -z "$1" ]; then
  echo "Usage: $0 {build|run|exec|exec_root|stop}"
  exit 1
fi

# Subcommands
case "$1" in
  build)
    echo "Building Docker image..."
    docker build \
      --pull \
      --build-arg http_proxy=http://gw.ncc.go.jp:8080 \
      --build-arg https_proxy=http://gw.ncc.go.jp:8080 \
      --build-arg no_proxy=localhost,127.0.0.1 \
      --build-arg UID=$(id -u) \
      --build-arg GID=$(id -g) \
      --build-arg USER=$USER \
      --build-arg PASSWORD=$PASSWORD \
      -t $IMAGE_NAME $DOCKER_DIR
    ;;

  run)
    echo "Running Docker container..."
    docker run --name $CONTAINER_NAME \
      -d \
      --gpus='"device=0"' \
      --cpus="32" --memory="64g" --memory-swap="-1" --shm-size="2g" \
      --rm \
      -it \
      -v $(pwd):$WORKDIR \
      -w $WORKDIR $IMAGE_NAME \
    ;;

  exec)
    echo "Exec Docker container..."
    docker exec -it $CONTAINER_NAME /bin/zsh
    ;;
  
  exec_root)
    echo "Exec Docker container as root..."
    docker exec -u 0 -it $CONTAINER_NAME /bin/zsh
    ;;
  
  stop)
    docker stop $CONTAINER_NAME
    ;;

  *)
    echo "Invalid subcommand: $1"
    echo "Usage: $0 {build|run|exec|exec_root|stop}"
    exit 1
    ;;
esac
