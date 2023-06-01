#!/bin/bash

HOME_PATH="$(dirname "$(realpath "$0")")"
cd $HOME_PATH


docker build -t mfv-alexnet-export_temp .
(docker run -it --name mfv-alexnet-export_temp mfv-alexnet-export_temp && docker cp mfv-alexnet-export_temp:/workspace/net.pt . ) || echo "Error"
docker rm mfv-alexnet-export_temp
