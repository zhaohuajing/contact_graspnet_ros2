#!/usr/bin/env bash
set -e
echo "Running contact_graspnet docker container test script:"

# Automatically remove any old container with the same name
if [ "$(docker ps -aq -f name=contact_graspnet_container)" ]; then
  docker rm -f contact_graspnet_container
fi


docker run --gpus all -it --rm --shm-size=32g \
  --name contact_graspnet_container \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/graspnet_ws:/root/graspnet_ws \
  cuda118:contact_graspnet \
  bash -lc "\
    cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/ && \
    conda run -n contact-graspnet bash compile_pointnet_tfops.sh && \
    cd /root/graspnet_ws/ && \
    exec bash -l"