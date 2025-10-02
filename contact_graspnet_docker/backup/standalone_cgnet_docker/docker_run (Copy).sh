#!/usr/bin/env bash
set -e
echo "Running contact_graspnet docker container test script:"

docker run --gpus all -it --rm --shm-size=32g \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/csrobot/contact_graspnet_docker/contact_graspnet/:/workspace/contact_graspnet \
  -v /home/csrobot/graspnet_ws/src/contact_graspnet_ros2/checkpoints/:/workspace/contact_graspnet/checkpoints \
  -v /home/csrobot/graspnet_ws/src/contact_graspnet_ros2/test_data/:/workspace/contact_graspnet/test_data \
  contact_graspnet:cuda118 \
  bash -lc "\
    conda run -n contact-graspnet bash compile_pointnet_tfops.sh && \
    exec bash -l"
