#!/usr/bin/env bash
set -e
echo "Running contact_graspnet docker container that contains both contact-graspnet and unseen_object_clustering"

# # Automatically remove any old container with the same name
# if [ "$(docker ps -aq -f name=contact_graspnet_container)" ]; then
#   docker rm -f contact_graspnet_container
# fi

# # Allow passing the environment name as the first argument, defaults to "contact-graspnet", or "./docker_run.sh unseen_obj" to runs with unseen_obj
# ENV_NAME=${1:-contact-graspnet}
# echo "Running docker container with env: $ENV_NAME"


docker run --gpus all -it --rm --shm-size=32g \
  --name cuda118 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/graspnet_ws:/root/graspnet_ws \
  cuda118:unseen_obj \
  bash -lc "cd /root/graspnet_ws/ && conda run -n unseen_obj && exec bash -l"


  # --name contact_graspnet_container \
  # contact_graspnet:cuda118 \
  # bash -lc "\
  #   cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/ && \
  #   conda run -n contact-graspnet bash compile_pointnet_tfops.sh && \
  #   cd /root/graspnet_ws/ && \
  #   exec bash -l" \