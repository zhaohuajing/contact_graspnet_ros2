#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
colcon build \
  --cmake-clean-cache \
  --cmake-args \
    -DPYTHON_EXECUTABLE=$HOME/.local/python3.10/bin/python3 \
    -DPYTHON_LIBRARY=$HOME/.local/python3.10/lib/libpython3.10.so \
    -DPYTHON_INCLUDE_DIR=$HOME/.local/python3.10/include/python3.10 \
  "$@"
