cmake_minimum_required(VERSION 3.5)
project(contact_graspnet_ros2)

# Core deps
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Grasps.msg"
  "srv/GetGrasps.srv"
  DEPENDENCIES std_msgs geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)

# --- Install Python backbone (embedded) ---

# 1) Install upstream Contact-GraspNet python package
#    The upstream module lives under: contact_graspnet/contact_graspnet
ament_python_install_package(contact_graspnet
  PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/contact_graspnet/contact_graspnet"
)

# 2) Install pointnet2 python namespace so "from pointnet2.utils import tf_util" works
#    We install the whole pointnet2 tree (py files only) as a package
install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/contact_graspnet/pointnet2/
  DESTINATION lib/${PROJECT_NAME}/python_ext/site-packages/pointnet2
  FILES_MATCHING PATTERN "*.py"
)

# Make sure Python sees that extra site-packages path at runtime:
# We'll export it with an environment hook.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/resource/pythonpath.dsv.in
  ${CMAKE_CURRENT_BINARY_DIR}/pythonpath.dsv
  @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pythonpath.dsv
  DESTINATION share/${PROJECT_NAME}/hook
)

# 3) (Optional) Install prebuilt TF custom ops (.so) if you have them
#    Comment this in if tf_ops have already been compiled for your TF/CUDA.
# install(DIRECTORY
#   ${CMAKE_CURRENT_SOURCE_DIR}/contact_graspnet/pointnet2/tf_ops/
#   DESTINATION lib/${PROJECT_NAME}/python_ext/site-packages/pointnet2/tf_ops
#   FILES_MATCHING PATTERN "*.so"
# )

# --- Install your ROS 2 python nodes as executables ---
install(PROGRAMS
  contact_graspnet_ros2/grasp_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME grasp_server
)
install(PROGRAMS
  contact_graspnet_ros2/client_grasp_request.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME client_grasp_request
)

# Launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Resource index (package discovery)
install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/
)

# Checkpoints: install to share so code can find them via get_package_share_directory
install(DIRECTORY checkpoints/
  DESTINATION share/${PROJECT_NAME}/checkpoints
)

ament_package()
