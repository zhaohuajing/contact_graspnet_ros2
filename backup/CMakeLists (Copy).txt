cmake_minimum_required(VERSION 3.5)
project(contact_graspnet_ros2)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Grasps.msg"
  "srv/GetGrasps.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install Python nodes as executables
# Install grasp_server as an executable
install(PROGRAMS
  contact_graspnet_ros2/grasp_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME grasp_server
)

# Install client_grasp_request as an executable
install(PROGRAMS
  contact_graspnet_ros2/client_grasp_request.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME client_grasp_request
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install resource index (needed for package discovery)
install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
