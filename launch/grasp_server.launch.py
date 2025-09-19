from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='contact_graspnet_ros2',
            executable='grasp_server',
            name='grasp_server',
            output='screen'
        )
    ])
