"""script to execute a grasp using a UR e-series robot and ZED camera. Requires manually running the inference code."""
import numpy as np
import pathlib 
import json

from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_dataset_tools.data_parsers.pose import Pose
from airo_camera_toolkit.utils.image_converter import ImageConverter
import open3d as o3d
from airo_spatial_algebra.se3 import normalize_so3_matrix


filepath = pathlib.Path(__file__)

# read the predictions for the 'scene.npz' file
grasp_predictions_file = filepath.parents[1] / "results" / "predictions_scene.npz"
grasp_predictions_dict = np.load(grasp_predictions_file,allow_pickle=True)
grasp_predictions_in_camera_frame = grasp_predictions_dict["pred_grasps_cam"]

# dict as np array, need to get object first and then select dict key for segmask
grasp_predictions_in_camera_frame = grasp_predictions_in_camera_frame.item()[1] 

grasp_scores = grasp_predictions_dict["scores"]

# dict as np array, need to get object first and then select dict key
grasp_scores = grasp_scores.item()[1]

highest_idx = np.argmax(grasp_scores)
best_grasp_in_camera_frame = grasp_predictions_in_camera_frame[highest_idx]

print(best_grasp_in_camera_frame)

# normalize orientation 
best_grasp_in_camera_frame[:3,:3] = normalize_so3_matrix(best_grasp_in_camera_frame[:3,:3])

# important step, express in TCP of instead of (panda) gripper base
# so that the poses can be used by other grippers (including our Robotiq gripper   )
best_grasp_in_camera_frame[:3,3] += best_grasp_in_camera_frame[:3,2]*0.11 # express in TCP instead of (panda) gripper base

# create pregrasp pose
pre_grasp_in_camera_frame = best_grasp_in_camera_frame.copy()
pre_grasp_in_camera_frame[:3,3] -= pre_grasp_in_camera_frame[:3,2]*0.1 # move 10cm in minus z direction

# read camera extrinsics (obtained through hand-eye calibration)
camera_pose_file = filepath.parent / "camera_pose.json"
with open(camera_pose_file, "r") as f:
    camera_pose_in_robot_frame = Pose(**json.load(f)).as_homogeneous_matrix()
camera_pose_in_robot_frame[:3,:3] = normalize_so3_matrix(camera_pose_in_robot_frame[:3,:3])

# connect to robot and camera
gripper = Robotiq2F85("10.42.0.162")
robot = URrtde("10.42.0.162",manipulator_specs=URrtde.UR3E_CONFIG,gripper=gripper)
camera = Zed2i(resolution=Zed2i.RESOLUTION_720,depth_mode=Zed2i.NEURAL_DEPTH_MODE)
camera.runtime_params.texture_confidence_threshold = 95
camera.runtime_params.confidence_threshold = 80


# collect point cloud for visualization
rgb = camera.get_rgb_image()
depth_image = camera.get_depth_image()
depth_map = camera.get_depth_map()

intrinsics = camera.intrinsics_matrix()
rgb = camera.get_rgb_image()
rgb = ImageConverter.from_numpy_format(rgb).image_in_opencv_format



rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(rgb),
    o3d.geometry.Image(depth_map),
    depth_scale=1.0,
    depth_trunc=1.0,
    convert_rgb_to_intensity=False,
)

o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    *camera.resolution, intrinsics
)

pcd_legacy = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, o3d_intrinsics
)
pregrasp_open3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
pregrasp_open3d.transform(pre_grasp_in_camera_frame)
grasp_open3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
grasp_open3d.transform(best_grasp_in_camera_frame)
o3d.visualization.draw_geometries([pcd_legacy,pregrasp_open3d,grasp_open3d])


c = input("press C to continue executing the grasp. press any key to quit")
if not c == "c":
    raise ValueError("invalid user input")

# collision free home position for my setup
joints_home_position = np.deg2rad(np.array([60,-80,40,-50,-90,130]))
robot.move_to_joint_configuration(joints_home_position).wait()
gripper.open().wait()


pregrasp_in_robot_frame = camera_pose_in_robot_frame @ pre_grasp_in_camera_frame
pregrasp_in_robot_frame[:3,:3] = normalize_so3_matrix(pregrasp_in_robot_frame[:3,:3])

grasp_in_robot_frame = camera_pose_in_robot_frame @ best_grasp_in_camera_frame
grasp_in_robot_frame[:3,:3] = normalize_so3_matrix(grasp_in_robot_frame[:3,:3])

robot.move_linear_to_tcp_pose(pregrasp_in_robot_frame).wait()
robot.move_linear_to_tcp_pose(grasp_in_robot_frame).wait()
robot.gripper.close().wait()
robot.move_linear_to_tcp_pose(pregrasp_in_robot_frame).wait()
robot.move(joints_home_position).wait()

