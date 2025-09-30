"""script to perform grasping using a UR e-series robot and ZED camera.
Uses the dockerized contact-graspnet inference server to obtain grasp proposals.

Can be used with object of interest or region of interest masks.
"""
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

from airo_camera_toolkit.utils.annotation_tool import Annotation, get_manual_annotations
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
import requests

filepath = pathlib.Path(__file__)

# should correspond to the docker mount point
flask_dir = filepath.parent.parent / "flask_files"
# these names are hardcoded in the webserver
flask_input_file = flask_dir / "flask_input.npz"
flask_output_file = flask_dir / "flask_output.npz"


def get_best_prediction_from_output_file(output_path):
    # read the predictions for the 'scene.npz' file
    grasp_predictions_dict = np.load(output_path,allow_pickle=True)
    grasp_predictions_in_camera_frame = grasp_predictions_dict["pred_grasps_cam"]

    # dict as np array, need to get object first and then select dict key for segmask
    grasp_predictions_in_camera_frame = grasp_predictions_in_camera_frame.item()[1] 
    grasp_scores = grasp_predictions_dict["scores"]

    # dict as np array, need to get object first and then select dict key for segmask
    grasp_scores = grasp_scores.item()[1]

    # get the best grasp
    if len(grasp_scores) == 0:
        return None # no grasp found
    highest_idx = np.argmax(grasp_scores)
    best_grasp_in_camera_frame = grasp_predictions_in_camera_frame[highest_idx]

    # normalize orientation to make proper SE3 pose
    best_grasp_in_camera_frame[:3,:3] = normalize_so3_matrix(best_grasp_in_camera_frame[:3,:3])

    # important step, express in TCP of instead of (panda) gripper base
    # so that the poses can be used by other grippers (including our Robotiq gripper   )
    best_grasp_in_camera_frame[:3,3] += best_grasp_in_camera_frame[:3,2]*0.11 # express in TCP instead of (panda) gripper base
    return best_grasp_in_camera_frame


def visualize_grasp(depth_map, rgb,intrinsics,grasp_frame,pregrasp_frame):
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
    pregrasp_open3d.transform(pregrasp_frame)
    grasp_open3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    grasp_open3d.transform(grasp_frame)
    o3d.visualization.draw_geometries([pcd_legacy,pregrasp_open3d,grasp_open3d])

def manual_segmap(rgb):
    segmap = np.zeros_like(depth_map, dtype=np.uint8)

    annotation_spec = {
        "polygon": Annotation.Polygon,
    }

    annotations = get_manual_annotations(rgb, annotation_spec)
    polygon = annotations["polygon"]
    poly_list = []
    for point in polygon:
        poly_list.append(point[0])
        poly_list.append(point[1])
    poly_list = [poly_list]
    obj_map = BinarySegmentationMask.from_coco_segmentation_mask(poly_list,width=segmap.shape[1],height=segmap.shape[0]).bitmap
    segmap[obj_map == 1 ] = 1
    return segmap



# read camera extrinsics (obtained through hand-eye calibration)
camera_pose_file = filepath.parent / "camera_pose.json"
with open(camera_pose_file, "r") as f:
    camera_pose_in_robot_frame = Pose(**json.load(f)).as_homogeneous_matrix()
camera_pose_in_robot_frame[:3,:3] = normalize_so3_matrix(camera_pose_in_robot_frame[:3,:3])

# connect to robot and camera
gripper = Robotiq2F85("10.42.0.162")
robot = URrtde("10.42.0.162",manipulator_specs=URrtde.UR3E_CONFIG,gripper=gripper)

# collision free home position for my setup
joints_home_position = np.deg2rad(np.array([30,-80,40,-50,-90,130]))

camera = Zed2i(resolution=Zed2i.RESOLUTION_720,depth_mode=Zed2i.NEURAL_DEPTH_MODE)
camera.runtime_params.texture_confidence_threshold = 95
camera.runtime_params.confidence_threshold = 80

segmap = None
while True:
    # collect point cloud for visualization
    rgb = camera.get_rgb_image()
    depth_image = camera.get_depth_image()
    depth_map = camera.get_depth_map()

    intrinsics = camera.intrinsics_matrix()
    rgb = camera.get_rgb_image()
    rgb = ImageConverter.from_numpy_format(rgb).image_in_opencv_format

    if segmap is None:
        segmap = manual_segmap(rgb)
    else:
        c = input("press Enter to reuse the previous segmentation. press any other key to redo the segmentation")
        if not c == "":
            segmap = manual_segmap(rgb)

    np.savez(flask_input_file,rgb=rgb,depth=depth_map,K=intrinsics,segmap=segmap)
    
    # send request to flask server
    r = requests.get("http://localhost:5000/")
    if not r.status_code == 200:
        raise ValueError("grasp prediction failed. check container logs")
    
    # read the predictions for the 'scene.npz' file
    best_grasp_in_camera_frame = get_best_prediction_from_output_file(flask_output_file)
    if best_grasp_in_camera_frame is None:
        print("no grasp found, retrying (grasp proposals are stochastic)")
        continue

    # create pregrasp pose
    pre_grasp_in_camera_frame = best_grasp_in_camera_frame.copy()
    pre_grasp_in_camera_frame[:3,3] -= pre_grasp_in_camera_frame[:3,2]*0.1 # move 10cm in minus z direction


    visualize_grasp(depth_map, rgb,intrinsics,best_grasp_in_camera_frame,pre_grasp_in_camera_frame)

    c = input("press Enter to continue executing the grasp. press q to stop grasping. press any other key to retry grasping")
    if c == "q":
        break
    if not c == "":
        continue



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
    robot.move_to_joint_configuration(joints_home_position).wait()
    gripper.open().wait()

