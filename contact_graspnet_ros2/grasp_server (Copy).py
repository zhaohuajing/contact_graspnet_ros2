#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point
from contact_graspnet_ros2.srv import GetGrasps
from contact_graspnet_ros2.msg import Grasps

import numpy as np
# import transforms3d as tfs
import tf_transformations as tfs

# from contact_graspnet_ros2.pointnet2_grasp_direct import GraspEstimator, config_utils  # update path as needed
import os
import sys
sys.path.append(os.path.expanduser('~/graspnet_ws/src/contact_graspnet_ros2/checkpoints/scene_test_2048_bs3_hor_sigma_001'))
# sys.path.append('/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/checkpoints/scene_test_2048_bs3_hor_sigma_001')
sys.path.append(os.path.expanduser('~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/contact_graspnet'))

from pointnet2_grasp_direct import GraspEstimator

import config_utils

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class GraspPlanner(Node):

    def __init__(self):
        super().__init__('grasp_server')
        self.srv = self.create_service(GetGrasps, 'get_grasps', self.handle_grasp_request)

        # TensorFlow and GraspNet loading
        config_path = 'checkpoints/scene_test_2048_bs3_hor_sigma_001/'  # <-- Update this
        self.config = config_utils.load_config(config_path, batch_size=5)
        self.grasp_estimator = GraspEstimator(self.config)
        self.grasp_estimator.build_network()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        self.grasp_estimator.load_weights(self.sess, saver, config_path, mode='test')

        self.frame_rotate = np.array([
            [0, -1., 0., 0.],
            [0., 0, -1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])
        # if using tf_transformations as tfs
        self.rotate_z = np.round(tfs.rotation_matrix(np.pi, [0, 0, 1]))
        # if using transforms3d as tfs
        # rot_z = np.eye(4)
        # rot_z[:3, :3] = tfs.axangles.axangle2mat([0, 0, 1], np.pi)
        # self.rotate_z = np.round(rot_z)

        self.extra_rotations = [np.eye(4)]

    def handle_grasp_request(self, request, response):
        points = np.array(request.points, dtype=np.float32).reshape(-1, 3)
        mask = np.array(request.mask, dtype=np.uint32)

        points = (self.frame_rotate[:3, :3] @ points.T).T

        pose_list, score_list, sample_list, object_list = [], [], [], []
        for rot in self.extra_rotations:
            points_t = (rot[:3, :3] @ points.T).T

            pc_segments = {}
            for i in range(32):
                obj_mask = (mask & (1 << i)).astype(bool)
                if np.count_nonzero(obj_mask) > 0:
                    pc_segments[i + 1] = points_t[obj_mask]

            grasps, scores, samples, _ = self.grasp_estimator.predict_scene_grasps(
                self.sess, points_t, pc_segments=pc_segments,
                local_regions=True, filter_grasps=True, forward_passes=5)

            for i in grasps:
                for pose, score, sample in zip(grasps[i], scores[i], samples[i]):
                    pose = rot.T @ pose
                    pose = self.frame_rotate.T @ pose
                    pose = pose @ np.array([
                        [0., 1., 0., 0.],
                        [-1, 0., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                    ])
                    pose_rot = pose @ self.rotate_z
                    for pose_i in [pose, pose_rot]:
                        ros_pose = Pose()
                        ros_pose.position.x = pose_i[0, 3]
                        ros_pose.position.y = pose_i[1, 3]
                        ros_pose.position.z = pose_i[2, 3]
                        # if tf_transformations as tfs
                        quat = tfs.quaternion_from_matrix(pose_i)
                        # if using transforms3d as tfs
                        # quat = tfs.quaternions.mat2quat(pose_i[:3, :3])  # mat2quat expects a 3x3 rotation
                        ros_pose.orientation.x = quat[0]
                        ros_pose.orientation.y = quat[1]
                        ros_pose.orientation.z = quat[2]
                        ros_pose.orientation.w = quat[3]
                        pose_list.append(ros_pose)
                        score_list.append(score)
                        sample_list.append(Point(x=sample[0], y=sample[1], z=sample[2]))
                        object_list.append(i - 1)

        grasps_msg = Grasps()
        grasps_msg.poses = pose_list
        grasps_msg.scores = score_list
        grasps_msg.samples = sample_list
        grasps_msg.object_ids = object_list
        response.grasps = grasps_msg
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GraspPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
