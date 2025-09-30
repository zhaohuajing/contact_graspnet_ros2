#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from contact_graspnet_ros2.srv import GetGrasps
import numpy as np
import os, sys


class GraspClient(Node):

    def __init__(self):
        super().__init__('grasp_client')
        self.client = self.create_client(GetGrasps, 'get_grasps')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.base_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet"
        self.scene_id = int(sys.argv[1])

        npy_path = os.path.join(self.base_path, "test_data", f"{self.scene_id}.npy") 
        depth, K, seg = self.load_npy_file(npy_path)
        points = self.depth_to_point_cloud(depth, K)
        mask = self.flatten_segmentation(seg)

        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        mask = mask[valid]

        req = GetGrasps.Request()
        req.points = points.astype(np.float32).flatten().tolist()
        req.mask = mask.tolist()
        req.scene_id = self.scene_id

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info(f'Received {len(future.result().grasps.poses)} grasps')
        else:
            self.get_logger().error('Service call failed')

    def depth_to_point_cloud(self, depth, K):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = depth.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x3 = (x - cx) * z / fx
        y3 = (y - cy) * z / fy
        return np.stack((x3, y3, z), axis=-1).reshape(-1, 3)

    def flatten_segmentation(self, seg):
        return seg.flatten().astype(np.uint32)

    def load_npy_file(self, path):
        data = np.load(path, allow_pickle=True).item()
        return data['depth'], data['K'], data['seg']

def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Usage: python client_grasp_request.py <scene_id>")
        return
    GraspClient()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
