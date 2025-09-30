import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose, Point
from contact_graspnet_ros2.srv import GetGrasps
from contact_graspnet_ros2.msg import Grasps
# TF rotations
import tf_transformations as tfs
import os
import subprocess

def run_inference_in_docker(scene_id):
        # container_name = "contact_graspnet_container"
        container_name = "magical_lovelace"
        np_path = f"test_data/{scene_id}.npy"

        cmd = [
            "docker", "exec", container_name,
            "bash", "-lc",
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet bash compile_pointnet_tfops.sh && "
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet python contact_graspnet/inference.py --np_path={np_path} --local_regions --filter_grasps"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Inference failed: {result.stderr}")
        return result.stdout


class GraspServer(Node):
    def __init__(self):
        super().__init__('grasp_server')
        self.srv = self.create_service(GetGrasps, 'get_grasps', self.handle_grasp_request)
        self.get_logger().info('Grasp server ready (executing inference inside a docker container).')
        self.base_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet"

    def handle_grasp_request(self, request, response):
        # For now we assume scene_id=0, later you can parse from request
        # self.scene_id = 0
        self.scene_id = request.scene_id
        # self.get_logger().info(f"Loading precomputed results for {result_path}")
        self.get_logger().info(f"Running inference in Docker for scene {self.scene_id}...")

        output = run_inference_in_docker(self.scene_id)
        self.get_logger().info(f"Inference finished. Output:\n{output}")

        result_path = os.path.join(self.base_path, "results", f"predictions_{self.scene_id}.npz")

        data = np.load(result_path, allow_pickle=True)
        pred_grasps_cam = data['pred_grasps_cam'].item()
        scores = data['scores'].item()
        contact_pts = data['contact_pts'].item()

        pose_list, score_list, sample_list, object_list = [], [], [], []

        self.get_logger().info(f"Loading grasp results for scene {self.scene_id}:")

        for obj_id in pred_grasps_cam.keys():
            for pose, score, sample in zip(pred_grasps_cam[obj_id], scores[obj_id], contact_pts[obj_id]):
                ros_pose = Pose()
                ros_pose.position.x = float(pose[0, 3])
                ros_pose.position.y = float(pose[1, 3])
                ros_pose.position.z = float(pose[2, 3])
                quat = tfs.quaternion_from_matrix(pose)
                ros_pose.orientation.x = float(quat[0])
                ros_pose.orientation.y = float(quat[1])
                ros_pose.orientation.z = float(quat[2])
                ros_pose.orientation.w = float(quat[3])

                pose_list.append(ros_pose)
                score_list.append(float(score))
                sample_list.append(Point(x=float(sample[0]), y=float(sample[1]), z=float(sample[2])))
                object_list.append(int(float(obj_id)))

            self.get_logger().info(f"Obtained {len(scores[obj_id])} grasps for object {int(obj_id)}")

        grasps_msg = Grasps()
        grasps_msg.poses = pose_list
        grasps_msg.scores = score_list
        grasps_msg.samples = sample_list
        grasps_msg.object_ids = object_list
        response.grasps = grasps_msg

        self.get_logger().info(f"Responded with {len(pose_list)} grasps from scene {self.scene_id}\n")
        return response


def main(args=None):
    rclpy.init(args=args)
    server = GraspServer()
    rclpy.spin(server)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
