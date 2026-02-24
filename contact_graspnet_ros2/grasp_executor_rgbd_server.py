#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose, Point, PoseArray
from contact_graspnet_ros2.srv import GetGrasps
from contact_graspnet_ros2.msg import Grasps

import tf_transformations as tfs
import os
import subprocess
import json

import tf2_ros
import tf_transformations as tft
import rclpy.duration


class GraspServer(Node):
    def __init__(self):
        super().__init__('grasp_server')
        self.srv = self.create_service(GetGrasps, '/get_grasps', self.handle_grasp_request)
        self.get_logger().info('Grasp server ready (executing inference inside a docker container).')

        # Base path inside the docker container and host
        ome_dir = os.path.expanduser("~")
        self.base_path = os.path.join(home_dir, 'graspnet_ws/src', 'contact_graspnet_ros2/contact_graspnet')

        # Whether to parse JSON from stdout or load the .npz file directly
        self.result_loading = "_use_json"  # ["_use_json", "_use_npz"]

        # Frames
        self.base_frame = 'simple_pedestal' # 'panda_link0'  #'simple_workstation'  # 
        self.camera_frame = 'rgbd_camera/camera_link/rgbd_camera'

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


    # ------------------------------------------------------------------
    # Docker inference
    # ------------------------------------------------------------------
    def run_inference_in_docker(self,scene_name) -> str:
        container_name = "contact_graspnet_container"
        # container_name = "magical_lovelace"
        np_path = f"test_data/{scene_name}.npy"

        # cmd = [
        #     "docker", "exec", container_name,
        #     "bash", "-lc",
        #     f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
        #     f"conda run -n contact-graspnet bash compile_pointnet_tfops.sh && "
        #     f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
        #     f"conda run -n contact-graspnet python contact_graspnet/inference.py --np_path={np_path} --local_regions --filter_grasps"
        # ]

         # The shared object we expect if tf_ops are compiled
        compiled_lib = (
            "/root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/"
            "pointnet2/tf_ops/sampling/tf_sampling_so.so"
        )

        compile_cmd = (
            f"if [ ! -f {compiled_lib} ]; then "
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet bash compile_pointnet_tfops.sh; "
            f"fi"
        )

        inference_cmd = (
            "cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet python contact_graspnet/inference.py "
            f"--np_path={np_path} --local_regions --filter_grasps"
        )

        cmd = [
            "docker", "exec", container_name,
            "bash", "-lc", f"{compile_cmd} && {inference_cmd}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Inference failed: {result.stderr}")

        start_marker = "<<<BEGIN_JSON>>>"
        end_marker = "<<<END_JSON>>>"

        json_text = None
        start = result.stdout.find(start_marker)
        end = result.stdout.find(end_marker, start)
        if start != -1 and end != -1:
            json_text = result.stdout[start + len(start_marker):end].strip()
            self.get_logger().info("Extracted JSON using markers.")
        else:
            for line in result.stdout.splitlines():
                if line.strip().startswith("{") and line.strip().endswith("}"):
                    json_text = line.strip()
                    break

        if json_text is None:
            self.get_logger().error(
                f"No JSON found in inference output.\nFirst 500 chars:\n{result.stdout[:500]}"
            )
            raise RuntimeError("Inference did not return valid JSON")

        return json_text

    # ------------------------------------------------------------------
    # Coordinate-frame helpers
    # ------------------------------------------------------------------
    def cgn_optical_to_ros_cam(self, T_cgn: np.ndarray) -> np.ndarray:
        """
        Contact-GraspNet grasps are expressed in the camera *optical* frame:
          x_right, y_down, z_forward.

        The URDF/TF frame `rgbd_camera/camera_link/rgbd_camera` is a standard
        ROS camera_link frame:
          X_forward, Y_left, Z_up.

        This applies the fixed rotation R_opt->cam_link so that the resulting
        4x4 matrix is in the ROS camera_link convention, rooted at the camera.
        """
        R = np.array([
            [0.0,  0.0, 1.0],   # z_opt -> X_cam
            [-1.0, 0.0, 0.0],   # x_opt -> -Y_cam
            [0.0, -1.0, 0.0],   # y_opt -> -Z_cam
        ], dtype=np.float64)

        # R = np.array([
        #     [1.0, 0.0, 0.0],   
        #     [0.0, 1.0, 0.0],   
        #     [0.0, 0.0, 1.0],  
        # ], dtype=np.float64)

        T_ros = np.eye(4, dtype=np.float64)
        T_ros[:3, :3] = R @ T_cgn[:3, :3]
        T_ros[:3, 3] = R @ T_cgn[:3, 3]
        return T_ros

    def transform_pose_array(self, pose_array: PoseArray,
                             from_frame: str,
                             to_frame: str) -> PoseArray:
        """
        Transform a PoseArray from `from_frame` to `to_frame` using TF2.
        Returns a new PoseArray in the target frame;
        if TF fails, returns the input pose_array.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time(),  # latest
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup {to_frame} <- {from_frame} failed: {e}")
            return pose_array

        trans = t.transform.translation
        rot = t.transform.rotation

        # 4x4 transform matrix base <- camera
        T_bc = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        T_bc[0, 3] = trans.x
        T_bc[1, 3] = trans.y
        T_bc[2, 3] = trans.z

        out = PoseArray()
        out.header.frame_id = to_frame
        out.header.stamp = pose_array.header.stamp

        for p in pose_array.poses:
            # Pose in camera frame as 4x4
            q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            T_cp = tft.quaternion_matrix(q)
            T_cp[0, 3] = p.position.x
            T_cp[1, 3] = p.position.y
            T_cp[2, 3] = p.position.z

            # base <- camera <- pose
            T_bp = T_bc @ T_cp

            pos = T_bp[:3, 3]
            q_bp = tft.quaternion_from_matrix(T_bp)

            p_out = Pose()
            p_out.position.x = float(pos[0]) # - 0.45
            p_out.position.y = float(pos[1]) # + 0.05
            p_out.position.z = float(pos[2]) # - 0.7
            p_out.orientation.x = float(q_bp[0])
            p_out.orientation.y = float(q_bp[1])
            p_out.orientation.z = float(q_bp[2])
            p_out.orientation.w = float(q_bp[3])

            out.poses.append(p_out)

        return out

    # ------------------------------------------------------------------
    # Service callback
    # ------------------------------------------------------------------
    def handle_grasp_request(self, request, response):
        self.scene_name = request.scene_name
        self.get_logger().info(f"Running inference in Docker for scene {self.scene_name}...")

        output = self.run_inference_in_docker(self.scene_name)
        self.get_logger().info("Inference finished")

        # ---------------------------------------------------
        # Load inference results (JSON or NPZ)
        # ---------------------------------------------------
        if self.result_loading == "_use_json":
            # Save stdout for debugging
            out_path = f"{self.base_path}/results/inference_output_{self.scene_name}.txt"
            with open(out_path, "w") as f:
                f.write(output)
            self.get_logger().info(f"Saved raw inference output to {out_path}")

            results = json.loads(output)
            pred_grasps_cam = {
                k: [np.array(g) for g in v]
                for k, v in results["pred_grasps_cam"].items()
            }
            scores = {k: np.array(v) for k, v in results["scores"].items()}
            contact_pts = {k: np.array(v) for k, v in results["contact_pts"].items()}

            self.get_logger().info(
                f"Received grasp results from docker for scene {self.scene_name}:"
            )

        elif self.result_loading == "_use_npz":
            result_path = os.path.join(
                self.base_path, "results", f"predictions_{self.scene_name}.npz"
            )
            data = np.load(result_path, allow_pickle=True)
            pred_grasps_cam = data["pred_grasps_cam"].item()
            scores = data["scores"].item()
            contact_pts = data["contact_pts"].item()

            self.get_logger().info(
                f"Loaded grasp results from docker for scene {self.scene_name}:"
            )

        # ---------------------------------------------------
        # Build PoseArray in ROS camera_link frame
        # (first convert CGN optical -> camera_link)
        # ---------------------------------------------------
        grasps_cam_pa = PoseArray()
        grasps_cam_pa.header.frame_id = self.camera_frame
        grasps_cam_pa.header.stamp = self.get_clock().now().to_msg()

        score_list = []
        sample_list = []
        object_list = []

        for obj_id, T_list in pred_grasps_cam.items():
            obj_scores = scores[obj_id]
            obj_samples = contact_pts[obj_id]

            for T_cgn, score, sample in zip(T_list, obj_scores, obj_samples):
                # 1) CGN optical frame -> ROS camera_link
                T_cam = self.cgn_optical_to_ros_cam(T_cgn)


                # ----- NEW: constant “gripper frame” rotation offset -----
                #
                # This encodes the fixed transform between the Contact-GraspNet
                # grasp frame and your robot’s *gripper frame* (e.g. panda_hand).
                #
                # Start with identity; you can tune rx, ry, rz as needed.
                # (values are in *degrees* here for convenience)
                rx_deg, ry_deg, rz_deg = 0.0, 0.0, 0.0
                rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

                # 4x4 homogeneous transform in the *grasp frame*:
                # T_grasp->gripper  (rotation only, no translation)
                T_gripper_offset = tft.euler_matrix(rx, ry, rz, 'sxyz')
                T_gripper_offset[0:3, 3] = [0.0, 0.0, 0.0]

                # If you decide you *also* want the SceneReplica-style X/Y swap,
                # you can fold it into the same offset like this:
                
                swap_xy = np.array([
                    [0.,  1., 0., 0.],
                    [-1., 0., 0., 0.],
                    [0.,  0., 1., 0.],
                    [0.,  0., 0., 1.],
                ])
                T_gripper_offset = T_gripper_offset @ swap_xy

                 # ----- NEW: apply constant gripper-frame rotation -----
                # T_cam is (camera_link -> CGN_graspFrame).
                # We want (camera_link -> robot_gripperFrame):
                T_cam = T_cam @ T_gripper_offset
                # ------------------------------------------------------


                 # 2) Convert to Pose in camera frame
                ros_pose = Pose()
                ros_pose.position.x = float(T_cam[0, 3])
                ros_pose.position.y = float(T_cam[1, 3])
                ros_pose.position.z = float(T_cam[2, 3])

                quat = tfs.quaternion_from_matrix(T_cam)
                ros_pose.orientation.x = float(quat[0])
                ros_pose.orientation.y = float(quat[1])
                ros_pose.orientation.z = float(quat[2])
                ros_pose.orientation.w = float(quat[3])

                grasps_cam_pa.poses.append(ros_pose)

                score_list.append(float(score))
                sample_list.append(
                    Point(
                        x=float(sample[0]),
                        y=float(sample[1]),
                        z=float(sample[2]),
                    )
                )
                object_list.append(int(float(obj_id)))

            self.get_logger().info(
                f"Obtained {len(obj_scores)} grasps for object {obj_id}"
            )

        # Optional: quick Z debug
        if grasps_cam_pa.poses:
            zs_cam = [p.position.z for p in grasps_cam_pa.poses]
            self.get_logger().info(
                f"Camera-frame grasp Z range: [{min(zs_cam):.3f}, {max(zs_cam):.3f}]"
            )

        # ---------------------------------------------------
        # Transform PoseArray cam -> base using TF2
        # ---------------------------------------------------
        grasps_in_base_pa = self.transform_pose_array(
            grasps_cam_pa,
            from_frame=self.camera_frame,
            to_frame=self.base_frame,
        )

        if grasps_in_base_pa.poses:
            zs_base = [p.position.z for p in grasps_in_base_pa.poses]
            self.get_logger().info(
                f"Base-frame grasp Z range:   [{min(zs_base):.3f}, {max(zs_base):.3f}]"
            )

        # ---------------------------------------------------
        # Fill Grasps msg (poses now in base frame)
        # ---------------------------------------------------
        grasps_msg = Grasps()
        grasps_msg.poses = list(grasps_in_base_pa.poses)
        grasps_msg.scores = score_list
        grasps_msg.samples = sample_list
        grasps_msg.object_ids = object_list

        response.grasps = grasps_msg

        self.get_logger().info(
            f"Responded with {len(grasps_msg.poses)} grasps in frame "
            f"'{self.base_frame}' for scene {self.scene_name}"
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()