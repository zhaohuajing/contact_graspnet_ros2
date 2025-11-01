#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point
from contact_graspnet_ros2.srv import GetGrasps
from contact_graspnet_ros2.msg import Grasps

# tf transformations for 4x4 -> quaternion
import tf_transformations as tfs


class GraspServer(Node):
    """
    ROS 2 service that runs Contact-GraspNet inference inside a Docker container.
    Accepts either a live Nx3 point cloud (flattened) or a scene_id referencing a sample .npy.
    Returns grasps via JSON parsed from inference stdout.
    """

    def __init__(self):
        super().__init__('grasp_server')

        # ---------- Config ----------
        # Absolute service name avoids namespace discovery issues
        self.srv = self.create_service(GetGrasps, '/get_grasps', self.handle_grasp_request)

        # Container + paths
        self.container_name = 'contact_graspnet_container'  # set to your container name
        # Host repo path (bind-mounted to /root/graspnet_ws/src/... inside container)
        self.host_repo = os.path.expanduser('~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet')

        # Results loading mode: JSON only (robust; no host/container file juggling)
        self.result_loading = '_use_json'

        # Whether your *sample* scenes include segmentation (segmap); if True we can enable region/filter flags for samples
        self.sample_has_segmap = False  # set True if your samples ship with segmaps and you want local regions

        self.get_logger().info('Grasp server ready (Docker-based inference).')

    # ------------------------------- Inference helpers -------------------------------

    def _ensure_repo_dirs(self):
        os.makedirs(os.path.join(self.host_repo, 'test_data'), exist_ok=True)
        os.makedirs(os.path.join(self.host_repo, 'results'), exist_ok=True)

    def _save_live_npy(self, points_flat, mask=None) -> str:
        """
        Save live Nx3 float32 to the bind-mounted repo so the container can read it.
        Returns container-relative path used by inference (--np_path).
        """
        self._ensure_repo_dirs()

        pts = np.asarray(points_flat, dtype=np.float32).reshape((-1, 3))
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.uint32) > 0
            if mask_arr.shape[0] == pts.shape[0]:
                pts = pts[mask_arr]

        # Deterministic name that inference.py will map to predictions_scene_live.npz/json
        host_np = os.path.join(self.host_repo, 'test_data', 'scene_live.npy')
        np.save(host_np, pts)
        return 'test_data/scene_live.npy'  # relative to repo root inside container

    def _extract_json_from_stdout(self, text: str) -> str | None:
        start_marker, end_marker = '<<<BEGIN_JSON>>>', '<<<END_JSON>>>'
        s, e = text.find(start_marker), text.find(end_marker)
        if s != -1 and e != -1:
            return text[s + len(start_marker):e].strip()
        # Fallback: scan first JSON-looking line
        for line in text.splitlines():
            l = line.strip()
            if l.startswith('{') and l.endswith('}'):
                return l
        return None

    def run_inference_in_docker(self, np_relpath: str, *, use_local_regions: bool, use_filter_grasps: bool) -> dict:
        """
        Execute inference.py in the container and return parsed JSON dict.
        """
        compiled_lib = "/root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/pointnet2/tf_ops/sampling/tf_sampling_so.so"

        compile_cmd = (
            f"if [ ! -f {compiled_lib} ]; then "
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet bash compile_pointnet_tfops.sh; "
            f"fi"
        )

        flags = []
        if use_local_regions:
            flags.append('--local_regions')
        if use_filter_grasps:
            flags.append('--filter_grasps')
        flags_str = ' '.join(flags)

        inference_cmd = (
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet python contact_graspnet/inference.py "
            f"--np_path={np_relpath} {flags_str}"
        )

        cmd = ["docker", "exec", self.container_name, "bash", "-lc", f"{compile_cmd} && {inference_cmd}"]
        self.get_logger().info(f"Running in container: {' '.join(cmd[-1:])}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.get_logger().error(f"Inference failed (rc={result.returncode}): {result.stderr}")
            raise RuntimeError('Contact-GraspNet inference failed')

        json_text = self._extract_json_from_stdout(result.stdout)
        if not json_text:
            self.get_logger().error('No JSON block found in inference output.')
            # Log a short tail of stdout for context
            tail = '\n'.join(result.stdout.splitlines()[-15:])
            self.get_logger().error(f'Inference stdout tail:\n{tail}')
            raise RuntimeError('Missing JSON from Contact-GraspNet')

        try:
            data = json.loads(json_text)
        except Exception as e:
            self.get_logger().error(f'JSON parse error: {e}')
            self.get_logger().error(f'JSON text was: {json_text[:500]}...')
            raise
        return data

    # ------------------------------- Service callback -------------------------------

    def handle_grasp_request(self, request, response):
        """
        Builds input (.npy) for the container, runs inference, converts results to ROS types.
        """
        try:
            # Decide live vs sample
            use_live = len(request.points) >= 3

            if use_live:
                self.get_logger().info('Live point cloud received; preparing scene_live.npy')
                np_relpath = self._save_live_npy(request.points, getattr(request, 'mask', None))
                # Live Nx3 has no segmap → don’t use seg-dependent flags
                use_local_regions = False
                use_filter_grasps = False
            else:
                scene_id = int(request.scene_id)
                self.get_logger().info(f'No live points. Using sample scene {scene_id}.')
                np_relpath = f'test_data/{scene_id}.npy'
                # Only enable seg-dependent flags if your samples truly have segmaps
                use_local_regions = self.sample_has_segmap
                use_filter_grasps = self.sample_has_segmap

            # Run inference in Docker and parse JSON
            data = self.run_inference_in_docker(
                np_relpath,
                use_local_regions=use_local_regions,
                use_filter_grasps=use_filter_grasps
            )

            # Expect keys: pred_grasps_cam, scores, contact_pts (by object id)
            pred_grasps_cam = {k: [np.array(g) for g in v] for k, v in data.get('pred_grasps_cam', {}).items()}
            scores = {k: np.array(v) for k, v in data.get('scores', {}).items()}
            contact_pts = {k: np.array(v) for k, v in data.get('contact_pts', {}).items()}

            total = sum(len(v) for v in pred_grasps_cam.values())
            self.get_logger().info(f'Total grasps predicted: {total}')
            if total == 0:
                self.get_logger().warn('No grasps returned by CGN (check input frame, flags, or cloud quality).')

            # Pack ROS message
            pose_list, score_list, sample_list, object_list = [], [], [], []
            for obj_id_str, T_list in pred_grasps_cam.items():
                # Some JSON writers may use string keys; keep as-is but coerce to int when storing.
                try:
                    obj_id = int(float(obj_id_str))
                except Exception:
                    obj_id = 0
                sc = scores.get(obj_id_str, [])
                cp = contact_pts.get(obj_id_str, [])
                for i, T in enumerate(T_list):
                    if T.shape != (4, 4):
                        continue
                    ros_pose = Pose()
                    ros_pose.position.x = float(T[0, 3])
                    ros_pose.position.y = float(T[1, 3])
                    ros_pose.position.z = float(T[2, 3])
                    q = tfs.quaternion_from_matrix(T)
                    ros_pose.orientation.x = float(q[0])
                    ros_pose.orientation.y = float(q[1])
                    ros_pose.orientation.z = float(q[2])
                    ros_pose.orientation.w = float(q[3])

                    pose_list.append(ros_pose)
                    # align score/sample lengths defensively
                    score_list.append(float(sc[i]) if i < len(sc) else 0.0)
                    if i < len(cp):
                        sample = Point(x=float(cp[i][0]), y=float(cp[i][1]), z=float(cp[i][2]))
                    else:
                        sample = Point()
                    sample_list.append(sample)
                    object_list.append(obj_id)

            grasps_msg = Grasps()
            grasps_msg.poses = pose_list
            grasps_msg.scores = score_list
            grasps_msg.samples = sample_list
            grasps_msg.object_ids = object_list
            response.grasps = grasps_msg

            self.get_logger().info(f'Responded with {len(pose_list)} grasps.')
            return response

        except Exception as e:
            self.get_logger().error(f'handle_grasp_request failed: {e}')
            # On failure, return empty response; FlexBE state will emit 'failed'
            return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
