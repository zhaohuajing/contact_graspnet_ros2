# Contact-GraspNet ROS 2 Wrapper  

This package provides a ROS 2 service–client wrapper around **Contact-GraspNet**, using a `subprocess` call inside the ROS 2 server to run grasp inference in a Docker container.  

This design allows us to:  
- Keep ROS 2 running on the host system (e.g., Python 3.12, CUDA 12.2).  
- Execute Contact-GraspNet inference in a controlled environment (Docker with Python 3.9, CUDA 11.8).  
- Cleanly return grasp results (`pred_grasps_cam`, `scores`, `contact_pts`) to the ROS 2 ecosystem.  

The same approach can be extended to other grasp planners or perception algorithms (e.g., **UnseenObjectClustering**) running in Docker or conda environments.  

---

## Architecture  

```text
+-----------------+        +-------------------+        +------------------+
|   ROS2 Client   | -----> |   ROS2 Server     | -----> |   Docker (CGN)   |
| (grasp request) |        | (subprocess call) | <----- |   inference.py   |
+-----------------+        +-------------------+        +------------------+
        ^                             |                   (grasp planning)                     
        |                             v                               
        +-----  Grasp Results  <------+
```

Flow:  
1. Client sends a scene ID to the server.  
2. Server launches `inference.py` inside Docker via `subprocess`.  
3. Inference produces grasp predictions.  
4. Server extracts predictions (`pred_grasps_cam`, `scores`, `contact_pts`) from JSON.  
5. Results are returned to the client as a ROS 2 message.  

---

## Setup instruction  

#### 1. Prerequisites:

- **ROS 2 Jazzy** (or compatible distro) installed on host.  
- **Docker** with GPU runtime enabled (`nvidia-docker2` or `nvidia-container-toolkit`).  
- **Built Docker image** for Contact-GraspNet (see `Dockerfile_CGN`).  

- Assume repository `contact_graspnet_ros2` is put under `~/graspnet_ws/src`

#### 2. Setup Docker container:

- **Download the Docker files**: 

	```bash
	   git clone https://github.com/zhaohuajing/contact_graspnet_docker 
	```

- **Build the Docker image**:
	```bash
	docker build -t cuda118:contact_graspnet -f Dockerfile_CGN .
	```
	Alternatively, you may use the following command to pull the docker image for contact-graspnet from docker hub:
   ```
   docker pull zhaohuajing/cuda118:contact_graspnet
   ```

- **Start the Docker container**:  
   ```bash
   ./run_docker.sh
   ```
	This script launches the Contact-GraspNet container with the proper environment and names it as: `contact_graspnet_container`


#### 3. Compile the ROS 2 package:

- Start an new terminal on the host machine (i.e., outside of the docker container). Assume repository `contact_graspnet_ros2` is put under `~/graspnet_ws/src`, run the following command:
	```bash
	cd ~/graspnet_ws
	colcon build --symlink-install
	source install/setup.bash
	```

#### 4. Test run of the ROS 2 server WITHOUT real-time inputs:

- Both server and client commands should run on the host machine (i.e., outside of the docker container).

- **Run the test ROS 2 server (in one terminal)**:

	```bash
	ros2 run contact_graspnet_ros2 grasp_executor_server
	```
	
- **Run the test ROS 2 client (in another terminal)**:
	```bash
	ros2 run contact_graspnet_ros2 client_grasp_request <scene_name>
	```
This requests grasps for test_data/<scene_name>.npy.


#### Notes

 - The server uses subprocess + docker exec to call inference inside the container.
 - You can extend this wrapper for other perception or grasp planning modules by reusing the same server–client communication pattern.

 ---


 ## Real-time integrations for ROS 2 grasp servers

This repository provides **two complementary ROS 2 wrappers for Contact-GraspNet**, enabling integration with real-world sensor inputs depending on the perception pipeline and available modalities.

 ### 1. Real-time RGB-D Scene Integration (**Recommended**)
We introduce a ROS 2 server, **`grasp_executor_rgbd_server`**, which enables Contact-GraspNet to operate directly on **live RGB-D scenes** (e.g., from Gazebo or a physical camera), instead of only static, pre-generated datasets.

**Key features:**
- ROS 2 service interface for grasp requests.
- Converts live RGB-D inputs (and optional instance segmentation outputs) into Contact-GraspNet-compatible scene files.
- Launches Contact-GraspNet inference **inside Docker via `subprocess`**, enabling:
  - ROS 2 on the host (modern Python, CUDA, drivers).
  - Contact-GraspNet running in a controlled container environment.
- Parses inference outputs and returns grasp poses to ROS 2 clients for planning and execution.


**Run the ROS 2 server with Live RGB-D inputs**
```bash
ros2 run contact_graspnet_ros2 grasp_executor_rgbd_server
```

The **RGB-D wrapper** is designed for perception pipelines that start from synchronized color and depth images. This variant:

- Accepts live RGB-D images from simulation (Gazebo) or physical cameras  
- Converts RGB-D observations into Contact-GraspNet scene representations  
- Applies explicit camera and gripper frame alignment for correct TF integration  
- Works **out of the box** with our ROS 2 Unseen Object Clustering wrapper  


This design supports modular integration with upstream perception modules, including:
- **Unseen Object Clustering (ROS 2 wrapper)**  
  https://github.com/zhaohuajing/unseen_obj_clst_ros2
- Other RGB-D or image-based object detection and segmentation algorithms.

A **full perception-to-action pipeline example** using FlexBE state machines is available at:
- https://github.com/zhaohuajing/compare_flexbe  
  (branch: `feature/cgn`)


This enables a full **RGB-D → segmentation → grasp planning → MoveIt** pipeline without requiring intermediate point cloud processing by the user. For this reason, the RGB-D interface is currently the **recommended entry point** for end-to-end perception-to-action workflows in both simulation and real hardware.

---

### 2. PointCloud Scene Integration (NOT recommended)

We also provide a **point cloud–based Contact-GraspNet wrapper**, intended for pipelines that operate directly on 3D geometry rather than images. This variant:

- Accepts point clouds as input  
- Bypasses RGB-D image handling and segmentation  
- Supports Contact-GraspNet inference on raw or preprocessed point clouds  

**Run the ROS 2 server with Live PointCloud inputs**

```bash
ros2 run contact_graspnet_ros2 grasp_executor_cloud_server
```

However, this point cloud interface is **not directly compatible** with the Unseen Object Clustering RGB-D pipeline provided in this repository, which operates on image-based segmentation. Instead, it is better suited for integration with:
- Point cloud–based object detection or segmentation models  
- Scene reconstruction or multi-view fusion pipelines  
- External perception systems that already output filtered or labeled point clouds  

With appropriate upstream perception, the point cloud wrapper can be used as an alternative grasp planning backend, but it requires the user to manage object isolation and point cloud preparation externally.


---

## Additional Features: Frame Alignment Between Contact-GraspNet and ROS TF

A major contribution of this work is the **explicit and correct alignment of frame conventions** between Contact-GraspNet and standard ROS TF / URDF definitions.

### Camera frame alignment

Contact-GraspNet internally represents grasps in the **camera optical frame** which mismatches with ROS camera frames (e.g., `camera_link`).

We apply a fixed rotation: `R_optical → camera_link` to map Contact-GraspNet grasp poses into the ROS TF tree correctly. This resolves systematic position errors such as grasps floating above the table or shifted laterally.

### Gripper / end-effector frame alignment

Contact-GraspNet’s **grasp frame** does not exactly match the Panda gripper (`panda_hand`) convention used by ROS and MoveIt.

Based on inspection of prior implementations (e.g., SceneReplica), we introduce an additional **constant gripper-frame rotation** to reconcile differences in:
- Palm orientation
- End-effector X/Y axis definitions

After applying both:
1. Camera optical → ROS camera frame rotation, and  
2. Contact-GraspNet grasp frame → Panda gripper frame rotation,

the resulting grasp poses are:
- Correctly aligned in position,
- Correctly oriented for execution,
- Directly usable by MoveIt without ad-hoc offsets.

These transformations are implemented in `grasp_executor_rgbd_server.py` and documented inline.
