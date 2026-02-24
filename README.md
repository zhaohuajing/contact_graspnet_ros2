# Contact-GraspNet ROS 2 Wrapper  

This package provides a ROS 2 service–client wrapper around **Contact-GraspNet**, using a `subprocess` call inside the ROS 2 server to run grasp inference in a Docker container.  

This design allows us to:  
- Keep ROS 2 running on the **host system** compatible with **ROS 2** (e.g., Ubuntu 24.04 with Python 3.12, CUDA 12.2), which uses **subprocess** bridge to call the source code inside a Docker container.  
- Execute Contact-GraspNet inference in a controlled **Docker environment** (e.g., Ubuntu 22.04 with Ubuntu 22.04 Python 3.9, CUDA 11.8).  
- Cleanly return grasp results (`pred_grasps_cam`, `scores`, `contact_pts`) to the ROS 2 ecosystem.
- Addded additional features for Frame Alignment Between Contact-GraspNet and ROS TF for robot manipulation planning.

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

- **ROS 2 Jazzy** (or compatible distro) installed on host (i.e., Ubuntu 24.04).  
- **Docker** with GPU runtime enabled (`nvidia-docker2` or `nvidia-container-toolkit`).

#### 2. Setup the ros2 server and the source code for Contact-Graspnet:

- **Clone this repo for ros2 server** (assume repository `contact_graspnet_ros2` is put under `~/graspnet_ws/src`):
	```bash
	cd ~/graspnet_ws/src/
	git clone -b ros2_server https://github.com/zhaohuajing/contact_graspnet_ros2.git
	```

- **Clone the contact_graspnet source repo** from: https://github.com/zhaohuajing/compare_contact_graspnet.git and create local folder named `contact_graspnet`:
	```bash
	cd ~/graspnet_ws/src/contact_graspnet_ros2/
	git clone https://github.com/zhaohuajing/compare_contact_graspnet contact_graspnet
	```

#### 3. Setup Docker container for Contact-Graspnet:

We use `Dockerfile_CGN` from https://github.com/zhaohuajing/contact_graspnet_docker to build a docker image with Base image: CUDA 11.8 with cuDNN 8 on Ubuntu 22.04 for Contact-GraspNet. We use `run_docker.sh` to start the docker container, which at the same time mount the local workspace (i.e., `~/graspnet_ws/src`) from the host machine to the docker container.

- **Clone the Docker files** (CUDA 11.8 with cuDNN 8 on Ubuntu 22.04): 

	```bash
 	cd ~/graspnet_ws/src/contact_graspnet_ros2/
	git clone https://github.com/zhaohuajing/contact_graspnet_docker 
	```

- **Build the Docker image**:
	```bash
 	cd ~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet_docker
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
<<<<<<< HEAD
	This script launches the Contact-GraspNet container with the proper environment and names it as: `contact_graspnet_container`. Note that `run_docker.sh` script will mount the entire workspace (i.e., `~/graspnet_ws/src`) to the docker container through `-v ~/graspnet_ws:/root/graspnet_ws`; you may adjust the name of workspace to your local setup as needed.

	Once the container is running, simply leave that terminal open. No manual commands need to be executed inside the container. All ROS 2 server interactions are initiated from separate terminals on the host machine. These ROS 2 nodes use subprocess calls to automatically enter the running container and execute the required inference scripts internally.

=======
	This script launches the Contact-GraspNet container with the proper environment and names it as: `contact_graspnet_container`. Note that `run_docker.sh` script will mount the entire workspace (i.e., `~/graspnet_ws/src`) to the docker container through `-v ~/graspnet_ws:/root/graspnet_ws`; you may adjust the name of workspace to your local setup as needed. 
	Once the container is running, simply leave that terminal open. No manual commands need to be executed inside the container. All ROS 2 server interactions are initiated from separate terminals on the host machine. These ROS 2 nodes use subprocess calls to automatically enter the running container and execute the required inference scripts internally.
>>>>>>> refs/remotes/origin/ros2_server

#### 4. Compile the ROS 2 package:

- Start an new terminal on the host machine (i.e., outside of the docker container). Assume repository `contact_graspnet_ros2` is put under `~/graspnet_ws/src`, run the following command:
	```bash
	cd ~/graspnet_ws
	colcon build --symlink-install
	source install/setup.bash
	```

#### 5. Test run of the ROS 2 server WITHOUT real-time inputs:

Both **server** and **client** commands should run on the host machine (i.e., Ubuntu 24.04 compatible with **ROS 2 Jazzy** outside of the docker container).

- **Run the test ROS 2 server (in one terminal)**:

	```bash
	ros2 run contact_graspnet_ros2 grasp_executor_server
	```
	
- **Run the test ROS 2 client (in another terminal)**:
	```bash
	ros2 run contact_graspnet_ros2 client_grasp_request <scene_name>
	```

This requests grasps for `~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/test_data/<scene_name>.npy`. Example `<scene_name>` can be `0`, `1`, ..., `13`.


#### Notes

 - The server uses subprocess + docker exec to call inference inside the container.
 - You can extend this wrapper for other perception or grasp planning modules by reusing the same server–client communication pattern.

 ---


## Real-time integrations for ROS 2 grasp servers

This repository provides **two complementary ROS 2 wrappers for Contact-GraspNet**, enabling integration with real-world sensor inputs depending on the perception pipeline and available modalities.

### 1. Real-time RGB-D Scene Integration (**Recommended**)
We introduce a ROS 2 server, **`grasp_executor_rgbd_server`**, which enables Contact-GraspNet to operate directly on **live RGB-D scenes** (e.g., from Gazebo or a physical camera), instead of only static, pre-generated datasets.

**Key features:**
- ROS 2 service interface `contact_graspnet_ros2/grasp_executor_rgbd_server` for grasp requests.
- Converts live RGB-D inputs and optional instance segmentation features (e.g., check [`contact_graspnet/contact_graspnet/test_data/sample3/`](https://github.com/zhaohuajing/compare_contact_graspnet/tree/main/test_data/sample_scene_ucn/sample_3)) into Contact-GraspNet-compatible scene files.
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
- https://github.com/zhaohuajing/compare_flexbe    (branch: `feature/cgn`)

This enables a full **RGB-D → segmentation → grasp planning → MoveIt** pipeline without requiring intermediate point cloud processing by the user. For this reason, the RGB-D interface is currently the **recommended entry point** for end-to-end perception-to-action workflows in both simulation and real hardware.


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

## Additional Features for Real-time integrations

### Frame Alignment Between Contact-GraspNet and ROS TF

A major contribution of this work is the **explicit and correct alignment of frame conventions** between Contact-GraspNet and standard ROS TF / URDF definitions in **`grasp_executor_rgbd_server`**.

#### Camera frame alignment

Contact-GraspNet internally represents grasps in the **camera optical frame** which mismatches with ROS camera frames (e.g., `camera_link`).

We apply a fixed rotation: `R_optical → camera_link` to map Contact-GraspNet grasp poses into the ROS TF tree correctly. This resolves systematic position errors such as grasps floating above the table or shifted laterally.

#### Gripper / end-effector frame alignment

Contact-GraspNet’s **grasp frame** does not exactly match the Panda gripper (`panda_hand`) convention used by ROS and MoveIt.

Based on inspection of prior implementations (e.g., SceneReplica), we introduce an additional **constant gripper-frame rotation** to reconcile differences in:
- Palm orientation
- End-effector X/Y axis definitions

#### Note:
After applying both:
1. Camera optical → ROS camera frame rotation, and  
2. Contact-GraspNet grasp frame → Panda gripper frame rotation,

the resulting grasp poses are:
- Correctly aligned in position,
- Correctly oriented for execution,
- Directly usable by MoveIt without ad-hoc offsets.

These transformations are implemented in `grasp_executor_rgbd_server.py` and documented inline.

