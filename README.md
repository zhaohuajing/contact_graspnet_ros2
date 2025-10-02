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

## Prerequisites  

- **ROS 2 Jazzy** (or compatible distro) installed on host.  
- **Docker** with GPU runtime enabled (`nvidia-docker2` or `nvidia-container-toolkit`).  
- **Built Docker image** for Contact-GraspNet (see `Dockerfile`).  

---

## Usage  

1. **Start the Docker container** (if not already running):  
   ```bash
   ./run_docker.sh
   ```
This script launches the Contact-GraspNet container with the proper environment and names it:
	```contact_graspnet_container
	```

2. **Compile the ROS 2 package** (needed after code changes; assuming put under ~/graspnet_ws/src):
	```cd ~/graspnet_ws
	colcon build --symlink-install
	source install/setup.bash
	```
3. **Run the ROS 2 server** (in one terminal):
	```cd ~/graspnet_ws/src/contact_graspnet_ros2
	python grasp_executor_server.py
	```
4. **Run the ROS 2 client** (in another terminal):
	```cd ~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet_ros2
	python client_grasp_request.py <scene_id>
	```
This requests grasps for test_data/<scene_id>.npy.

---

## Notes

 - The server uses subprocess + docker exec to call inference inside the container.
 - Inference results are serialized to JSON (<<<BEGIN_JSON>>> ... <<<END_JSON>>>) inside Docker and parsed by the server. If JSON extraction fails, the server falls back to raw line parsing for robustness.
 - You can extend this wrapper for other perception or grasp planning modules by reusing the same server–client communication pattern.