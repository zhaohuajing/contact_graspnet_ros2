# Dockerized Contact-GraspNet  
original readme below. All credits to the authors.

This fork provides a (minimal-effort) docker image to facilitate integration of the contact-graspnet grasp predictions. The codebase requires tensorflow 2.5, which makes it hard to integrate directly in your python projects, hence the docker image.

Make sure to make yourself familiar with the original repo first. I recommend running inference on the test examples manually.  You can use this updated [conda env file](./environment.yaml) and follow the original Readme.

Integration example of the dockerized grasp predictor (code available [here](robot/grasping_using_docker.py)):

https://github.com/tlpss/contact_graspnet/assets/37955681/2236c3f0-8157-4cf0-a167-cc2ecae7a4df


## Using the docker image 

The docker image wraps the inference code of the grasp estimator in a Flask webserver for ease of use. Depth images/pointclouds and other input as well as the results are passed using file sharing in a docker volume.

The steps to obtain grasp proposals are:
1. create a numpy npz file that contains a depth map (in meters), rgb image, intrinsics matrix and segmentation map (can have object of interest segmentation or region of interest segmentation, defaults to the entire image):

 `np.save('inputs.npz', rgb=..,depth=.., segmap=..,K=..)`

2. send a request to the webserver endpoint (`localhost:5000/` endpoint)

3. load the grasp proposals and their scores from the resulting npz file.

The grasp proposals are expressed in the base frame of a panda gripper, make sure to transform them to the TCP frame (11cm in Z direction) or to your grippers base frame. (only aspect that matters is distance between fingertips and gripper base).

See `robot/grasping_using_docker.py` for an example on how to do use the webserver. See `robot/dump_pointcloud.py` and `robot/grasp.py` for code to manually create an input file and to execute the best grasp.
These scripts make use of [airo-mono](https://github.com/airo-ugent/airo-mono) to quickly develop a robotic application. Make sure to install it if you want to use the scripts.


Example video can be seen [here](examples/contact-graspnet-dockerized.mp4).

## Starting the docker container 

Make sure you have installed the nvidia container toolkit: https://github.com/NVIDIA/nvidia-container-toolkit.

Run the following command `xhost +local:` in your terminal.

Use the following command in the terminal to start the container. Adapt the mount location to the directory where you will save/load the images.
`docker run --gpus all  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --network=host -v "$(pwd)"/flask_files:/contact_graspnet/flask_files tlpss/contact-graspnet-flask`

Note that this is by no means safe for production (and even then, the original license does not allow commercial use anyways)

You can now access the webserver from your python script on the `localhost:5000/` endpoint after you have saved your inputs in the the `.npz` file.
## Building the container 

- download the checkpoint (see original repo)
- `docker build . -f docker/Dockerfile -t contact-graspnet-flask`

---
original readme



# Contact-GraspNet  

### Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes   
Martin Sundermeyer, Arsalan Mousavian, Rudolph Triebel, Dieter Fox  
ICRA 2021    

[paper](https://arxiv.org/abs/2103.14127), [project page](https://research.nvidia.com/publication/2021-03_Contact-GraspNet%3A--Efficient), [video](http://www.youtube.com/watch?v=qRLKYSLXElM)

<p align="center">
  <img src="examples/2.gif" width="640" title="UOIS + Contact-GraspNet"/>
</p>

## Installation

This code has been tested with python 3.7, tensorflow 2.2, CUDA 11.1

Create the conda env
```
conda env create -f contact_graspnet_env.yml
```

### Troubleshooting

- Recompile pointnet2 tf_ops:
```shell
sh compile_pointnet_tfops.sh
```

### Hardware
Training: 1x Nvidia GPU >= 24GB VRAM, >=64GB RAM  
Inference: 1x Nvidia GPU >= 8GB VRAM (might work with less)

## Download Models and Data
### Model
Download trained models from [here](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing) and copy them into the `checkpoints/` folder.
### Test data
Download the test data from [here](https://drive.google.com/drive/folders/1v0_QMTUIEOcu09Int5V6N2Nuq7UCtuAA?usp=sharing) and copy them them into the `test_data/` folder.

## Inference


Contact-GraspNet can directly predict a 6-DoF grasp distribution from a raw scene point cloud. However, to obtain object-wise grasps, remove background grasps and to achieve denser proposals it is highly recommended to use (unknown) object segmentation [e.g. [1](https://github.com/chrisdxie/uois), [2](https://arxiv.org/abs/2103.06796)] as preprocessing and then use the resulting segmentation map to crop local regions and filter grasp contacts.

Given a .npy/.npz file with a depth map (in meters), camera matrix K and (optionally) a 2D segmentation map, execute:

```shell
python contact_graspnet/inference.py \
       --np_path=test_data/*.npy \
       --local_regions --filter_grasps
```

<p align="center">
  <img src="examples/7.png" width="640" title="UOIS + Contact-GraspNet"/>
</p>
--> close the window to go to next scene

Given a .npy/.npz file with just a 3D point cloud (in meters), execute [for example](examples/realsense_crop_sigma_001.png):
```shell
python contact_graspnet/inference.py --np_path=/path/to/your/pc.npy \
                                     --forward_passes=5 \
                                     --z_range=[0.2,1.1]
```

`--np_path`: input .npz/.npy file(s) with 'depth', 'K' and optionally 'segmap', 'rgb' keys. For processing a Nx3 point cloud instead use 'xzy' and optionally 'xyz_color' as keys.  
`--ckpt_dir`: relative path to checkpooint directory. By default `checkpoint/scene_test_2048_bs3_hor_sigma_001` is used. For very clean / noisy depth data consider `scene_2048_bs3_rad2_32` / `scene_test_2048_bs3_hor_sigma_0025` trained with no / strong noise.   
`--local_regions`: Crop 3D local regions around object segments for inference. (only works with segmap)  
`--filter_grasps`: Filter grasp contacts such that they only lie on the surface of object segments. (only works with segmap)  
`--skip_border_objects` Ignore segments touching the depth map boundary.  
`--forward_passes` number of (batched) forward passes. Increase to sample more potential grasp contacts.  
`--z_range` [min, max] z values in meter used to crop the input point cloud, e.g. to avoid grasps in the foreground/background(as above).  
`--arg_configs TEST.second_thres:0.19 TEST.first_thres:0.23` Overwrite config confidence thresholds for successful grasp contacts to get more/less grasp proposals 


## Training

### Download Data 

Download the Acronym dataset, ShapeNet meshes and make them watertight, following these [steps](https://github.com/NVlabs/acronym#using-the-full-acronym-dataset).

Download the training data consisting of 10000 table top training scenes with contact grasp information from [here](https://drive.google.com/drive/folders/1eeEXAISPaStZyjMX8BHR08cdQY4HF4s0?usp=sharing) and extract it to the same folder:

```
acronym
├── grasps
├── meshes
├── scene_contacts
└── splits
```

### Train Contact-GraspNet

When training on a headless server set the environment variable
```shell
export PYOPENGL_PLATFORM='egl'
```

Start training with config `contact_graspnet/config.yaml`
```
python contact_graspnet/train.py --ckpt_dir checkpoints/your_model_name \
                                 --data_path /path/to/acronym/data
```

### Generate Contact Grasps and Scenes yourself (optional)

The `scene_contacts` downloaded above are generated from the Acronym dataset. To generate/visualize table-top scenes yourself, also pip install the [acronym_tools]((https://github.com/NVlabs/acronym)) package in your conda environment as described in the acronym repository.

In the first step, object-wise 6-DoF grasps are mapped to their contact points saved in `mesh_contacts`

```
python tools/create_contact_infos.py /path/to/acronym
```

From the generated `mesh_contacts` you can create table-top scenes which are saved in `scene_contacts` with

```
python tools/create_table_top_scenes.py /path/to/acronym
```

Takes ~3 days in a single thread. Run the command several times to process on multiple cores in parallel.

You can also visualize existing table-top scenes and grasps

```
python tools/create_table_top_scenes.py /path/to/acronym \
       --load_existing scene_contacts/000000.npz -vis
```

## Citation

```
@article{sundermeyer2021contact,
  title={Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes},
  author={Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```
