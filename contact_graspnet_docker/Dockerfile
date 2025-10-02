# Base image: CUDA 11.8 with cuDNN 8 on Ubuntu 22.04 (includes dev‑tools)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# NOTE: To build docker, run: "docker build -t contact_graspnet:cuda118 ."

# Remove dangling images left from previous builds
docker image prune -f

# Force Qt (used by Mayavi/VTK) into offscreen rendering mode
ENV QT_QPA_PLATFORM=offscreen

# Install system utilities
RUN apt-get update && apt-get install -y \
        wget \
        git \
        libxrender1 \
        mesa-common-dev \
        tree \
        sudo \
    && rm -rf /var/lib/apt/lists/*


# Install Sublime Text: Install prerequisites first
RUN apt-get update && apt-get install -y wget gnupg2 apt-transport-https ca-certificates
# Add Sublime Text GPG key
RUN wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor > /usr/share/keyrings/sublimehq-archive-keyring.gpg
# Add Sublime Text repository
RUN echo "deb [signed-by=/usr/share/keyrings/sublimehq-archive-keyring.gpg] https://download.sublimetext.com/ apt/stable/" \
    > /etc/apt/sources.list.d/sublime-text.list
# Install Sublime Text
RUN apt-get update && apt-get install -y sublime-text


# Set up Miniconda installation directory
ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda init bash \
    && conda config --system --set auto_activate_base false

# Use bash login shell so that .bashrc (with conda init) is sourced
SHELL ["/bin/bash", "-lc"]

# WORKDIR /workspace/contact_graspnet
WORKDIR /root/graspnet_ws


# Clone the Contact-GraspNet source
RUN git clone https://github.com/JohnBrann/contact_graspnet .

# Remove the mayavi pin from environment.yaml (we'll install via conda-forge)
RUN sed -i '/mayavi==/d' environment.yaml

# added: Accept ToS inside Dockerfile
RUN conda config --add channels defaults && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Build the conda env and install extras: 
RUN conda env create -f environment.yaml \
    && conda install -n contact-graspnet -c conda-forge vtk mayavi matplotlib -y \
    && conda install -n contact-graspnet numpy=1.23.5 -y \
    && conda run -n contact-graspnet pip install --no-cache-dir pyyaml==5.3.1

# added: Install EGL/OSMesa support inside the Dockerfile
RUN apt-get update && apt-get install -y \
    libosmesa6-dev libglu1-mesa-dev freeglut3-dev mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages for ROS 2 compatibility
RUN /opt/conda/envs/contact-graspnet/bin/pip install \
    empy==3.3.4 \
    lark-parser \
    setuptools==58.0.4 \
    catkin_pkg \
    rospkg

# added (temp: Dockerfile-based installation not tested) Install ROS 2 Humble base + build tools
# Set environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install basic dependencies and ROS 2 Humble
RUN apt update && apt install -y \
    curl \
    gnupg \
    lsb-release \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
    && echo "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2-latest.list \
    && apt update && apt install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-pip \
    python3-vcstool \
    python3-rosdep \
    build-essential \
    git \
    && rosdep init && rosdep update

# Source ROS 2 setup in every shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc


RUN echo "conda activate contact-graspnet" >> ~/.bashrc

# Default to an interactive bash login shell
CMD ["bash", "-l"]
