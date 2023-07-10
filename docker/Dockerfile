FROM ubuntu:20.04
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install vim wget nano curl git git-lfs ca-certificates -y
RUN apt-get update
RUN apt-get install -y gnupg2

RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-focal.list
RUN curl http://repo.ros2.org/repos.key | apt-key add -
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install ros-noetic-desktop-full git-lfs ros-noetic-object-recognition-msgs ros-noetic-moveit-core ros-noetic-moveit-ros-perception ros-noetic-moveit-ros-planning-interface ros-noetic-velocity-controllers ros-noetic-twist-mux python3-rostopic ros-noetic-effort-controllers ros-noetic-position-controllers ros-noetic-joint-trajectory-controller ros-noetic-moveit ros-noetic-rviz-visual-tools ros-noetic-moveit-visual-tools -y
# Nano settings
#Clone factory_ws
WORKDIR /
# Dependencies and Python
WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN echo 'conda activate main' >> ~/.bashrc
RUN /root/miniconda3/bin/conda create --name main python==3.9
RUN apt-get update && apt-get install -y -qq --no-install-recommends libgl1-mesa-glx libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
RUN /root/miniconda3/bin/conda run -n main conda install -c conda-forge catkin_tools
RUN /root/miniconda3/bin/conda run -n main conda install -c conda-forge rosdep
RUN /root/miniconda3/bin/conda run -n main pip install empy
RUN /root/miniconda3/bin/conda run -n main rosdep init
RUN /root/miniconda3/bin/conda run -n main rosdep update
RUN /root/miniconda3/bin/conda run -n main pip install ipython
RUN /root/miniconda3/bin/conda run -n main pip install opencv-python
RUN /root/miniconda3/bin/conda run -n main pip install cvbridge3
RUN /root/miniconda3/bin/conda run -n main conda install numpy
RUN /root/miniconda3/bin/conda run -n main pip install seaborn thop ipython pybboxes
RUN /root/miniconda3/bin/conda run -n main pip install opencv-python pycocotools matplotlib onnxruntime onnx
RUN /root/miniconda3/bin/conda run -n main pip install wheel
WORKDIR /root/
COPY requirements.txt /root/
RUN /root/miniconda3/bin/conda run -n main pip install -r /root/requirements.txt





RUN apt update && apt install iputils-ping

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get install -y git wget ipython3 python3-rosinstall python3-catkin-tools python3-rosdep python-is-python3 ros-noetic-actionlib-tools ros-noetic-moveit-commander
WORKDIR /workspace