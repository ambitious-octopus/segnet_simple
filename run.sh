xhost +local:root
docker build -t segnet_simple docker/
docker run -d -t --net host --gpus all -v $PWD/:/workspace  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri --privileged -v /home/$USER/.Xauthority:/root/.Xauthority segnet_simple