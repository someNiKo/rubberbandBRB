FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    build-essential

RUN curl -sSL "http://packages.ros.org/ros.key" | apt-key add - && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosbag \
    python3-tf \
    python3-matplotlib \
    python3-rosdep

RUN rosdep init && rosdep update

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

WORKDIR /workspace
COPY . /workspace

CMD ["bash"]