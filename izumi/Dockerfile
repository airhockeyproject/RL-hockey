# ========================================
# ベース：CUDA + Ubuntu20.04（ROS2 Foxy対応）
# ========================================
FROM nvidia/cuda:12.4.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# ========================================
# 1. 基本パッケージとPython3.8のインストール
# ========================================
RUN apt-get update && apt-get install -y \
    lsb-release \
    curl \
    gnupg2 \
    software-properties-common \
    python3 python3-dev python3-venv python3-pip \
    libgl1-mesa-glx libegl1 libglfw3 libglew-dev libosmesa6-dev \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    x11-apps mesa-utils ffmpeg git git-lfs nano vim unzip wget \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Pythonエイリアス設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# git-lfs初期化
RUN git lfs install

# ========================================
# すべてのRUNをbashで実行
# ========================================
SHELL ["/bin/bash", "-c"]

# ========================================
# 2. ROS 2 Foxyのインストール
# ========================================
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

RUN apt-get update && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    add-apt-repository universe && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list && \
    apt-get update && \
    apt-get install -y ros-foxy-desktop ros-foxy-xacro && \
    apt-get clean

# colconインストール
RUN apt-get update && apt-get install -y python3-colcon-common-extensions

# よく使うROS 2実行系ツール & GUIツール
RUN apt-get update && apt-get install -y \
    ros-foxy-ros2launch \
    ros-foxy-rqt \
    ros-foxy-rqt-common-plugins \
    ros-foxy-joint-state-publisher-gui \
    ros-foxy-rviz2 \
    ros-foxy-xterm \
    && apt-get clean

# ROS 2環境を.bashrcに追加
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

# ========================================
# 3. Python パッケージのインストール（MuJoCo, Gym, etc）
# ========================================
ENV MUJOCO_GL=egl
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    "numpy>=1.20" \
    mujoco \
    PyOpenGL==3.1.6 \
    matplotlib \
    imageio \
    mediapy \
    "stable-baselines3[extra]" \
    "gymnasium[mujoco]"

# ========================================
# 4. ROS2ワークスペースとCrane_X7の構築
# ========================================
WORKDIR /workspace
RUN git clone https://github.com/rt-net/crane_x7_description.git && \
    cd crane_x7_description && git lfs pull

RUN mkdir -p /workspace/ros2_ws/src && \
    mv /workspace/crane_x7_description /workspace/ros2_ws/src/

# colcon build
RUN source /opt/ros/foxy/setup.bash && \
    cd /workspace/ros2_ws && \
    colcon build --symlink-install

# URDF展開（use_gazebo:=false）
RUN source /opt/ros/foxy/setup.bash && \
    source /workspace/ros2_ws/install/setup.bash && \
    mkdir -p /workspace/ros2_ws/src/crane_x7_description/urdf_generated && \
    xacro /workspace/ros2_ws/src/crane_x7_description/urdf/crane_x7.urdf.xacro use_gazebo:=false > /workspace/ros2_ws/src/crane_x7_description/urdf_generated/crane_x7.urdf

# 起動時にROS2環境を読み込む
RUN echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc
