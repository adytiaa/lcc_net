
# KILIDAR Project

## Details of implementation of an online self-calibration method adapted from the paper “LCCNet: Lidar and Camera Self-Calibration Using Cost Volume Network” for High Flash Lidar data.

# Prerequirities

## Ubuntu 20.04.6 LTS
- device: `Nvidia Jetson AGX Orin`
- cuda: `build cuda_11.4.r11.4/compiler.31964100_0`
- ros: `noetic` (ROS v1)
- python3: `Python 3.8.10`
- required python3 packages: `python3 -m pip install -r requirements_ubuntu20.txt`
- **additionally needed**: `correlation_cuda`, install with `cd src/models/correlation_package/ && bash install.sh; cd ../../..` !! probably this does not work yet for an virtual-environment

## Ubuntu 22.04.6 LTS
- device: `Lenovo ThinkPad X250`
- ros: `humble` (ROS v2)
- python3: `Python 3.10.6`
- required python3 packages: `python3 -m pip install -r requirements_ubuntu22.txt`
- **additionally needed**: `correlation_cuda`, install with `cd src/models/correlation_package/ && bash install.sh; cd ../../..` !! probably this does not work yet for an virtual-environment

# ROS node as input pipeline
this program will launch a ROS node to wait for incoming data, for more details look into folder <a hreaf="ros/">ros/</a>.<br>
the subscribed topics will be:<br>
- `/camera/image_raw` which is an uint8 RGB-Image with 3 channels
- `/HFL_PC` a PointCloud2 with float32 values
- `/HFL_Depth` an uint16 depth RGB-Image with 1 channel

the subscriptions can be extended adequately in the files of  <a hreaf="ros/">ros/</a>  

# Configuration
the configuration parameters can be adjusted in the .yaml file <a hreaf="src/config.yaml">src/config.yaml</a>. For example where to decide which ros-version is used

## visualization using RViz
- how to use: https://sebastiangrans.github.io/Visualizing-PCD-with-RViz/
- run `ros2 run rviz2 rviz2` and select PointCloud2 and Image with topics `/HFL_PC` and `/camera/image_raw`\

<br>
<br>
<br>

# Evaluation Process

## proof-of-concept launch
``$ bash launch_example_ros2.sh`` will launch ... \
... a publisher node that periodically replays sample_bag.db3 bag file \
... a subscriber node that subscribes to the topics, synchronizes the data by arrival time and stores it in a queue. a consumer thread will read from this queue and feed the data into a LCCNet.

## Description
- the incoming data will be (time-synchronized) pairs of (rgb_image, lidar_cloud) which are used as input for the trained LCCNet
- the ouptut of the LCCNet will be the estimated 6-DoF vector that is used to construct the homogenous 4x4 matrix $T_{pred}$

## Details: LCCNet Paper
https://arxiv.org/pdf/2012.13901.pdf

## input processing

### Parameters:
- initial extrinsic $T_{init}$, homogenous 4x4 matrix, $T_{init} = 
\left(\begin{array}{cc} 
R_{init} & t_{init}\\
0 & 1\\
\end{array}\right)$
- $R_{init}$: 3x3 rotation matrix of $T_{init}$
- $t_{init}$: 3x1 translation vector of $T_{init}$
- calibration matrix $K$, 3x3 matrix
- lidar Point cloud $P_i = [X_i\ Y_i\ Z_i] \in R^3$
- virtual image plane $p_i = [u_i\ v_i] \in R^2$
- $\hat P_i, \hat p_i$ homogenous coordinates of $P_i$ and $p_i$

### Projection Process

- (1) $Z_i^{init} \cdotp \hat p_i\\ = Z_i^{init} \cdot [u_i\ v_i\ 1]^T \\= K \cdot [R_{init}|t_{init}] \cdot \hat P_i \\= K \cdot [R_{init}|t_{init}] \cdot [X_i\ Y_i\ Z_i\ 1]^T$

- zu (1): $(3 \times 3) \cdotp (3 \times 1) = ... = (3 \times 3) \cdotp (3 \times 4) \cdotp (4 \times 1) \rightarrow (3 \times 1) $

- "By using a Z-buffer method, the depth image $D_{init}$ is computed to determine the visibility of points along the same projection line, where every pixel ($u_i, v_i$) preserves the depth value $Z_i^{init}$ of a 3D point $P_i$ on camera coordinate."
