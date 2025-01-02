
# KILIDAR Project

## Details of implementation of an online self-calibration method adapted from the paper “LCCNet: Lidar and Camera Self-Calibration Using Cost Volume Network” for High Flash Lidar data.


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
