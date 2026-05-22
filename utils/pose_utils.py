import numpy as np
import torch
import math

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def SE3_log(T):
    dtype = T.dtype
    device = T.device

    R = T[:3, :3]
    t = T[:3, 3]

    # 首先恢复旋转部分θ
    theta = SO3_log(R)

    # 然后恢复平移部分ρ
    theta_norm = torch.norm(theta)
    if theta_norm < 1e-8:
        # 小角度情况，使用泰勒展开近似
        V_inv = torch.eye(3, device=device, dtype=dtype) - 0.5 * skew_symmetric(theta)
    else:
        A = math.sin(theta_norm) / theta_norm
        B = (1 - math.cos(theta_norm)) / (theta_norm ** 2)
        V_inv = torch.eye(3, device=device, dtype=dtype) - 0.5 * skew_symmetric(theta) + \
                (1 - A / (2 * B)) / (theta_norm ** 2) * (skew_symmetric(theta) @ skew_symmetric(theta))

    rho = V_inv @ t
    tau = torch.cat([rho, theta])
    return tau


def skew_symmetric(v):

    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=v.dtype, device=v.device)


def SO3_log(R):

    trace = torch.trace(R)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    if theta < 1e-8:
        # 小角度情况
        w = 0.5 * torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    else:
        # 一般情况
        w = (theta / (2 * math.sin(theta))) * torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    return w
def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T
    tau_m=SE3_exp(tau)

    new_w2c =tau_m  @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
