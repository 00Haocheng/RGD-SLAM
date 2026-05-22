#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        #高斯的位置,
        self._xyz = torch.empty(0, device="cuda") #创建一个空的张量，用于存储高斯球的坐标
        #颜色(应该是球谐波函数相关)
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        #高斯的尺度, 旋转, 不透明度, 2d半径
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self._potential_dynamic= None#1 means normal;0 menas potential dynamic
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None #输入的点云数据，在create_pcd_from_image_and_depth函数中被赋值。

        self.isotropic = False
    
    # 从尺度以及旋转构建协方差
    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None,last_keyframe_viewpoint =None):
        cam = cam_info #相机信息，包含曝光、原始图像和深度信息等。
        # 根据相机曝光参数对原始图像进行曝光校正。
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        # 将校正后的图像进行范围限制，确保其值在 [0, 1] 范围内。
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        # 将校正后的图像转换为 numpy 数组，便于后续处理。
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        # 检查是否提供了深度图像。
        if depthmap is not None:
            # 将 RGB 图像数据转换为 Open3D 图像对象。
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            # 将深度图像数据转换为 Open3D 图像对象。
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else: #如果没有提供深度图像，则执行以下操作：
            depth_raw = cam.depth #获取相机的深度信息。
            if depth_raw is None: #如果深度信息为空，则创建一个空的深度图像。
                depth_raw = np.empty((cam.image_height, cam.image_width))

            # 检查数据集的传感器类型是否为单目相机：
            if self.config["Dataset"]["sensor_type"] == "monocular":
                # 如果是单目相机，则通过随机生成的深度值创建深度图像，并将其缩放。
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale
            # 将 RGB 图像数据转换为 Open3D 图像对象。
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            # 将深度图像数据转换为 Open3D 图像对象。
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))
        
        # 调用函数create_pcd_from_image_and_depth
        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init,cam_info.dynamic_mask,self.config["Dataset"]["exp_size"],last_keyframe_viewpoint =last_keyframe_viewpoint)

    # def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False,dynamicmask=None,expansion_size=50): #ori
    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, dynamicmask=None, expansion_size=40,last_keyframe_viewpoint =None,num_dep=None):
        if dynamicmask is not None:
            dynamicmask=dynamicmask.astype(np.uint8)
            inverted_mask = (dynamicmask == 0).astype(np.uint8)
            kernel = np.ones((expansion_size, expansion_size), np.uint8)
            dilated_mask = cv2.dilate(inverted_mask, kernel)
            extentmask=dynamicmask*dilated_mask#1 means people extent pixel;

        if init: #如果需要进行初始化，则执行以下操作
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"] #获取初始化时的下采样因子。
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"] #获取下采样因子。
        point_size = self.config["Dataset"]["point_size"] #获取点的大小。应该是点之间的间距？

        if self.config["Dataset"]["adaptive_pointsize"]:
            point_size = min(0.05, 0.01 * np.median(depth)) #根据深度图像的中值调整点的大小。
            # depth_np = np.asarray(depth)  # 将 Open3D 图像转换为 NumPy 数组然后进行比较
            # depth_cal_point = depth_np[depth_np > 0]
            # quarter = np.percentile(depth_cal_point, [20, 50])
            # point_size = min(0.05,0.01 * quarter[0])
            # point_size = min(0.05, 0.01 * np.median(depth_cal_point))
            # print(point_size)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        ) #根据RGB图像和深度图像创建 Open3D 的 RGBD 图像对象。

        # 获取相机的世界坐标系到相机坐标系的转换矩阵，并将其转换为NumPy数组。转到世界坐标系下
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()#Tcw
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        full_mask =None
        if last_keyframe_viewpoint is not None:#只处理了深度图中有效点的部分，判断他们是否在上一个关键帧中被看到
            pcd_point=np.asarray(pcd_tmp.points)#当前帧中根据深度图得到点云(世界坐标下)
            last_keyframe_Tcw=getWorld2View2(last_keyframe_viewpoint.R, last_keyframe_viewpoint.T).cpu().numpy()#上一个关键帧的Tcw
            ones = np.ones_like(pcd_point[:, 0]).reshape(-1, 1)
            pcd_point_con = np.concatenate(
                [pcd_point, ones], axis=1).reshape(-1, 4, 1)#点云齐次坐标(世界坐标下)
            points_in_lastkf=last_keyframe_Tcw@pcd_point_con#当前帧点云在上一帧关键帧坐标系下的位置
            points_in_lastkf =points_in_lastkf[:,:3]
            K = np.array([[cam.fx, .0, cam.cx], [.0, cam.fy, cam.cy], [.0, .0, 1.0]]).reshape(3, 3)
            uv = K @ points_in_lastkf  # 内参投影
            z_lastkf = uv[:, -1:] + 1e-5
            uv = uv[:, :2] / z_lastkf  # 转为像素坐标
            # uv = uv.astype(np.float32)
            uv = uv.astype(np.int32).squeeze()
            edge=20
            valid_mask = (uv[:, 0] < 640 - edge) * (uv[:, 0] > edge) * \
                   (uv[:, 1] < 480 - edge) * (uv[:, 1] > edge)#判断落在上一帧关键帧范围内的点的mask(落在上一帧区域内的为真)

            valid_u_coords = uv[valid_mask][:,0]
            valid_v_coords = uv[valid_mask][:,1]
            # 创建一个掩膜，标记有效位置(人物区域)对应的为0的部分
            mask = last_keyframe_viewpoint.dynamic_mask[valid_v_coords ,valid_u_coords ] == 0#本帧点落在上一帧中人物遮挡没有看到的区域的mask也就是人物区域为真，其余为假
            # 上一帧没有看到的区域因为也没有生成高斯点所以为真
            full_mask = np.ones(uv.shape[0], dtype=bool)  # 创建全为真的掩码
            full_mask[valid_mask] = mask  # 上一帧的人物区域应该为真（因为人物区域在上一帧没有生成高斯点）非人物区域已经有高斯点，置为假;在上一帧看到的区域，只有人物区域为真，其他的都是假
            full_mask=full_mask.astype(np.int).reshape(-1,1)
            znew=np.median(z_lastkf[full_mask])
            # point_size = min(0.05, 0.01 * znew)
            # print(point_size)


        if self.config["Training"]["stratified_points"]:
            normals = np.asarray(pcd_tmp.colors)
            depth_np = np.asarray(depth)
            mm=0
            for i in range(extentmask.shape[0]*extentmask.shape[1]):
                # 对于每个点，获取其对应的 mask 标签
                label = extentmask[i // 640, i % 640]  # 从 mask 中获取标签
                depth_i=depth_np[i // 640, i % 640]
                if depth_i>0:
                    if label >0:
                        normals[mm] = [0, 0, 0]  # 对应标签 0 的法线值
                    else:
                        normals[mm] = [1, 1, 1]

                    mm=mm+1
            if full_mask is not None:
                normals=normals*full_mask#
            pcd_tmp.normals=o3d.utility.Vector3dVector(normals)#normal为0代表人物附近潜在动态点或者上一帧已经生成的高斯点的区域
        if last_keyframe_viewpoint is not None and self.config["Training"]["stratified_points"]:
            normal_0_mask = (normals[:, 0] == 0)
            normal_1_mask = (normals[:, 0] == 1)
            points_normal_0 = np.asarray(pcd_tmp.points)[normal_0_mask]
            colors_normal_0=np.asarray(pcd_tmp.colors)[normal_0_mask]
            points_normal_1 = np.asarray(pcd_tmp.points)[normal_1_mask]
            colors_normal_1 = np.asarray(pcd_tmp.colors)[normal_1_mask]
            # 对 normal == 0 的点进行下采样
            num_points_0 = points_normal_0.shape[0]
            pcd_mask0 =np.random.rand(num_points_0) < (1.0 / downsample_factor)
            downsampled_points_0 = points_normal_0[pcd_mask0]
            downsampled_colors_0 = colors_normal_0[pcd_mask0]
            # 对 normal == 1 的点进行下采样
            num_points_1 = points_normal_1.shape[0]
            stratified_points_multi=self.config["Training"]["stratified_points_multi"]
            pcd_mask1 = np.random.rand(num_points_1) < (stratified_points_multi / downsample_factor)
            # num_zeros = np.sum(depth_np == 0)
            # num_dep = num_zeros / total_elements
            # if num_dep<0.5:
            #     stratified_points_multi=8
            # elif num_dep<0.65:
            #     stratified_points_multi = 64
            # else:
            #     stratified_points_multi = 128
            # pcd_mask1 = np.random.rand(num_points_1) < (stratified_points_multi / downsample_factor)
            downsampled_points_1 = points_normal_1[pcd_mask1]
            downsampled_colors_1 = colors_normal_1[pcd_mask1]

            new_xyz = np.vstack((downsampled_points_0, downsampled_points_1))
            new_rgb = np.vstack((downsampled_colors_0, downsampled_colors_1))
            # 如果需要，也可以更新 normal 信息
            downsampled_normals = np.concatenate([np.zeros((downsampled_points_0.shape[0], 3)),
                                                  np.ones((downsampled_points_1.shape[0], 3))])
        else:
            pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
            # 获取点云的坐标和颜色信息。
            new_xyz = np.asarray(pcd_tmp.points)
            new_rgb = np.asarray(pcd_tmp.colors)
            downsampled_normals =pcd_tmp.normals
        # 创建一个基本的点云对象。
        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd
        # 将点云的点坐标转换为 PyTorch 张量，并移到GPU上。
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        # 将点云的颜色信息转换为 PyTorch 张量，然后再转成sh系数，并移到GPU上。
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )#初始化点云的特征表示。这个特征是用来干嘛的,是跟谐波函数相关的吗
        features[:, :3, 0] = fused_color#点云颜色
        features[:, 3:, 1:] = 0.0
        extendmask_pcd_flaten=None
        if self.config["Training"]["stratified_points"]:
            point_size_view = np.full(new_xyz.shape[0], point_size)
            extendmask_pcd = np.asarray(downsampled_normals)
            # 使用 np.all 和轴来检查每行是否全是 1 或全是 0
            extendmask_pcd_flaten = np.where(np.all(extendmask_pcd == 1, axis=1), 1, 0)#0代表人物附近潜在动态点或者上一帧已经生成的高斯点的区域
            if self.config["Training"]["make_stratified_pointssize_0"]:
                point_size = point_size_view * extendmask_pcd_flaten #make 人物附近潜在动态点或者上一帧已经生成的高斯点的区域 pointsize=0
                point_size = torch.from_numpy(point_size).float().cuda()
            else:
                point_size = torch.from_numpy(point_size_view).float().cuda()
            extendmask_pcd_flaten=torch.from_numpy(extendmask_pcd_flaten).float().cuda().unsqueeze(1)
        # if self.config["Training"]["new_scale"] :
        #     camera_position = torch.from_numpy(np.linalg.inv(W2C)[:3, 3]).float().cuda()
        #     # 计算每个点到相机位置的距离
        #     distances = torch.norm(fused_point_cloud - camera_position, dim=1)
        #     distancesv = distances.cpu().numpy()
        #
        #     def custom_y(x, threshold1=3, threshold2=8.5, y_min=1e-9, y1=0.035, y_max=0.04):
        #         """
        #         自定义分段函数，根据x值返回y值。
        #
        #         参数：
        #         - x: 输入的x值（可以是标量或数组）
        #         - threshold1: 阈值1，x小于此值时y非常缓慢增加
        #         - threshold2: 阈值2，x接近此值时y平滑增至y1
        #         - y_min: 当x小于threshold1时的y值
        #         - y1: y在threshold2处接近的值
        #         - y_max: 最大 y 值，x 无穷大时趋近于此值
        #
        #         返回：
        #         - y: 根据x计算得到的y值
        #         """
        #         x = np.asarray(x)  # 将 x 转为 numpy 数组以支持向量化计算
        #
        #         # 初始化 y 值
        #         y = np.zeros_like(x)
        #
        #         # 第一部分: x < threshold1 时 y 非常缓慢从 0 增至 y_min
        #         mask1 = x < threshold1
        #         y[mask1] = y_min * (np.exp((x[mask1] / threshold1) ** 2) - 1) / (np.e - 1)
        #
        #         # 第二部分: threshold1 <= x <= threshold2 时 y 平滑快速增大到 y1
        #         mask2 = (x >= threshold1) & (x <= threshold2)
        #         y[mask2] = y_min + (y1 - y_min) * (
        #                 np.tanh((x[mask2] - threshold1) / (threshold2 - threshold1) * 3 - 1.5) + 1) / 2
        #
        #         # 第三部分: x > threshold2 时 y 缓慢增大并趋近 y_max
        #         mask3 = x > threshold2
        #         y[mask3] = y1 + (y_max - y1) * (1 - np.exp(-0.2 * (x[mask3] - threshold2)))
        #
        #         return y
        #
        #
        #     point_size = custom_y(distancesv)
        #
        #     point_size = torch.from_numpy(point_size).float().cuda()
        # 根据点与相邻点之间的距离计算尺度。
        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )#means得到的就是每个点到最近的三个点的平均距离的平方
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic: #如果不是各向同性，则将尺度复制到每个颜色通道。
            scales = scales.repeat(1, 3)

        # 初始化旋转矩阵。
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        # 初始化不透明度。
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities,extendmask_pcd_flaten,new_xyz
#这里的extendmask_pcd_flaten意味着，为1的部分是新出现的重要点，为0的部分是之前有点的区域二次生成的或者动态人物周围的店，他们需要更高的不透明度阈值
    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(self, fused_point_cloud, features, scales, rots, opacities, kf_id,potential_dynamic):
        # 将点云数据转换为可训练的参数，并标记为需要梯度计算。
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 将点云的特征表示拆分为直流分量和余弦分量，并转换为可训练的参数，并标记为需要梯度计算。
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )#这是特征中的颜色
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )#这不知道是什么玩意
        # 将尺度、旋转和不透明度转换为可训练的参数，并标记为需要梯度计算。
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        # 创建一个张量，其长度与点云中点的数量相同，每个点对应于关键帧ID。
        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        # 创建一个张量，其长度与点云中点的数量相同，用于记录每个点的观测次数。
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        # 调用 densification_postfix 方法，将新的3D高斯点数据添加到地图中。
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            potential_dynamic=potential_dynamic,
        )

    # 从点云数据序列中扩展高斯模型
    def extend_from_pcd_seq(
        self, 
        cam_info, #相机信息，用于创建点云
        kf_id=-1,  #关键帧的标识符，默认为-1。
        init=False, #一个布尔值，指示是否进行初始化，默认为False。
        scale=2.0,  #缩放因子，默认为2.0。
        depthmap=None, #度图像，可选参数，默认为None。
            zerogs=False,
        last_keyframe_viewpoint =None
    ):
        # self.zerogs=zerogs
        #调用方法create_pcd_from_image
        fused_point_cloud, features, scales, rots, opacities,potential_dynamic,newgs_xyz = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap,last_keyframe_viewpoint =last_keyframe_viewpoint)
        )#返回了融合的点云、特征、缩放、旋转和不透明度。
        #这里除了点云, features一个通道代表颜色, 另一个通道不知道是啥赋值为0, 剩余都是简单初始化一下,
        # 只把对准了shape, 具体值都是0或者随便一个数

        # 调用了另一个方法 extend_from_pcd，以融合的点云、特征、缩放、旋转、不透明度和关键帧标识符作为参数。
        # 向地图中添加新初始化的高斯点
        if not init and self.config["ablation"]["noGSadd"] :
            pass
        else:
            self.extend_from_pcd(
                fused_point_cloud, features, scales, rots, opacities, kf_id,potential_dynamic
            )#
        return newgs_xyz


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)#新的不透明度置为0.4

        for filter in visibility_filters:#替换不可见的高斯点的不透明度
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self._potential_dynamic is not None:
            self._potential_dynamic=self._potential_dynamic[valid_points_mask]
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
            potential_dynamic=None
    ):
        # 创建字典 d，将输入的张量按键值对的形式存储在字典中。
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        # 调用 cat_tensors_to_optimizer 方法将字典中的张量连接到一个优化器可优化的张量中。
        optimizable_tensors = self.cat_tensors_to_optimizer(d)


        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.config["Training"]["stratified-opa"]:
            if self._potential_dynamic is not None:
                self._potential_dynamic = torch.cat((self._potential_dynamic, potential_dynamic), dim=0)

            else:
                self._potential_dynamic = potential_dynamic



        # 初始化一些辅助张量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)
        if self.config["Training"]["stratified-opa"]:
            new_potentialdynamic = self._potential_dynamic[selected_pts_mask].repeat(N, 1)
        else:
            new_potentialdynamic =None
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            potential_dynamic=new_potentialdynamic,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        if self.config["Training"]["stratified-opa"]:
            new_potential = self._potential_dynamic[selected_pts_mask]
        else:
            new_potential =None
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            potential_dynamic=new_potential,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        opacity_list=self.get_opacity
        if self.config["Training"]["stratified-opa"]:
            potential_dynamic=self._potential_dynamic#normal为0代表人物附近潜在动态点或者上一帧已经生成的高斯点的区域
            potential_dynamic_clamp=torch.clamp(potential_dynamic, min=0.9, max=1)
            opacity_list=opacity_list*potential_dynamic_clamp
        prune_mask = (opacity_list < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        if not self.config["ablation"]["noprune"]:
            self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
