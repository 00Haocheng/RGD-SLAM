import time
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, GenericMask
from utils.slam_backend import edm

import sys
from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.utils.logger import setup_logger
new_path="OneFormermain/demo"
if new_path not in sys.path:
    sys.path.append(new_path)
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo
import cv2
from detectron2.data import MetadataCatalog
class FrontEnd(mp.Process):
    def __init__(self, config,save_dir):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.cameras_eva = dict()
        self.device = "cuda:0"
        self.pause = False
        self.save_dir=save_dir

        self.ekf_dynamic_mask = edm(config)
        self.notconver_num=0
        def setup_cfg():
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_common_config(cfg)
            add_swin_config(cfg)
            add_dinat_config(cfg)
            add_convnext_config(cfg)
            add_oneformer_config(cfg)
            cfg.merge_from_file("OneFormermain/configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml")
            opts=['MODEL.IS_TRAIN', 'False', 'MODEL.IS_DEMO', 'True', 'MODEL.WEIGHTS',
             'OneFormermain/checkpoint/150_16_swin_l_oneformer_coco_100ep.pth']
            cfg.merge_from_list(opts)
            cfg.freeze()
            return cfg

        mp.set_start_method("spawn", force=True)
        # args2 = get_parser().parse_args()
        cfg2 = setup_cfg()
        self.predictor2 = VisualizationDemo(cfg2)


    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.min_depth =( self.config["Training"]["min_depth"] if "min_depth" in self.config["Training"] else 0.1)


    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels （略无效的 RGB 像素值，并将它们的深度设置为零。）
        return initial_depth[0].numpy()

    # 初始化SLAM-tracker
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        while not self.backend_queue.empty():
            self.backend_queue.get()
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False
    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        if cur_frame_idx!=-10:
        # if cur_frame_idx <3:
            viewpoint.update_RT(prev.R, prev.T)
        else:
            prevpre = self.cameras[cur_frame_idx - self.use_every_n_frames- self.use_every_n_frames]
            T_w2c1 = torch.eye(4, device=viewpoint.device)
            T_w2c1[0:3, 0:3] = prev.R
            T_w2c1[0:3, 3] = prev.T
            T_w2c2 = torch.eye(4, device=viewpoint.device)
            T_w2c2[0:3, 0:3] = prevpre.R
            T_w2c2[0:3, 3] = prevpre.T
            a=torch.linalg.inv(T_w2c2)@T_w2c1
            T_w2c=T_w2c1@a
            R_assum=T_w2c[0:3,0:3]
            T_assum=T_w2c[0:3, 3]
            viewpoint.update_RT(R_assum, T_assum)
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )

        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        # 用 Adam 优化器来优化相机的姿态参数。
        pose_optimizer = torch.optim.Adam(opt_params)
        # 循环执行跟踪迭代次数。
        converged_num=0
        track_mask_dict = None
        height=self.config["Dataset"]["Calibration"]["height"]
        width=self.config["Dataset"]["Calibration"]["width"]
        if self.config["Tracking"]["edg_filter"] and len(self.current_window) >=2:
            pcd_point =self.newgs_xyz
            cur_frame_Tcw = getWorld2View2(viewpoint.R,
                                               viewpoint.T).cpu().numpy()
            ones = np.ones_like(pcd_point[:, 0]).reshape(-1, 1)
            pcd_point_con = np.concatenate(
                [pcd_point, ones], axis=1).reshape(-1, 4, 1)
            points_in_curkf = cur_frame_Tcw @ pcd_point_con
            points_in_curkf = points_in_curkf[:, :3]
            K = np.array([[viewpoint.fx, .0, viewpoint.cx], [.0, viewpoint.fy, viewpoint.cy], [.0, .0, 1.0]]).reshape(3, 3)
            uv = K @ points_in_curkf  # 内参投影
            z_curkf = uv[:, -1:] + 1e-5
            uv = uv[:, :2] / z_curkf  # 转为像素坐标
            uv = uv.astype(np.int32).squeeze()
            min_x = np.min(uv[:, 0])
            max_x = np.max(uv[:, 0])
            x_left=min_x-0
            x_right=width-max_x
            if max(x_right,x_left)==x_left:
                x_edg=x_left
                xedg_direction="left"
            else:
                x_edg=x_right
                xedg_direction = "right"
            min_y = np.min(uv[:, 1])
            max_y = np.max(uv[:, 1])
            y_top=min_y-0
            y_bottom=height-max_y
            if max(y_top,y_bottom)==y_top:
                y_edg=y_top
                yedg_direction="top"
            else:
                y_edg=y_bottom
                yedg_direction = "bottom"
            track_mask_dict = {'xedg_direction': xedg_direction, 'x_edg': x_edg, 'y_edg': y_edg,'yedg_direction': yedg_direction}
            # # print("%d frame tracking edg minx %d,maxx %d,miny %d maxy %d"%(cur_frame_idx,min_x,max_x,min_y,max_y))
            # left_right_border = track_mask_dict["x_edg"]
            # left_right_border = max(20, min(left_right_border, self.config["Tracking"]["left_right_maxpixel"]))
            # top_bottom_border = track_mask_dict["y_edg"]
            # top_bottom_border = max(10, min(top_bottom_border, self.config["Tracking"]["top_bottom_maxpixel"]))
            # print("%d frame tracking edg minx %d,maxx %d,miny %d maxy %d" % (cur_frame_idx, min_x,max_x,min_y,max_y))
            # print("select edg mask x mask is %d, y maks is %d"%(left_right_border,top_bottom_border))

        for tracking_itr in range(self.tracking_itr_num):
            #  调用 render 函数，生成渲染的图像、深度和不透明度信息。
            torch.cuda.empty_cache()  # Add this line to clear CUDA cache and free memory
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad() #梯度清零。

            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint, track_mask_dict, curid=cur_frame_idx,
                handle_dynamic=self.config["Tracking"]["handle_dynamic"]
            )
            loss_tracking.backward()#反向传播，计算梯度。

            with torch.no_grad():
                pose_optimizer.step() #更新参数，尝试使损失函数最小化。
                converged = update_pose(viewpoint) #更新相机的姿态。

            if tracking_itr % 10 == 0: #每隔10次迭代，将当前帧的信息传递给gui。发送到 q_main2vis 队列中，用于可视化。
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            converged_num+=1

            if converged or converged_num==80: #如果收敛了，就退出
                # if cur_frame_idx % 1 == 0:
                #     print("frame id " + str(cur_frame_idx) + " converged_num :" + str(tracking_itr) + "window: " + str(
                #         self.current_window))
                if not converged:
                    print("not converge frame id " + str(cur_frame_idx))
                    self.notconver_num+=1
                else:
                    self.notconver_num=0
                store_vis_in_track=self.config["Tracking"]["store_vis_in_track"]
                if store_vis_in_track and cur_frame_idx%20==0:
                    image, visibility_filter, radii, render_depth = (
                        render_pkg["render"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                    )
                    gt_image = viewpoint.original_image.cuda()
                    depth_map = viewpoint.depth
                    color_np = image.detach().cpu().numpy().transpose(1, 2, 0)
                    depth_np = render_depth.detach().cpu().numpy()
                    depth_residual = np.abs(depth_map - depth_np)
                    gt_color_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)

                    color_residual = np.abs(gt_color_np - color_np)

                    gt_depth_np = depth_map
                    # if cur_frame_idx==220:
                    #     print()
                    fig, axs = plt.subplots(2, 3)
                    fig.tight_layout()
                    max_depth = np.max(gt_depth_np)

                    gt_color_np = np.clip(gt_color_np, 0, 1)
                    color_np = np.clip(color_np, 0, 1)
                    color_residual = np.clip(color_residual, 0, 1)
                    color_np = np.clip(color_np, 0, 1)
                    depth_np = depth_np.squeeze()
                    depth_residual = depth_residual.squeeze()
                    axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                    axs[0, 0].set_title('Input Depth')
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                    axs[0, 1].set_title('Generated Depth')
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
                    axs[0, 2].set_title('Depth Residual')
                    axs[0, 2].set_xticks([])
                    axs[0, 2].set_yticks([])
                    axs[1, 0].imshow(gt_color_np, cmap="plasma")
                    axs[1, 0].set_title('Input RGB')
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(color_np, cmap="plasma")
                    axs[1, 1].set_title('Generated RGB')
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[1, 2].imshow(color_residual, cmap="plasma")
                    axs[1, 2].set_title('RGB Residual')
                    axs[1, 2].set_xticks([])
                    axs[1, 2].set_yticks([])

                    plt.subplots_adjust(wspace=0, hspace=0)
                    visname=self.config["Dataset"]["vis_name"]
                    # name = "viewweight/" + f'{cur_frame_idx:05d}.jpg'
                    name="vis_"+visname+"/"+f'{cur_frame_idx:05d}.jpg'
                    plt.savefig(name, bbox_inches='tight', pad_inches=0.2, dpi=300)
                    plt.cla()
                    plt.clf()

                break


        self.median_depth,self.median_depth2 = get_median_depth(depth, opacity) #计算深度图的中值深度。

        return render_pkg,converged_num #返回渲染包（render_pkg），其中包含了渲染的图像、深度和不透明度。

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)#Tcw
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)

        # rotation_scale_factor = 2.0  # 旋转缩放因子
        # modified_pose_CW = pose_CW.clone()
        # modified_pose_CW[0:3, 0:3] *= rotation_scale_factor
        # dist = torch.norm((modified_pose_CW @ last_kf_WC)[0:3, 3])

        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check


    def add_to_window2(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):

        def get_samples( n, H, W, fx, fy, cx, cy, c2ws,  people_mask,device):

            c2ws = c2ws.unsqueeze(0)
            if people_mask is not None:
                people_mask=torch.from_numpy(people_mask).to(device)
                sampled_indices_noseen = []

                valid_indices_noseen = torch.nonzero(people_mask == 0, as_tuple=False)
                if valid_indices_noseen.shape[0] < n:
                    sampled_indices_i_noseen = valid_indices_noseen[
                        torch.randperm(valid_indices_noseen.size(0))[:valid_indices_noseen.shape[0]]]
                else:
                    sampled_indices_i_noseen = valid_indices_noseen[
                        torch.randperm(valid_indices_noseen.size(0))[:n]]
                sampled_indices_noseen.append(sampled_indices_i_noseen)
                sampled_indices_noseen = torch.stack(sampled_indices_noseen)
                i_noseen = sampled_indices_noseen[:, :, 1].squeeze(-1)
                j_noseen = sampled_indices_noseen[:, :, 0].squeeze(-1)

                def get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device):
                    dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i, device=device)], -1)
                    dirs = dirs.unsqueeze(-2)
                    # Rotate ray directions from camera frame to the world frame
                    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
                    rays_d = torch.sum(dirs * c2ws[:, None, :3, :3], -1)
                    rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)
                    return rays_o, rays_d
                rays_o_noseen, rays_d_noseen = get_rays_from_uv(i_noseen, j_noseen, c2ws, H, W, fx, fy, cx, cy, device)

                return  rays_o_noseen.reshape(-1, 3), rays_d_noseen.reshape(-1, 3)

        def transform_rays_to_frames_batch( rays, T_batch,device):
            """
            将射线从当前帧变换到其他多个帧。
            rays: 射线向量，形状为 (n, 3)
            T_batch: 当前帧到目标帧的变换矩阵，形状为 (batch_size, 4, 4)
            返回变换后的射线向量，形状为 (batch_size, n, 3)
            """
            # rays=rays.to(self.device)
            n = rays.shape[0]

            batch_size = T_batch.shape[0]
            ones = torch.ones((rays.shape[0], 1)).to(device).unsqueeze(-1)
            # 将射线扩展为齐次坐标，并扩展为batch_size维度
            rays_h = torch.ones(n, rays.shape[1], 4).to(device)
            rays_h[:, :, :3] = rays
            # rays_h = torch.cat([rays, ones], dim=1).repeat(batch_size, 1,1)  # (batch_size, n, 4)
            # transformed_rays =rays_h
            rays_h = rays_h.transpose(0, 1)
            rays_expanded = rays_h.expand(batch_size, -1, -1)
            # 进行变换
            # transformed_rays = torch.bmm(T_batch, rays_h.permute(0, 2, 1)).permute(0, 2, 1)  # (batch_size, n, 4)
            transformed_rays = torch.matmul(T_batch, rays_expanded.transpose(1, 2))  # Tc2w*ray_w
            transformed_rays = transformed_rays.transpose(1, 2)
            # 返回变换后的射线方向（去掉齐次坐标）
            return transformed_rays[:, :, :3]

        def compute_mask_ratio(rays_o_noseen,rays_d_noseen,w2cs,K,H,W,keyframes_peoplemasks_nolast,device):
            rays_nopeople = rays_o_noseen[..., None, :] + rays_d_noseen[..., None, :] * 2  # 100,1,3

            transformed_rays = transform_rays_to_frames_batch(rays_nopeople, w2cs,device)  # 采样的射线在其他帧坐标系下的三维值

            # near2 = 1
            # far2 = 4
            # t_vals2 = torch.linspace(0., 1., steps=num_samples-1).to(device)
            # z_vals2 = near2 * (1. - t_vals2) + far2 * (t_vals2)
            # rays_nopeople = rays_o_noseen[..., None, :] + rays_d_noseen[..., None, :] * z_vals2[..., :, None]  # [num_rays, num_samples, 3]
            rays_nopeople2 = rays_nopeople.reshape(1, -1, 3)

            ones2 = torch.ones_like(rays_nopeople2[..., 0], device=device).reshape(1, -1, 1)
            homo_pts2 = torch.cat([rays_nopeople2, ones2], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1,
                                                                                               -1)
            w2cs_exp2 = w2cs.unsqueeze(1).expand(-1, homo_pts2.shape[1], -1, -1)  # [n_frames,n_points,4,4]
            cam_cords_homo2 = w2cs_exp2 @ homo_pts2
            cam_cords2 = cam_cords_homo2[:, :, :3]
            edge = 20

            uv_people = K @ transformed_rays.unsqueeze(3)
            z_people = uv_people[:, :, -1:] + 1e-5
            uv_people = uv_people[:, :, :2] / z_people  ##射线在其他帧坐标系下的像素值
            u_noseen = uv_people[..., 0, 0].long()  # 形状为 [n_f, n_p]
            v_noseen = uv_people[..., 1, 0].long()

            mask_noseen = (uv_people[:, :, 0] < W - edge) * (uv_people[:, :, 0] > edge) * \
                          (uv_people[:, :, 1] < H - edge) * (uv_people[:, :, 1] > edge)  ##[n_frames,n_points,1]
            # mask_noseen = mask_noseen(z_people[:, :, 0] < 0)   # 这里注意一下，前面给z的是深度的负数值
            mask_noseen = mask_noseen & (z_people[:, :, 0] > 0)
            mask_noseen = mask_noseen.squeeze(-1)

            mask_people_noseen = torch.zeros((mask_noseen.shape[0], mask_noseen.shape[1]), dtype=torch.bool,
                                             device=device)  # 初始化一个全为false的人掩码

            valid_indices_noseen = mask_noseen.nonzero(as_tuple=True)
            # a=valid_indices_noseen[0]
            # b=v_noseen[valid_indices_noseen]
            # c=u_noseen[valid_indices_noseen]
            mask_people_noseen[valid_indices_noseen] = keyframes_peoplemasks_nolast[valid_indices_noseen[0], v_noseen[valid_indices_noseen],
                                                                                    u_noseen[valid_indices_noseen]]
            mask_noseen_considerpeople = mask_noseen & mask_people_noseen  # 重要度排序，图片中非人区域的点在其他帧的非人
            percent_inside_noseen_considerpeople = mask_noseen_considerpeople.sum(dim=1) / uv_people.shape[1]
            return percent_inside_noseen_considerpeople

        N_dont_touch = 2#确保最新加入的两关键帧在窗口中不会被移除
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))#Twc
        if len(window) > self.config["Training"]["window_size"]:
            current_mask=curr_frame.dynamic_mask
            H=self.config["Dataset"]["Calibration"]["height"]
            W=self.config["Dataset"]["Calibration"]["width"]

            rays_o_noseen,rays_d_noseen=get_samples(self.config["Training"]["rays_num_dynamicmask"],H,W,curr_frame.fx,curr_frame.fy,curr_frame.cx,curr_frame.cy,kf_0_WC,current_mask,curr_frame.device)
            K = torch.tensor([[curr_frame.fx, .0, curr_frame.cx], [.0, curr_frame.fy, curr_frame.cy],
                              [.0, .0, 1.0]], device=curr_frame.device).reshape(3, 3)

        w2c_list_nolast=[]
        keyframes_peoplemasks_nolast_list=[]
        percent_covisibility_list=[]
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()#通过逻辑与计算当前地图中高斯点在两个关键帧中都可见的点的数目
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )#当前地图中高斯点在两个帧中可见个数，取较小的一个帧
            point_ratio_2 = intersection / denom#两个掩码重叠区域（交集）相对于较小可见区域的比率。
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)
            if len(window) > self.config["Training"]["window_size"]:
                kf_i = self.cameras[kf_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                w2c_list_nolast.append(kf_i_CW)
                peoplemask_i=torch.from_numpy(kf_i.dynamic_mask).to(curr_frame.device).to(torch.bool)
                keyframes_peoplemasks_nolast_list.append(peoplemask_i)
                percent_covisibility_list.append(point_ratio_2)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        if len(window) > self.config["Training"]["window_size"]:
            w2cs_nolast = torch.stack(w2c_list_nolast, dim=0)
            keyframes_peoplemasks_nolast = torch.stack(keyframes_peoplemasks_nolast_list, dim=0)
            percent_inside_noseen_considerpeople = compute_mask_ratio(rays_o_noseen, rays_d_noseen, w2cs_nolast, K, H,
                                                                      W, keyframes_peoplemasks_nolast,
                                                                      curr_frame.device)
            percent_covisibility = torch.stack(percent_covisibility_list)
            percent_important = percent_inside_noseen_considerpeople * 0.5 + percent_covisibility * 0.5
            _, selected_indice = torch.topk(percent_important, 1, largest=False, sorted=True)
            removed_frame = window[N_dont_touch +selected_indice]
            window.remove(removed_frame)


            # # we need to find the keyframe to remove...(origin_MonoGS)
            # inv_dist = []
            # for i in range(N_dont_touch, len(window)):
            #     inv_dists = []
            #     kf_i_idx = window[i]
            #     kf_i = self.cameras[kf_i_idx]
            #     kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
            #     for j in range(N_dont_touch, len(window)):
            #         if i == j:
            #             continue
            #         kf_j_idx = window[j]
            #         kf_j = self.cameras[kf_j_idx]
            #         kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
            #         T_CiCj = kf_i_CW @ kf_j_WC
            #         inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
            #     T_CiC0 = kf_i_CW @ kf_0_WC
            #     k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
            #     inv_dist.append(k * sum(inv_dists))
            #
            # idx = np.argmax(inv_dist)
            # removed_frame = window[N_dont_touch + idx]
            # window.remove(removed_frame)

        return window, removed_frame
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)#Tcw
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))#Twc
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    # 这个方法将传递的数据中的高斯模型、可见性信息和关键帧信息分别赋值给前端的对应属性。然后遍历关键帧信息列表，对于每个关键帧，更新相应的相机参数。
    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0 #初始化当前帧的索引为 0
        # 获取投影矩阵（三维点到像素坐标系上）
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device) #将投影矩阵转移到GPU上
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty(): #如果gui队列为空
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"]) #如果gui暂停了，那么就通知后端暂停
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty(): #如果前端队列为空
                tic.record() #记录当前时间，用于计算处理时间。
                if cur_frame_idx >= len(self.dataset): #如果当前帧的索引大于数据集的长度，也就是遍历完了~
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                #检查是否有初始化请求
                if self.requested_init: 
                    time.sleep(0.01)
                    continue
                
                # 检查是否处于单线程模式且有请求的关键帧。
                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                # 检查是否未初始化且有请求的关键帧。
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                #从数据集中获取当前帧的图像、深度图和位姿等数据(viewpoint)。 
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                # from PIL import Image
                # pp=self.dataset.depth_paths[423]
                # vpp=np.array(Image.open(pp)) / 5000.0
                viewpoint.compute_grad_mask(self.config) #计算梯度掩码


                color_seg_data = viewpoint.original_image
                color_seg_data = color_seg_data.permute(1, 2, 0) * 255
                color_seg_data = color_seg_data.cpu().numpy()  # 480,640,3 numpy
                # outputs = self.predictor(color_seg_data)
                semantictype=self.config["Dataset"]["semantic_type"]
                outputs, visualized_output = self.predictor2.run_on_image(color_seg_data, semantictype)

                if self.reset:
                    R=viewpoint.R_gt
                    T=viewpoint.T_gt
                else:
                    if cur_frame_idx!=-10:
                    # if cur_frame_idx <3:
                        prev = self.cameras[
                            cur_frame_idx - self.use_every_n_frames]  # 从当前帧往回倒退 self.use_every_n_frames(设置为1就是每帧都使用) 帧，获取前一帧的相机信息作为参考。
                        R,T=prev.R, prev.T
                    else:
                        prev = self.cameras[
                            cur_frame_idx - self.use_every_n_frames]  # 从当前帧往回倒退 self.use_every_n_frames(设置为1就是每帧都使用) 帧，获取前一帧的相机信息作为参考。
                        prevpre = self.cameras[cur_frame_idx - self.use_every_n_frames - self.use_every_n_frames]
                        T_w2c1 = torch.eye(4, device=viewpoint.device)
                        T_w2c1[0:3, 0:3] = prev.R
                        T_w2c1[0:3, 3] = prev.T
                        T_w2c2 = torch.eye(4, device=viewpoint.device)
                        T_w2c2[0:3, 0:3] = prevpre.R
                        T_w2c2[0:3, 3] = prevpre.T
                        a = torch.linalg.inv(T_w2c2) @ T_w2c1
                        T_w2c = T_w2c1 @ a
                        R = T_w2c[0:3, 0:3]
                        T = T_w2c[0:3, 3]

                time_start1 = time.time()  # 记录开始时间
                W2C = torch.linalg.inv(getWorld2View2(R, T))  # Twc
                twc_pose = W2C.cpu()
                depth_map=viewpoint.depth
                depth_seg_data = depth_map  # 480,640numpy
                depth_motionmask_degree=depth_map[:,:].copy()
                depth_filter, depth_filter_mask, people_mask = self.ekf_dynamic_mask.core(outputs, depth_seg_data,
                                                                                      twc_pose)  # pose means Twc
                if self.config["Training"]["dynamic_ekf_mask"] and cur_frame_idx>2:
                    people_mask=depth_filter_mask*people_mask
                # time_end1 = time.time()
                # time_sum1 = time_end1 - time_start1
                # print("seg time is "+str(time_sum1))

                # # ######temp#####这是创造seg_mask 的
                from PIL import Image
                depth = ( depth_map.clip(0, 5) * 255).astype(np.uint8)
                depth = np.stack((depth, depth, depth), axis=-1)
                # image1 = Image.fromarray(depth,"RGB")
                # image1.save('depthimage.png')
                binary_mask = (people_mask * 255).astype(np.uint8)
                inverted_mask = np.where(binary_mask == 255, 0, 255).astype(np.uint8)
                image = Image.fromarray(inverted_mask, mode='L')
                colorname=viewpoint.color_name
                storename="seg_mask2/"+colorname
                directory = os.path.dirname(storename)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                image.save(storename)
                # # ######temp#####

                # # ########################   Ablatio : n% segmentation error#########################
                # nosie=0.6
                # people_mask_copy=people_mask
                # p=np.where(people_mask_copy==0)
                # zi=list(zip(p[0],p[1]))
                # num_zero_change=int(len(zi)*nosie)
                # if num_zero_change>0:
                #     ch=np.random.choice(len(zi),size=num_zero_change,replace=False)
                #     for iidx in ch:
                #         i,j=zi[iidx]
                #         people_mask_copy[i,j]=1.0
                #     people_mask=people_mask_copy
                # depth_map=depth_motionmask_degree[:,:].copy()
                # #############################################

                if self.config["Dataset"]["using_sematic_mask"]:
                    depth_map = depth_map * people_mask
                else:
                    people_mask=np.ones_like(depth_map)
                depth_map[depth_map<self.min_depth]=0
                if self.config["Dataset"]["type"]=="scannet":
                    if self.config["Dataset"]["type2"]=="realsense":
                        depth_map[depth_map>2.5]=0

                ##########temp
                # if cur_frame_idx%1==0:
                #     view_depth=cv2.convertScaleAbs(depth_map, alpha=255.0 / depth_map.max())
                #     v = Visualizer(color_seg_data[:, :, ::-1], MetadataCatalog.get(self.cfg_rcnn.DATASETS.TRAIN[0]), scale=1)
                #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                #
                #
                #     # vistemp=out.get_image()
                #     # plt.figure(figsize=(12, 6))  # 调整画布大小
                #     # # 显示第一张图 (view_depth)
                #     # plt.subplot(1, 2, 1)  # 1行2列的第1个子图
                #     # plt.imshow(view_depth)
                #     # plt.title("Depth Map")
                #     # plt.axis('off')
                #     # # 显示第二张图 (vistemp)
                #     # plt.subplot(1, 2, 2)  # 1行2列的第2个子图
                #     # plt.imshow(vistemp)
                #     # plt.title("Visualization")
                #     # plt.axis('off')
                #     # plt.tight_layout()  # 自动调整子图间距
                #     # plt.show()#可视化没有bounding box的分割结果（需要同时可视化bounding box只能去eslam-dynamic里找了或者重新安装cv2）
                #     ##else  cv2安装好后可以使用下面的
                #     cv2.imshow("Demo0", (out.get_image()[:, :, ::-1]))
                #     cv2.imshow("Demo1", view_depth)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                viewpoint.depth = depth_map
                viewpoint.dynamic_mask = people_mask

                # 将当前帧的视角（viewpoint）保存到 self.cameras 中，以便后续使用。
                self.cameras[cur_frame_idx] = viewpoint

                # 如果需要重置系统，执行以下操作：
                if self.reset:#初始化后设置为 False。
                    self.initialize(cur_frame_idx, viewpoint) #使用当前帧初始化系统
                    self.current_window.append(cur_frame_idx) #将当前帧索引添加到窗口中，窗口可能用于跟踪一系列关键帧。
                    cur_frame_idx += 1
                    continue
                
                # 如果 self.initialized 已经被设置为真（即已经初始化），那么它的值将保持不变；
                # 如果 self.initialized 尚未被设置为真，但当前窗口中的帧数等于指定的窗口大小，则将 self.initialized 的值设置为真。
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking

                render_pkg,converge_num = self.tracking(cur_frame_idx, viewpoint) #似乎是获取渲染的结果



                current_window_dict = {} #创建一个空字典，用于存储当前窗口的关键帧。
                # 将当前窗口的关键帧存储到字典中，键为当前窗口的第一个帧，值为除第一个帧之外的其余帧。
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                # 据当前窗口的关键帧索引，获取对应的关键帧摄像机信息。
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                # 将高斯包装对象放入队列 q_main2vis 中，用于可视化。这个包装对象包含克隆的高斯模型、当前帧、关键帧列表和当前窗口的字典。
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                
                # 如果有请求的关键帧。
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx) #清理当前帧
                    cur_frame_idx += 1 #当前帧索引加一。
                    continue #跳过当前循环，继续执行下一次循环。

                last_keyframe_idx = self.current_window[0] #获取当前窗口的第一个关键帧索引。
                # 计算当前帧与上一个关键帧之间的时间间隔是否大于等于关键帧间隔。
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                # 获取当前帧的可见性？？？
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                # 根据一些条件判断是否创建关键帧，这些条件包括当前帧索引、上一个关键帧索引、当前帧的可见性以及其他一些参数。
                # if converge_num<79:
                #     check_time =check_time
                # else:
                #     check_time =False

                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                # 如果当前窗口的帧数小于指定的窗口大小，则将当前帧添加到窗口中。
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero() #计算当前帧可见性和上一个关键帧的可见性的并集中非零元素的数量。
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero() #计算当前帧可见性和上一个关键帧的可见性的交集中非零元素的数量。
                    point_ratio = intersection / union #计算交集与并集的比值，表示当前帧在上一个关键帧的可见性范围内的点所占的比例。
                    # 判断是否需要创建关键帧，条件是当前帧与上一个关键帧之间的时间间隔大于等于关键帧间隔，并且点的比例小于指定的阈值 kf_overlap。
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )

                # 如果是单线程模式
                if self.single_thread:
                    create_kf = check_time and create_kf #如果是单线程模式，并且满足时间间隔条件，那么就需要创建关键帧。这段代码的作用是确保在单线程模式下，即使点的比例也符合要求，依然需要创建关键帧。
                    if self.notconver_num>=3:
                        create_kf =True
                        self.notconver_num=0

                    # if create_kf:
                    #     num_zeros = np.sum(depth_map== 0)
                    #     total_elements = depth_map.size
                    #     per=num_zeros / total_elements
                        # if per>0.72:
                        #     create_kf = False
                        #     print("deal to zero percent, delet keyframe,percent is: "+str(per))


                if create_kf:
                    if self.config["Training"]["new_keyframe_selection"]:
                        self.current_window, removed = self.add_to_window2(
                            cur_frame_idx,
                            curr_visibility,
                            self.occ_aware_visibility,
                            self.current_window,
                        )#调用 add_to_window 方法，将当前帧添加到当前窗口中，并返回更新后的当前窗口和已移除的关键帧（如果有的话）。
                    else:
                        self.current_window, removed = self.add_to_window(
                            cur_frame_idx,
                            curr_visibility,
                            self.occ_aware_visibility,
                            self.current_window,
                        )  # 调用 add_to_window 方法，将当前帧添加到当前窗口中，并返回更新后的当前窗口和已移除的关键帧（如果有的话）。
                    # 如果是单目摄像头且地图尚未初始化且已移除了关键帧，则执行以下操作。
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True #将重置标志设置为True。因为如果地图尚未初始化且已移除了关键帧，那么就需要重置系统，也即需要重新初始化。
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    ) #调用 add_new_keyframe 方法，根据渲染包的深度和不透明度信息添加新的关键帧。
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    ) #请求添加关键帧。

                else: #如果不需要创建关键帧，那么就cleanup
                    self.cleanup(cur_frame_idx)

                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx) #进行ATE评估，并输出当前frame的索引。
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:#如果前端队列不为空
                data = self.frontend_queue.get() #从前端队列中获取数据。

                # 如果数据的第一个元素是 "sync_backend"，则执行以下操作：
                if data[0] == "sync_backend":
                    self.sync_backend(data) #调用 sync_backend 方法，将获取到的数据作为参数传递给该方法。

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.newgs_xyz=data[4]
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
