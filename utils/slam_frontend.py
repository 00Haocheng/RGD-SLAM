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
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
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
        viewpoint.update_RT(prev.R, prev.T)

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

        pose_optimizer = torch.optim.Adam(opt_params)
        converged_num=0
        track_mask_dict = None
        height=self.config["Dataset"]["Calibration"]["height"]
        width=self.config["Dataset"]["Calibration"]["width"]

        #### Determine where the boundary weights in the adaptive weights start to decrease #####
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
            uv = K @ points_in_curkf
            z_curkf = uv[:, -1:] + 1e-5
            uv = uv[:, :2] / z_curkf
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
            torch.cuda.empty_cache()  # Add this line to clear CUDA cache and free memory
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()

            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint, track_mask_dict, curid=cur_frame_idx,
                handle_dynamic=self.config["Tracking"]["handle_dynamic"]
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0: #for gui
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

            if converged or converged_num==80:
                if not converged:
                    print("not converge frame id " + str(cur_frame_idx))
                    self.notconver_num+=1
                else:
                    self.notconver_num=0

                #####Save the rendered frames in the tracking process#####
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
                    name="vis_"+visname+"/"+f'{cur_frame_idx:05d}.jpg'
                    plt.savefig(name, bbox_inches='tight', pad_inches=0.2, dpi=300)
                    plt.cla()
                    plt.clf()

                break


        self.median_depth,self.median_depth2 = get_median_depth(depth, opacity)

        return render_pkg,converged_num

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


    def add_to_window2(self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window):# Visibility-aware Keyframing
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
                    rays_d = torch.sum(dirs * c2ws[:, None, :3, :3], -1)
                    rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)
                    return rays_o, rays_d
                rays_o_noseen, rays_d_noseen = get_rays_from_uv(i_noseen, j_noseen, c2ws, H, W, fx, fy, cx, cy, device)

                return  rays_o_noseen.reshape(-1, 3), rays_d_noseen.reshape(-1, 3)

        def transform_rays_to_frames_batch( rays, T_batch,device):
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
            # transformed_rays = torch.bmm(T_batch, rays_h.permute(0, 2, 1)).permute(0, 2, 1)  # (batch_size, n, 4)
            transformed_rays = torch.matmul(T_batch, rays_expanded.transpose(1, 2))  # Tc2w*ray_w
            transformed_rays = transformed_rays.transpose(1, 2)
            return transformed_rays[:, :, :3]

        def compute_mask_ratio(rays_o_noseen,rays_d_noseen,w2cs,K,H,W,keyframes_peoplemasks_nolast,device):
            rays_nopeople = rays_o_noseen[..., None, :] + rays_d_noseen[..., None, :] * 2  # 100,1,3
            transformed_rays = transform_rays_to_frames_batch(rays_nopeople, w2cs,device)  # 采样的射线在其他帧坐标系下的三维值
            rays_nopeople2 = rays_nopeople.reshape(1, -1, 3)
            ones2 = torch.ones_like(rays_nopeople2[..., 0], device=device).reshape(1, -1, 1)
            homo_pts2 = torch.cat([rays_nopeople2, ones2], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1,-1)
            w2cs_exp2 = w2cs.unsqueeze(1).expand(-1, homo_pts2.shape[1], -1, -1)  # [n_frames,n_points,4,4]
            cam_cords_homo2 = w2cs_exp2 @ homo_pts2
            cam_cords2 = cam_cords_homo2[:, :, :3]
            edge = 20

            uv_people = K @ transformed_rays.unsqueeze(3)
            z_people = uv_people[:, :, -1:] + 1e-5
            uv_people = uv_people[:, :, :2] / z_people
            u_noseen = uv_people[..., 0, 0].long()  # 形状为 [n_f, n_p]
            v_noseen = uv_people[..., 1, 0].long()

            mask_noseen = (uv_people[:, :, 0] < W - edge) * (uv_people[:, :, 0] > edge) * \
                          (uv_people[:, :, 1] < H - edge) * (uv_people[:, :, 1] > edge)  ##[n_frames,n_points,1]

            mask_noseen = mask_noseen & (z_people[:, :, 0] > 0)
            mask_noseen = mask_noseen.squeeze(-1)

            mask_people_noseen = torch.zeros((mask_noseen.shape[0], mask_noseen.shape[1]), dtype=torch.bool,
                                             device=device)

            valid_indices_noseen = mask_noseen.nonzero(as_tuple=True)
            mask_people_noseen[valid_indices_noseen] = keyframes_peoplemasks_nolast[valid_indices_noseen[0], v_noseen[valid_indices_noseen],
                                                                                    u_noseen[valid_indices_noseen]]
            mask_noseen_considerpeople = mask_noseen & mask_people_noseen  # 重要度排序，图片中非人区域的点在其他帧的非人
            percent_inside_noseen_considerpeople = mask_noseen_considerpeople.sum(dim=1) / uv_people.shape[1]
            return percent_inside_noseen_considerpeople

        N_dont_touch = 2
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

            ####Sampling in the obscured area#####
            rays_o_noseen,rays_d_noseen=get_samples(self.config["Training"]["rays_num_dynamicmask"],H,W,curr_frame.fx,curr_frame.fy,curr_frame.cx,curr_frame.cy,kf_0_WC,current_mask,curr_frame.device)
            K = torch.tensor([[curr_frame.fx, .0, curr_frame.cx], [.0, curr_frame.fy, curr_frame.cy],
                              [.0, .0, 1.0]], device=curr_frame.device).reshape(3, 3)

        ######covisibility in static region#######
        w2c_list_nolast=[]
        keyframes_peoplemasks_nolast_list=[]
        percent_covisibility_list=[]
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

            ######complementarity in dynamic region#######
            percent_inside_noseen_considerpeople = compute_mask_ratio(rays_o_noseen, rays_d_noseen, w2cs_nolast, K, H,
                                                                      W, keyframes_peoplemasks_nolast,
                                                                      curr_frame.device)

            percent_covisibility = torch.stack(percent_covisibility_list)
            percent_important = percent_inside_noseen_considerpeople * 0.5 + percent_covisibility * 0.5
            _, selected_indice = torch.topk(percent_important, 1, largest=False, sorted=True)
            removed_frame = window[N_dont_touch +selected_indice]
            window.remove(removed_frame)
        return window, removed_frame

    def add_to_window(self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window):#original strategy
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
        cur_frame_idx = 0
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
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
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

                if self.requested_init: 
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )

                viewpoint.compute_grad_mask(self.config)


                color_seg_data = viewpoint.original_image
                color_seg_data = color_seg_data.permute(1, 2, 0) * 255
                color_seg_data = color_seg_data.cpu().numpy()  # 480,640,3 numpy
                semantictype=self.config["Dataset"]["semantic_type"]
                outputs, visualized_output = self.predictor2.run_on_image(color_seg_data, semantictype)

                if self.reset:
                    R=viewpoint.R_gt
                    T=viewpoint.T_gt
                else:
                    if cur_frame_idx!=-10:
                    # if cur_frame_idx <3:
                        prev = self.cameras[
                            cur_frame_idx - self.use_every_n_frames]
                        R,T=prev.R, prev.T
                    else:
                        prev = self.cameras[
                            cur_frame_idx - self.use_every_n_frames]
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


                W2C = torch.linalg.inv(getWorld2View2(R, T))  # Twc
                twc_pose = W2C.cpu()
                depth_map=viewpoint.depth
                depth_seg_data = depth_map  # 480,640numpy
                depth_motionmask_degree=depth_map[:,:].copy()
                depth_filter, depth_filter_mask, people_mask = self.ekf_dynamic_mask.core(outputs, depth_seg_data,
                                                                                      twc_pose)  # pose means Twc
                if self.config["Training"]["dynamic_ekf_mask"] and cur_frame_idx>2:
                    people_mask=depth_filter_mask*people_mask

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
                viewpoint.depth = depth_map
                viewpoint.dynamic_mask = people_mask

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking

                render_pkg,converge_num = self.tracking(cur_frame_idx, viewpoint)
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                    if self.notconver_num>=3:
                        create_kf =True
                        self.notconver_num=0

                if create_kf:
                    ############  Visibility-aware Keyframing    #########
                    if self.config["Training"]["new_keyframe_selection"]:
                        self.current_window, removed = self.add_to_window2(
                            cur_frame_idx,
                            curr_visibility,
                            self.occ_aware_visibility,
                            self.current_window,
                        )
                    else:
                        self.current_window, removed = self.add_to_window(
                            cur_frame_idx,
                            curr_visibility,
                            self.occ_aware_visibility,
                            self.current_window,
                        )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )

                else:
                    self.cleanup(cur_frame_idx)

                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx) #Evaluate ATE
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
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

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
