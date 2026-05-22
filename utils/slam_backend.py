import random
import time
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask
from scipy.spatial.distance import cdist
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.kalmanFilter import extendedKalmanFilter
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
import cv2
peopleid=0
class edm():
    def __init__(self,config):
        self._max_inactive_frames = 10  # Maximum nb of frames before destruction连续十帧没有在新帧里观察到已经建立的类，就会将其删除
        self.next_object_id = 0  # ID for next object
        self.objects_dict = {}
        self.var_init = 0
        self.cam_pos_qat = np.array([[0., 0., 0.], [0., 0., 0., 1.]], dtype=object)
        # self.cam_pos_qat = np.array([[0., 0., 0.], [0., 0., 0., 1.]])
        self.cam_pos = np.array([[0., 0., 0.], [0., 0., 0.]])

        self.dilatation = 1
        # self.score_threshold = 0.1
        # self.max_number_observation = 5
        # self.human_threshold = 0.01
        self.human_threshold = 0.001
        self.object_threshold = 0.15
        self.iou_threshold = 0.9
        self.selected_classes = [0, 56, 67]  #
        self.masked_id = []
        self.frame = []
        self.depth_frame = []
        self.fx,self.fy,self.cx,self.cy=config["Dataset"]["Calibration"]["fx"], config["Dataset"]["Calibration"]["fy"], config["Dataset"]["Calibration"]["cx"], config["Dataset"]["Calibration"]["cy"]
        self.config=config
    def point_cloud_to_image_mask(self,point_cloud, width, height):
        X, Y, Z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

        # 使用相机内参进行投影
        u = (self.fx * X / Z + self.cx).astype(int)
        v = (self.fy * Y / Z + self.cy).astype(int)
        mask = np.zeros((height, width), dtype=int)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        mask[v[valid], u[valid]] = 1
        return mask


    def depth_to_point_cloud(self,depth):
        # 获取深度图的尺寸
        height, width = depth.shape

        # 生成像素坐标网格
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

        # 获取相机内参
        f_x, f_y =self.fx,self.fy
        c_x, c_y = self.cx,self.cy

        # 计算归一化坐标
        x = (i - c_x) / f_x
        y = (j - c_y) / f_y

        # 展平数组
        x = x.flatten()
        y = y.flatten()
        depth = depth.flatten()

        # 计算三维点坐标
        X = x * depth
        Y = y * depth
        Z = depth

        # 将三维点合并为点云
        points = np.vstack((X, Y, Z)).T

        return points

    def get_active(self, val):
        for key in self.objects_dict:
            if self.objects_dict[key]["maskID"] == val:
                return self.objects_dict[key]["activeObject"]
        return "Key not exist"
    def get_have_been_active(self, val):
        for key in self.objects_dict:
            if self.objects_dict[key]["maskID"] == val:
                return self.objects_dict[key]["have_been_activate"]
        return "Key not exist"

    def get_classid(self, val):
        for key in self.objects_dict:
            if self.objects_dict[key]["maskID"] == val:
                return self.objects_dict[key]["classID"]
        return "Key not exist"

    def add_padding(self,img, pad_l, pad_t, pad_r, pad_b):
        height, width = img.shape
        # Adding padding to the left side.
        pad_left = np.zeros((height, pad_l), dtype=np.int)
        img = np.concatenate((pad_left, img), axis=1)

        # Adding padding to the top.
        pad_up = np.zeros((pad_t, pad_l + width))
        img = np.concatenate((pad_up, img), axis=0)

        # Adding padding to the right.
        pad_right = np.zeros((height + pad_t, pad_r))
        img = np.concatenate((img, pad_right), axis=1)

        # Adding padding to the bottom
        pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
        img = np.concatenate((img, pad_bottom), axis=0)

        return img
    def iou_centered_centroid(self, rois_old, rois_new, mask_old, mask_new):

        mask_old = mask_old.mask
        mask_new = mask_new.mask
        img_v = mask_old.shape[0]
        img_h = mask_old.shape[1]

        pad_x_old = int((img_v - (rois_old[3] - rois_old[1])) / 2)
        pad_y_old = int((img_h - (rois_old[2] - rois_old[0])) / 2)
        pad_x_new = int((img_v - (rois_new[3] - rois_new[1])) / 2)
        pad_y_new = int((img_h - (rois_new[2] - rois_new[0])) / 2)
        rois_old = np.array(rois_old.cpu(), dtype=int)
        rois_new = np.array(rois_new.cpu(), dtype=int)
        cropped_mask_old = mask_old[rois_old[1]:rois_old[3], rois_old[0]:rois_old[2]]
        cropped_mask_new = mask_new[rois_new[1]:rois_new[3], rois_new[0]:rois_new[2]]

        centered_mask_old = self.add_padding(cropped_mask_old, pad_y_old, pad_x_old, pad_y_old, pad_x_old)
        centered_mask_new = self.add_padding(cropped_mask_new, pad_y_new, pad_x_new, pad_y_new, pad_x_new)

        centered_mask_old_croped = centered_mask_old[1:455,1:615]
        centered_mask_new_croped = centered_mask_new[1:455, 1:615]
        intersection = np.logical_and(centered_mask_old_croped, centered_mask_new_croped)
        union = np.logical_or(centered_mask_old_croped, centered_mask_new_croped)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def add_object(self, centroid, dimensions, mask_id, class_id, mask_old, rois_old, pose,depth):
        dt = 0.25

        h_mat = np.linalg.inv(pose.numpy())  # 这里应该是Twc
        depth_point=depth*mask_old.mask[:, :]

        point_cloud_w=None
        # points=
        b = h_mat.dot(np.array([[centroid[0], centroid[1], centroid[2], 1]]).T)[0:3, :]

        y = np.array([b[0, 0], b[1, 0], b[2, 0]])

        x = [y[0], y[1], y[2], 0, 0, 0]

        P = np.eye(len(x))

        F = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        H = np.array([[0.001, 0, 0, 0, 0, 0],
                      [0, 0.001, 0, 0, 0, 0],
                      [0, 0, 0.001, 0, 0, 0]])

        if class_id == 1:  # 加速度？
            ax = 0.68
            ay = 0.68
            az = 0.68
        else:
            ax = 1
            ay = 1
            az = 1

        Q = np.array([[((dt ** 4) / 4) * (ax ** 2), 0.0, 0.0, ((dt ** 4) / 4) * (ax ** 3), 0.0, 0.0],
                      [0.0, ((dt ** 4) / 4) * (ay ** 2), 0.0, 0.0, ((dt ** 4) / 4) * (ay ** 3), 0.0],
                      [0.0, 0.0, ((dt ** 4) / 4) * (az ** 2), 0.0, 0.0, ((dt ** 4) / 4) * (az ** 3)],
                      [((dt ** 4) / 4) * (ax ** 3), 0.0, 0.0, (dt ** 2) * (ax ** 2), 0.0, 0.0],
                      [0.0, ((dt ** 4) / 4) * (ay ** 3), 0.0, 0.0, (dt ** 2) * (ax ** 2), 0.0],
                      [0.0, 0.0, ((dt ** 4) / 4) * (az ** 3), 0.0, 0.0, (dt ** 2) * (ax ** 2)]])

        R = np.array([[0.8, 0, 0],
                      [0, 0.8, 0],
                      [0, 0, 1.2]])

        self.objects_dict.update({self.next_object_id: {
            "kalmanFilter": extendedKalmanFilter(x, P, F, H, Q, R),
            "centroid": centroid,
            "dimension": dimensions,
            "classID": class_id,
            "roisOld": rois_old,
            "maskID": mask_id,
            "maskOld": mask_old,
            "worldPose": [0, 0, 0],
            "estimatedVelocity": [0, 0, 0],
            "estimatedPose": [0, 0, 0],
            "inactiveNbFrame": 0,
            "activeObject": 0,
            "have_been_activate":0,
            "have_been_activate_candiate":0,
        "people_points":point_cloud_w}})

        self.next_object_id = self.next_object_id + 1

    def delete_object(self, object_id):
        del self.objects_dict[object_id]

    def get_masking_depth(self, image, mask):
        """Apply the given mask to the image.
        """
        x = np.zeros([image.shape[0], image.shape[1]])
        y = np.zeros(len(mask))

        for i in range(len(mask)):

            x[:, :] = np.where(mask[i].mask != 1,
                               0,
                               image[:, :])

            x[:, :] = np.where(np.isnan(x[:, :]),
                               0,
                               x[:, :])

            if sum(sum((x[:, :] != 0))) == 0:
                y[i] = 0
            else:
                y[i] = (x[:, :].sum() / sum(sum((x[:, :] != 0))))

        return y

    def mask_to_centroid(self, rois, mask_depth):
        current_centroids = {}
        current_dimensions = {}
        for i in range(rois.shape[0]):

            fx = self.fx  # focal length x
            fy = self.fy  # focal length y
            cx = self.cx  # optical center x
            cy = self.cy  # optical center y
            if mask_depth[i] == -1:
                z = 0
            else:
                z = mask_depth[i]
            y = (((rois[i, 3] + rois[i, 1]) / 2) - cy) * z / fy
            x = (((rois[i, 2] + rois[i, 0]) / 2) - cx) * z / fx

            # Translation from point to world coord
            current_centroids.update({i: [x.cpu().numpy(), y.cpu().numpy(), z]})
            current_dimensions.update({i: [rois[i, 3] - rois[i, 1], rois[i, 2] - rois[i, 0]]})
        return current_centroids, current_dimensions

    def calculate_depth_mean(self, depth_image, mask, min_depth=0.05, max_depth=10.0):
        """
        计算掩码内的深度均值，排除深度为0和异常值的像素。

        参数：
        - depth_image: 2D numpy数组，表示深度图（单位假设为米）。
        - mask: 2D numpy数组，表示掩码图，其中掩码区域为0，非掩码区域为1。
        - min_depth: 过滤深度时的最小深度阈值，默认0.1米。
        - max_depth: 过滤深度时的最大深度阈值，默认10.0米。

        返回：
        - mean_depth: 掩码区域内的深度均值。
        """
        # 提取掩码内的深度值
        depth_values = depth_image[mask == 1]

        # 排除深度为0的像素
        valid_depth_values = depth_values[
            (depth_values > 0)]


        # 计算均值
        if len(valid_depth_values) > 0:
            mean_depth = np.mean(valid_depth_values)
            median_depth = np.median(valid_depth_values)
        else:
            mean_depth = np.nan
            median_depth = np.nan

        return mean_depth, median_depth

    def expand_mask_based_on_depth(self, depth_image, mask, mean_depth, max_diff=0.15, expansion_size=40):
        """基于深度均值扩展掩码区域"""
        kernel = np.ones((expansion_size, expansion_size), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel)
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        valid_depth_mask = (np.abs(depth_image - mean_depth) < max_diff)
        mm = np.logical_and(dilated_mask.astype(bool), valid_depth_mask)
        mm = np.logical_or(mm, mask)
        refined_mask[mm] = 1

        return refined_mask

    def apply_depth_image_masking(self, image_in, masks,class_ids,activate_r):
        """Apply the given mask to the image.
        """
        import copy
        from scipy import ndimage
        image_copy = image_in
        image = copy.deepcopy(image_in)
        image_mask = np.ones_like(image)
        people_mask=np.ones_like(image)
        for i in range(len(masks)):
            is_active = activate_r[i]
            # is_active = self.get_have_been_active(i)
            class_id=class_ids[i]
            mask = masks[i].mask[:, :]
            mask = ndimage.binary_dilation(mask, iterations=1)

            mask_fenkai = None
            if is_active == 1:
                if class_id == 0:
                    if self.config["Dataset"]["refine_mask"]:
                        labeled_mask, num_features = ndimage.label(mask)
                        mask_fenkai = [np.zeros_like(mask, dtype=bool) for _ in range(
                            num_features)]  # 遍历标记的连通区域，并生成独立的掩码
                        for i in range(1, num_features + 1):
                            mask_fenkai[i - 1] = (labeled_mask == i)
                        for single_people_mak in mask_fenkai:
                            meandepth, mediandepth = self.calculate_depth_mean(image, single_people_mak)
                            refined_mask = self.expand_mask_based_on_depth(image, single_people_mak, mediandepth,max_diff=self.config["Dataset"]["refine_mask_dis"],
                                                                   expansion_size=self.config["Dataset"]["refine_mask_expansion_size"])
                            image[:, :] = np.where(refined_mask == 1,
                                                   0,
                                                   image[:, :])
                            image_mask[:, :] = np.where(refined_mask == 1,
                                                        0,
                                                        image_mask[:, :])
                    else:
                        refined_mask=mask
                        image[:, :] = np.where(refined_mask == 1,
                                               0,
                                               image[:, :])
                        image_mask[:, :] = np.where(refined_mask == 1,
                                                    0,
                                                    image_mask[:, :])
                else:
                    image[:, :] = np.where(mask == 1,
                                           0,
                                           image[:, :])
                    image_mask[:, :] = np.where(mask == 1,
                                                0,
                                                image_mask[:, :])
            if class_id == 0:
                if self.config["Dataset"]["refine_mask"]:
                    if mask_fenkai is None:
                        labeled_mask, num_features = ndimage.label(mask)
                        mask_fenkai = [np.zeros_like(mask, dtype=bool) for _ in range(
                            num_features)]
                        for i in range(1, num_features + 1):
                            mask_fenkai[i - 1] = (labeled_mask == i)
                    for single_people_maks in mask_fenkai:
                        meandepth, mediandepth = self.calculate_depth_mean(image_copy, single_people_maks)
                        refined_mask = self.expand_mask_based_on_depth(image_copy, single_people_maks, mediandepth,max_diff=self.config["Dataset"]["refine_mask_dis"],
                                                                   expansion_size=self.config["Dataset"]["refine_mask_expansion_size"])
                        image_copy[:, :] = np.where(refined_mask == 1,
                                                    0,
                                                    image_copy[:, :])
                        people_mask[:, :] = np.where(refined_mask == 1,
                                                     0,
                                                     people_mask[:, :])
                else:
                    refined_mask =mask
                    people_mask[:, :] = np.where(refined_mask == 1,
                                                 0,
                                                 people_mask[:, :])
        return image,image_mask,people_mask

    def core(self, outputs, depth_data, pose):

        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)

        masks = [GenericMask(x, depth_data.shape[0], depth_data.shape[1]) for x in masks]
        r = {'class_ids': outputs["instances"].pred_classes, 'scores': outputs["instances"].scores,
             'rois': outputs["instances"].pred_boxes.tensor, "masks": masks}
        activate_r = [0] * len(outputs["instances"].pred_classes)
        objects_to_delete = []
        mask_depth = self.get_masking_depth(depth_data, r['masks'])
        # Main filter update and prediction step
        if r['rois'].shape[0] == 0:
            for i in self.objects_dict:
                self.objects_dict[i]["inactiveNbFrame"] = self.objects_dict[i][
                                                              "inactiveNbFrame"] + 1

                if self.objects_dict[i]["inactiveNbFrame"] > self._max_inactive_frames:
                    objects_to_delete.append(i)

            for i in objects_to_delete:
                self.delete_object(i)
        else:
            current_centroids, current_dimensions = self.mask_to_centroid(r['rois'],
                                                                          mask_depth)

            if not self.objects_dict:
                if not len(current_centroids) == 0:
                    for i in range(len(current_centroids)):
                        self.add_object(current_centroids[i], current_dimensions[i], i, r['class_ids'][i],
                                        r['masks'][i], r['rois'][i], pose,depth_data)

                    for i in self.objects_dict:  # 添加玩之后先传播一次
                        self.objects_dict[i]["kalmanFilter"].prediction()
                        self.objects_dict[i]["kalmanFilter"].update(self.objects_dict[i]["centroid"],
                                                                    pose.numpy())
                        self.objects_dict[i]["estimatedPose"] = self.objects_dict[i]["kalmanFilter"].x[
                                                                0:3]
                        self.objects_dict[i]["estimatedVelocity"] = self.objects_dict[i]["kalmanFilter"].x[3:6]
            else:
                objects_pose = np.zeros((len(self.objects_dict), 3))
                objects_ids = np.zeros((len(self.objects_dict)))
                index = 0
                for i in self.objects_dict:
                    objects_pose[index,] = self.objects_dict[i]["centroid"]
                    objects_ids[index] = i
                    index = index + 1
                centroids_pose = np.zeros((len(current_centroids), 3))
                for i in range(len(current_centroids)):
                    centroids_pose[i,] = current_centroids[i]

                eucledian_dist_pairwise = np.array(cdist(objects_pose, centroids_pose)).flatten()  #
                index_sorted = np.argsort(eucledian_dist_pairwise)
                used_objects = []
                used_centroids = []
                for index in range(len(eucledian_dist_pairwise)):
                    object_id = int(index_sorted[index] / len(centroids_pose))
                    centroid_id = index_sorted[index] % len(centroids_pose)

                    if not np.in1d(object_id, used_objects) and not np.in1d(centroid_id,
                                                                            used_centroids):
                        if self.objects_dict[objects_ids[object_id]]["classID"] == r['class_ids'][
                            centroid_id]:
                            timebefore = time.time()
                            used_objects.append(object_id)
                            used_centroids.append(centroid_id)
                            self.objects_dict[objects_ids[object_id]]["kalmanFilter"].prediction()
                            self.objects_dict[objects_ids[object_id]]["kalmanFilter"].update(
                                current_centroids[centroid_id], pose.numpy())
                            self.objects_dict[objects_ids[object_id]]["estimatedPose"] = \
                            self.objects_dict[objects_ids[object_id]]["kalmanFilter"].x[0:3]
                            self.objects_dict[objects_ids[object_id]]["estimatedVelocity"] = \
                            self.objects_dict[objects_ids[object_id]]["kalmanFilter"].x[3:6]

                            if self.objects_dict[objects_ids[object_id]]["classID"] == peopleid:  # 创建动-静的判断阈值
                                max_threshold = self.human_threshold
                            else:
                                max_threshold = self.object_threshold
                            if abs(self.objects_dict[objects_ids[object_id]]["estimatedVelocity"][
                                       0]) > max_threshold or abs(
                                self.objects_dict[objects_ids[object_id]]["estimatedVelocity"][
                                    1]) > max_threshold or abs(
                                self.objects_dict[objects_ids[object_id]]["estimatedVelocity"][2]) > max_threshold:
                                self.objects_dict[objects_ids[object_id]]["activeObject"] = 1
                                if self.objects_dict[objects_ids[object_id]]["have_been_activate_candiate"] ==1:
                                    self.objects_dict[objects_ids[object_id]]["have_been_activate"] = 1
                                self.objects_dict[objects_ids[object_id]]["have_been_activate_candiate"] = 1
                                # 运动的物体
                                activate_r[centroid_id]=1
                            else:
                                self.objects_dict[objects_ids[object_id]]["activeObject"] = 0
                            if self.objects_dict[objects_ids[object_id]]["classID"] == peopleid and \
                                    self.objects_dict[objects_ids[object_id]]["activeObject"] == 0:

                                iou = self.iou_centered_centroid(self.objects_dict[objects_ids[object_id]]["roisOld"],
                                                                 r['rois'][centroid_id],
                                                                 self.objects_dict[objects_ids[object_id]]["maskOld"],
                                                                 r['masks'][centroid_id])
                                if iou < self.iou_threshold:
                                    self.objects_dict[objects_ids[object_id]]["activeObject"] = 1
                                    if self.objects_dict[objects_ids[object_id]]["have_been_activate_candiate"] == 1:
                                        self.objects_dict[objects_ids[object_id]]["have_been_activate"] = 1
                                    self.objects_dict[objects_ids[object_id]]["have_been_activate_candiate"] = 1
                                else:
                                    x = 1
                            self.objects_dict[objects_ids[object_id]]["centroid"] = centroids_pose[centroid_id]
                            self.objects_dict[objects_ids[object_id]]["dimensions"] = current_dimensions[centroid_id]
                            self.objects_dict[objects_ids[object_id]]["inactiveNbFrame"] = 0
                            self.objects_dict[objects_ids[object_id]]["maskID"] = centroid_id
                            self.objects_dict[objects_ids[object_id]]["maskOld"] = r['masks'][centroid_id]
                            self.objects_dict[objects_ids[object_id]]["roisOld"] = r['rois'][centroid_id]
                if len(centroids_pose) < len(objects_pose):
                    for index in range(len(eucledian_dist_pairwise)):
                        object_id = int(index_sorted[index] / len(objects_pose))
                        if not np.in1d(object_id, used_objects):
                            self.objects_dict[objects_ids[object_id]]["inactiveNbFrame"] += 1

                            self.objects_dict[objects_ids[object_id]]["activeObject"] = 0
                            used_objects.append(object_id)
                            if self.objects_dict[objects_ids[object_id]][
                                "inactiveNbFrame"] >= self._max_inactive_frames:
                                self.delete_object(objects_ids[object_id])
                            else:
                                self.objects_dict[objects_ids[object_id]]["kalmanFilter"].prediction()
                                self.objects_dict[objects_ids[object_id]]["estimatedPose"] = \
                                    self.objects_dict[objects_ids[object_id]]["kalmanFilter"].x_[0:3]
                                self.objects_dict[objects_ids[object_id]]["estimatedVelocity"] = \
                                    self.objects_dict[objects_ids[object_id]]["kalmanFilter"].x_[3:6]

                elif len(centroids_pose) > len(objects_pose):
                    buff_id = self.next_object_id
                    for index in range(len(eucledian_dist_pairwise)):
                        centroid_id = index_sorted[index] % len(centroids_pose)
                        if not np.in1d(centroid_id, used_centroids):
                            self.add_object(current_centroids[centroid_id], current_dimensions[centroid_id],
                                            centroid_id, r['class_ids'][centroid_id], r['masks'][centroid_id],
                                            r['rois'][centroid_id], pose,depth_data)
                            self.objects_dict[buff_id]["kalmanFilter"].prediction()
                            self.objects_dict[buff_id]["kalmanFilter"].update(current_centroids[centroid_id],
                                                                              pose.numpy())
                            self.objects_dict[buff_id]["estimatedPose"] = self.objects_dict[buff_id]["kalmanFilter"].x[
                                                                          0:3]
                            self.objects_dict[buff_id]["estimatedVelocity"] = self.objects_dict[buff_id][
                                                                                  "kalmanFilter"].x[3:6]
                            buff_id = buff_id + 1

        result_dynamic_filter_image,dynamic_filter_mask,people_mask = self.apply_depth_image_masking(depth_data, r['masks'],r["class_ids"],activate_r)

        return result_dynamic_filter_image, dynamic_filter_mask,people_mask

class BackEnd(mp.Process):
    def __init__(self, config,save_dir):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0#
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {} #每一个建图用的关键帧都会存在这里
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.zerogs=False
        self.count_kf=0
        self.save_dir = save_dir
        self.received_chunks =0
        self.combined_data = {}
        # SEGMENT

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None,last_keyframe_viewpoint =None):
        self.newgs_xyz=self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,last_keyframe_viewpoint =last_keyframe_viewpoint
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):

        # 首先，检查当前窗口是否为空，如果为空则直接返回，不进行后续操作。
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        str=time.time()

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if self.config["Dataset"]["boon_type"] == "True":
                            if prune_mode == "odometry":
                                to_prune = self.gaussians.n_obs < 3
                            if prune_mode == "slam":
                                sorted_window = sorted(current_window, reverse=True)
                                mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                                if not self.initialized:
                                    mask = self.gaussians.unique_kfIDs >= 0
                                to_prune = torch.logical_and(
                                    self.gaussians.n_obs <= prune_coviz, mask
                                )


                        if to_prune is not None:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    #temp ban
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ) :
                    Log("Resetting the opacity of non-visible Gaussians ")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                #tem ban for dynamic test
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint) #更新相机位姿
        end=time.time()
        gap=end-str
        # print(gap)
        return gaussian_split

    def reconstruction(self,frames,before_refine=True):
        from argparse import ArgumentParser, Namespace
        import open3d as o3d
        from scipy.ndimage import median_filter
        class GroupParams:
            pass
        class ParamGroup:
            def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
                group = parser.add_argument_group(name)
                for key, value in vars(self).items():
                    shorthand = False
                    if key.startswith("_"):
                        shorthand = True
                        key = key[1:]
                    t = type(value)
                    value = value if not fill_none else None
                    if shorthand:
                        if t == bool:
                            group.add_argument(
                                "--" + key, ("-" + key[0:1]), default=value, action="store_true")
                        else:
                            group.add_argument(
                                "--" + key, ("-" + key[0:1]), default=value, type=t)
                    else:
                        if t == bool:
                            group.add_argument(
                                "--" + key, default=value, action="store_true")
                        else:
                            group.add_argument("--" + key, default=value, type=t)

            def extract(self, args):
                group = GroupParams()
                for arg in vars(args).items():
                    if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                        setattr(group, arg[0], arg[1])
                return group

        def filter_depth_outliers(depth_map, kernel_size=3, threshold=1.0):  # 移除深度图中的离群点（outliers）
            median_filtered = median_filter(depth_map, size=kernel_size)
            abs_diff = np.abs(depth_map - median_filtered)
            outlier_mask = abs_diff > threshold
            depth_map_filtered = np.where(outlier_mask, median_filtered, depth_map)
            return depth_map_filtered
        class OptimizationParams(ParamGroup):
            def __init__(self, parser):
                self.iterations = 30_000
                self.position_lr_init = 0.0001
                self.position_lr_final = 0.0000016
                self.position_lr_delay_mult = 0.01
                self.position_lr_max_steps = 30_000
                self.feature_lr = 0.0025
                self.opacity_lr = 0.05
                self.scaling_lr = 0.005  # before 0.005
                self.rotation_lr = 0.001
                self.percent_dense = 0.01
                self.lambda_dssim = 0.2
                self.densification_interval = 100
                self.opacity_reset_interval = 3000
                self.densify_from_iter = 500
                self.densify_until_iter = 15_000
                self.densify_grad_threshold = 0.0002
                super().__init__(parser, "Optimization Parameters")
        Log("Reconstruction")
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.config["Dataset"]["Calibration"]["width"], self.config["Dataset"]["Calibration"]["height"], self.config["Dataset"]["Calibration"]["fx"], self.config["Dataset"]["Calibration"]["fy"], self.config["Dataset"]["Calibration"]["cx"], self.config["Dataset"]["Calibration"]["cy"])
        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        num = range(0,len(frames),2)
        for viewpoint_cam_idx in num:

            viewpoint_cam = frames[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            estimate_w2c = getWorld2View2(viewpoint_cam.R, viewpoint_cam.T).cpu().numpy()
            image, visibility_filter, radii, render_depth = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
            )
            rendered_color = torch.clamp(image, min=0.0, max=1.0)
            rendered_depth =render_depth.detach().cpu().numpy().squeeze()
            rendered_color =rendered_color.detach()
            # rgb通道转换一下,并转为numpy
            rendered_color = (
                   (rendered_color.permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8)
            rendered_depth = filter_depth_outliers(
                rendered_depth, kernel_size=20, threshold=0.1)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
                o3d.geometry.Image(rendered_depth),
                depth_scale=scale,
                depth_trunc=30,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, estimate_w2c)
        o3d_mesh = volume.extract_triangle_mesh()
        compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                             scale / 512.0, -2.5 * scale / 512.0)
        o3d_mesh = o3d_mesh.translate(compensate_vector)
        visname = self.save_dir+"/"+self.config["Dataset"]["vis_name"]

        if before_refine:
            file_name = visname+"_try_mesh_before_refine.ply"
        else:
            file_name=visname+"_try_mesh_after_refine.ply"
        o3d.io.write_triangle_mesh(str(file_name), o3d_mesh)
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii,render_depth = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            # if iteration % 5000 == 0:
            if iteration % 50 == 0:
                depth_map=viewpoint_cam.depth
                view_depth = cv2.convertScaleAbs(depth_map, alpha=255.0 / depth_map.max())

                color_np = image.detach().cpu().numpy().transpose(1, 2, 0)
                depth_np=render_depth.detach().cpu().numpy()
                depth_residual = np.abs(depth_map - depth_np)
                # depth_residual[depth_map == 0.0] = 0.0
                gt_color_np=gt_image.detach().cpu().numpy().transpose(1, 2, 0)

                color_residual = np.abs(gt_color_np - color_np)

                gt_depth_np=depth_map


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
                visname = self.config["Dataset"]["vis_name"]
                name = "vis_" + visname + "/refine" + f'/{iteration:05d}.jpg'
                plt.savefig(name, bbox_inches='tight', pad_inches=0.2, dpi=300)
                plt.cla()
                plt.clf()

            gt_depth = torch.from_numpy(viewpoint_cam.depth).to(
                dtype=torch.float32, device=image.device
            )[None]
            render_depth = render_depth.squeeze()
            depth_pixel_mask = (gt_depth > 0.01).view(render_depth.shape)
            if viewpoint_cam.dynamic_mask is not None:
                dynamic_maks = viewpoint_cam.dynamic_mask > 0
                rgb_dynamic_mask = torch.from_numpy(dynamic_maks).to(image.device)
                depth_pixel_mask = depth_pixel_mask & rgb_dynamic_mask
                image = image * rgb_dynamic_mask
                gt_image = gt_image * rgb_dynamic_mask
            depth_alpha =self.config["Dataset"]["colorrefine_depth"]
            l1_depth = torch.abs(render_depth * depth_pixel_mask - gt_depth * depth_pixel_mask).mean()
            Ll1 = l1_loss(image, gt_image)
            color_loss = (1.0 - self.opt_params.lambda_dssim) * (Ll1) + self.opt_params.lambda_dssim * (
                        1.0 - ssim(image, gt_image))
            loss = l1_depth * depth_alpha + (1 - depth_alpha) * color_loss
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")


    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        if tag=="keyframe":
            msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes,self.newgs_xyz]
        else:
            msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self): #开启后端进程
        if self.single_thread:
            print("use single thread")
        else:
            print("no single thread")
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue 

                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue

                self.map(self.current_window)
                if self.config["ablation"]["noprune"]:
                    if self.last_sent >= 10:
                        self.map(self.current_window, prune=False, iters=10)
                        self.push_to_frontend()
                else:
                    if self.last_sent >= 10:
                        self.map(self.current_window, prune=True, iters=10)
                        self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.pause = False
                    total_chunks=data[1]
                    if self.received_chunks < total_chunks:
                        chunk = data[2]
                        self.combined_data.update(chunk)
                        self.received_chunks += 1
                        if self.received_chunks == total_chunks:
                            cameras=self.combined_data
                            opacticlist = self.gaussians.get_opacity
                            nun_gaussian = opacticlist.size(0)
                            print("num of gaussian is:",nun_gaussian)
                            self.reconstruction(cameras,before_refine=True)
                            self.color_refinement()
                            if self.config["Dataset"]["boon_type"] == "True":
                                self.reconstruction(cameras,before_refine=False)
                            self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":

                    self.count_kf+=1
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    print("add kf at frame "+str(cur_frame_idx))
                    print("current windows is " + str(current_window))
                    last_keyframe_id = current_window[1]
                    last_keyframe_viewpoint = self.viewpoints[last_keyframe_id]
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map,last_keyframe_viewpoint =last_keyframe_viewpoint)
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 20
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]

                        pose_optimize=self.config["Training"]["lr"]["BA_factor"]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * pose_optimize,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * pose_optimize,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    time_start2 = time.clock()
                    self.map(self.current_window, iters=iter_per_kf)

                    if self.config["ablation"]["noprune"]:
                        self.map(self.current_window, prune=False)
                    else:
                        self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
