import PIL.Image
import torch
import numpy
import numpy as np
mm=0


####Edge weight in adaptive weighting######
def create_weight_mask_track(config,height, width, top_bottom_border, left_right_border, min_weight, max_weight, track_mask_dict):

    height=config["Dataset"]["Calibration"]["height"]
    width=config["Dataset"]["Calibration"]["width"]
    mask = np.ones((height, width))
    if config["Tracking"]["edg_filter_region_adaptive"]:

        top_bottom_weights = np.linspace(min_weight, max_weight, top_bottom_border)
        top_bottom_minpixel=config["Tracking"]["top_bottom_minpixel"]
        top_bottom_weights2 = np.linspace(min_weight, max_weight, top_bottom_minpixel)
        if track_mask_dict['yedg_direction'] == "top":
            mask[: top_bottom_border, :] = top_bottom_weights[:, None]  # 上
            mask[-top_bottom_minpixel:, :] = top_bottom_weights2[::-1, None]  # 下
        elif track_mask_dict['yedg_direction'] == "bottom":
            mask[:top_bottom_minpixel, :] = top_bottom_weights2[:, None]  # 上
            mask[-top_bottom_border:, :] = top_bottom_weights[::-1, None]  # 下
        else:
            print("error, no tracking y mask is given")

        left_right_weights = np.linspace(min_weight, max_weight, left_right_border)
        left_right_minpixel=config["Tracking"]["left_right_minpixel"]
        left_right_weights2 = np.linspace(min_weight, max_weight, left_right_minpixel)
        if track_mask_dict['xedg_direction']=="left":
            mask[:, :left_right_border] = np.minimum(mask[:, :left_right_border],  left_right_weights)
            mask[:, -left_right_minpixel:] = np.minimum(mask[:, -left_right_minpixel:],  left_right_weights2[::-1])
        elif track_mask_dict['xedg_direction']=="right":
            mask[:, :left_right_minpixel] = np.minimum(mask[:, :left_right_minpixel], left_right_weights2)
            mask[:, -left_right_border:] = np.minimum(mask[:, -left_right_border:], left_right_weights[::-1])
        else:
            print("error, no tracking x mask is given")
    else:

        top_bottom_height = 40
        top_gradient = np.linspace(0.4, 0.8, top_bottom_height)

        left_right_width = 60
        left_gradient = np.linspace(0.4, 0.8, left_right_width)

        mask[:top_bottom_height, :] = top_gradient[:, None]  # 上
        mask[-top_bottom_height:, :] = top_gradient[::-1, None]  # 下
        mask[:, :left_right_width] = np.minimum(mask[:, :left_right_width], left_gradient)
        mask[:, -left_right_width:] = np.minimum(mask[:, -left_right_width:], left_gradient[::-1])
    return mask

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, track_mask_dict,initialization=False,curid=0,handle_dynamic=True):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint,track_mask_dict,curid=curid,handle_dynamic=handle_dynamic)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint,dynamic_mask=None,weight=None,handle_dynamic=True,curid=0):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask#use it or not
    if dynamic_mask is not None:
        rgb_pixel_mask = rgb_pixel_mask & dynamic_mask
    depthmask=depth>0
    if handle_dynamic:
        tmp = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        tmp_nozero =tmp[tmp!=0]
        thread=tmp_nozero.median()
        handle_dynamicmask = (tmp <= config["Tracking"]["handle_dynamic_color"] * thread)
        rgb_pixel_mask=rgb_pixel_mask&handle_dynamicmask

    if weight is not None:
        l1 = opacity * weight * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

        ####Visualized tracking weight map#####
        # if curid>1 and curid<20 :
        #     opacity_view = opacity.cpu().detach().squeeze().numpy()
        #     weight_view = weight.cpu().detach().numpy()
        #     residual_view=handle_dynamicmask.cpu().detach().numpy()
        #     residual_view = np.transpose(residual_view, (1, 2, 0))
        #     residual_view = np.all(residual_view, axis=-1)
        #     residual_view = residual_view.astype(np.float)
        #     # import matplotlib.colors as mcolors
        #     # cmap = mcolors.ListedColormap(['blue', 'red'])
        #     opacity_weight_view = opacity_view
        #     opacity_weight_view2 = opacity_view*residual_view*weight_view
        #     depthmmask=dynamic_mask.cpu().numpy().astype(np.float32)
        #     depthmmask2=np.squeeze(depthmmask)
        #     opacity_weight_view3 = opacity_weight_view2 *weight_view*depthmmask2
        #     import matplotlib.pyplot as plt
        #     # plt.imshow(residual_view, cmap=cmap)
        #     # plt.colorbar()
        #     # plt.show()
        #     # plt.savefig('residual_view.png')
        #     #
        #     # plt.imshow(weight_view,
        #     #            cmap='coolwarm')
        #     # plt.colorbar()
        #     # plt.title('Colored Representation of NumPy Array')
        #     # plt.savefig('viewweight/output_weight.png')
        #     # plt.imshow(residual_view,
        #     #            cmap='coolwarm')
        #     # plt.colorbar()
        #     # plt.title('Colored Representation of NumPy Array')
        #     # plt.savefig('viewweight/residual_view.png')
        #     opacityname = "viewweight/walk-half-sigmoid/final_weight" + str(curid)+".png"
        #     # opacityname = "viewweight/ballon2/final_weight" + str(curid)+".png"
        #     plt.imshow(opacity_weight_view2,
        #                cmap='coolwarm')
        #     plt.colorbar()
        #     plt.title('Colored Representation of NumPy Array')
        #     plt.savefig(opacityname)
        #     plt.cla()
        #     plt.clf()
    else:
        l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, track_mask_dict,initialization=False,handle_dynamic=True,curid=0
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)#####Opacity weight in adaptive weighting######

    ############temp for ablation
    if not config["Tracking"]["opacity"]:
        opacity_mask = (opacity > 0).view(*depth.shape)
        opacity=torch.ones_like(depth).to(image.device)
    #############temp


    if viewpoint.dynamic_mask is not None:
        dynamic_maks = viewpoint.dynamic_mask > 0
        dynamic_maks = torch.from_numpy(dynamic_maks).to(depth_pixel_mask.device).view(*depth.shape)
        depth_pixel_mask = depth_pixel_mask & dynamic_maks
    depth_mask = depth_pixel_mask * opacity_mask
    if handle_dynamic:  #######Residual weight in adaptive weighting########
        tmp = torch.abs(depth * depth_mask - gt_depth * depth_mask)
        tmp_nozero =tmp[tmp!=0]
        thread=tmp_nozero.median()
        # handle_dynamicmask = (tmp <= 10 * thread)
        handle_dynamicmask = (tmp <= config["Tracking"]["handle_dynamic_depth"] * thread)#temp
        depth_mask=depth_mask&handle_dynamicmask

    #######Edge weight in adaptive weighting######
    if config["Tracking"]["edg_filter"] and track_mask_dict is not None:
        height=config["Dataset"]["Calibration"]["height"]
        width=config["Dataset"]["Calibration"]["width"]
        left_right_border = track_mask_dict["x_edg"]
        left_right_border = max(config["Tracking"]["left_right_minpixel"], min(left_right_border, config["Tracking"]["left_right_maxpixel"]))
        top_bottom_border = track_mask_dict["y_edg"]
        top_bottom_border = max(config["Tracking"]["top_bottom_minpixel"], min(top_bottom_border, config["Tracking"]["top_bottom_maxpixel"]))
        min_weight = config["Tracking"]["min_weight"]
        max_weight = config["Tracking"]["max_weight"]
        weight_mask = create_weight_mask_track(config,height, width, top_bottom_border, left_right_border, min_weight, max_weight,track_mask_dict)
        weight_mask=torch.from_numpy(weight_mask).to(depth_pixel_mask.device)
        l1_depth =  (weight_mask * torch.abs(depth * depth_mask - gt_depth * depth_mask)).mean()
    else:
        weight_mask =None
        l1_depth =  torch.abs(depth * depth_mask - gt_depth * depth_mask).mean()
    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint,dynamic_maks,weight_mask,handle_dynamic=handle_dynamic,curid=curid)
    return alpha * l1_rgb + (1 - alpha) * l1_depth


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint,initialization)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    rgb_pixel_boundmaskview=rgb_pixel_mask.squeeze().cpu().numpy()
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    # rgb_pixel_mask =rgb_pixel_mask &depth_pixel_mask
    if viewpoint.dynamic_mask is not None:
        dynamic_maks=viewpoint.dynamic_mask>0
        dynamic_maks =torch.from_numpy(dynamic_maks).to(rgb_pixel_mask.device)
        rgb_pixel_mask =rgb_pixel_mask&dynamic_maks
    rgb_pixel_boundmaskview_dynamic=rgb_pixel_mask.squeeze().cpu().numpy()
    depth_pixel_maskview=depth_pixel_mask.squeeze().cpu().numpy()
    if config["Mapping"]["map_edg_filter"] :
        
        def create_weight_mask(height, width, top_bottom_border, left_right_border, min_weight, max_weight):
            mask = numpy.full((height, width), min_weight)
            top_bottom_weights = numpy.linspace(max_weight, min_weight, top_bottom_border)
            mask[:top_bottom_border, :] =top_bottom_weights[:, numpy.newaxis]
            mask[-top_bottom_border:, :] = top_bottom_weights[::-1][:,numpy.newaxis]
            # 创建左右边界的线性权重（从1到0.5）
            left_right_weights = numpy.linspace(max_weight, min_weight, left_right_border)
            mask[:, :left_right_border] = numpy.maximum(mask[:, :left_right_border], left_right_weights)
            mask[:, -left_right_border:] = numpy.maximum(mask[:, -left_right_border:], left_right_weights[::-1])
            return mask

        # 参数设置
        height=config["Dataset"]["Calibration"]["height"]
        width=config["Dataset"]["Calibration"]["width"]
        top_bottom_border = 40
        left_right_border = 60
        min_weight = 0.5
        max_weight = 1.0
        weight_mask = create_weight_mask(height, width, top_bottom_border, left_right_border, min_weight, max_weight)
        weight_mask = torch.from_numpy(weight_mask).to(depth_pixel_mask.device)

        with torch.no_grad():
            tmp = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
            tmp_nozero = tmp[tmp != 0]
            thread = tmp_nozero.median()
            handle_dynamicmask_depth = (tmp > 4 * thread)

            false_mask = numpy.zeros((height, width), dtype=bool)
            boundary_width = 60
            false_mask[:boundary_width, :] = True
            false_mask[-boundary_width:, :] = True
            false_mask[:, :boundary_width] = True
            false_mask[:, -boundary_width:] = True
            false_mask = numpy.expand_dims(false_mask, axis=0)
            false_mask =torch.from_numpy(false_mask).to(rgb_pixel_mask.device)
            bool_mask_depth=handle_dynamicmask_depth&false_mask
            tmp_color =  torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
            tmp_nozero_color = tmp_color[tmp_color != 0]
            thread_color = tmp_nozero_color.median()
            handle_dynamicmask_color = (tmp_color > 3 * thread_color)
            bool_mask_rgb = handle_dynamicmask_color & false_mask
            weights_depth = torch.where(bool_mask_depth, 3, 1)
            weights_rgb = torch.where(bool_mask_rgb, 3, 1)

        l1_rgb = weight_mask *torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        l1_depth = weight_mask *torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    else:
        l1_rgb =  torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        l1_depth =  torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
        # valid = torch.logical_and(valid, opacity > 0.92)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    # return valid_depth.median()
    mm=valid_depth.cpu().numpy()
    med=numpy.median(mm)
    quarter=numpy.percentile(mm, [20, 50, 75])
    return quarter[0],quarter[1]
