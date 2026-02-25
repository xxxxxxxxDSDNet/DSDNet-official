import os
import os.path as osp
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import random
import torchvision.transforms as transforms

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def tmm22F_rawrgb_paired_paths_from_folders(folders):
    assert len(folders) == 4, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_rgb_folder, input_raw_folder, gt_rgb_folder, gt_raw_folder = folders

    gt_paths = list(scandir(gt_rgb_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path)
        gt_name, ext = osp.splitext(gt_name)
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_name + '.png')
        gt_raw_path = osp.join(gt_raw_folder, gt_name + '.npz')
        lq_rgb_path = osp.join(input_rgb_folder, gt_name + '.png')
        lq_raw_path = osp.join(input_raw_folder, gt_name + '.npz')

        paths.append(dict(
            [('gt_rgb_path', gt_rgb_path), ('gt_raw_path', gt_raw_path), ('lq_rgb_path', lq_rgb_path),
             ('lq_raw_path', lq_raw_path), ('key', gt_name)]))
    return paths

def tmm22_rawrgb_paired_paths_from_folders(folders):
    assert len(folders) == 4, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_rgb_folder, input_raw_folder, gt_rgb_folder, gt_raw_folder = folders

    gt_paths = list(scandir(gt_rgb_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('_')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_name + '_gt.png')
        gt_raw_path = osp.join(gt_raw_folder, gt_name + '_gt.npz')
        lq_rgb_path = osp.join(input_rgb_folder, gt_name + '_m.png')
        lq_raw_path = osp.join(input_raw_folder, gt_name + '_m.npz')

        paths.append(dict(
            [('gt_rgb_path', gt_rgb_path), ('gt_raw_path', gt_raw_path), ('lq_rgb_path', lq_rgb_path),
             ('lq_raw_path', lq_raw_path), ('key', gt_name)]))
    return paths

def nips_paired_paths_from_folders(folders):
    assert len(folders) == 4, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_rgb_folder, input_raw_folder, gt_rgb_folder, gt_raw_folder = folders

    gt_paths = list(scandir(input_rgb_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('.')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_name + '.png')
        gt_raw_path = osp.join(gt_raw_folder, gt_name + '.npz')
        lq_rgb_path = osp.join(input_rgb_folder, gt_name + '.png')
        lq_raw_path = osp.join(input_raw_folder, gt_name + '.npz')

        paths.append(dict(
            [('gt_rgb_path', gt_rgb_path), ('gt_raw_path', gt_raw_path), ('lq_rgb_path', lq_rgb_path),
             ('lq_raw_path', lq_raw_path), ('key', gt_name)]))
    return paths

def inversegtRAW_paired_paths_from_folders(folders):
    assert len(folders) == 4, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_rgb_folder, input_raw_folder, gt_rgb_folder, gt_raw_folder = folders

    gt_paths = list(scandir(input_rgb_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('.')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_name + '.png')
        gt_raw_path = osp.join(gt_raw_folder, gt_name + '.npy')
        lq_rgb_path = osp.join(input_rgb_folder, gt_name + '.png')
        lq_raw_path = osp.join(input_raw_folder, gt_name + '.npz')

        paths.append(dict(
            [('gt_rgb_path', gt_rgb_path), ('gt_raw_path', gt_raw_path), ('lq_rgb_path', lq_rgb_path),
             ('lq_raw_path', lq_raw_path), ('key', gt_name)]))
    return paths


def tensor2numpy(tensor):
    img_np = tensor.squeeze().numpy()
    img_np[img_np < 0] = 0
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    return img_np.astype(np.float32)


def imwrite_gt(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)

    img = img.clip(0, 1.0)
    uint8_image = np.round(img * 255.0).astype(np.uint8)
    cv2.imwrite(img_path, uint8_image)
    return None
def imwrite_TMM_gtRaw(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)

    img = img.clip(0, 1.0)
    img = np.round(img * 4095.0).astype(np.uint16)
    img = depack_rggb(img)
    np.savez(img_path,patch_data=img)
    return None

def depack_rggb(raws):
    H, W, C = raws.shape
    output = np.zeros((H * 2, W * 2)).astype(np.uint16)

    output[0:2 * H:2, 0:2 * W:2] = raws[:, :, 0]
    output[0:2 * H:2, 1:2 * W:2] = raws[:, :, 1]
    output[1:2 * H:2, 0:2 * W:2] = raws[:, :, 2]
    output[1:2 * H:2, 1:2 * W:2] = raws[:, :, 3]

    return output

def imwrite_raw(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)

    img = img.squeeze(0).permute(1,2,0).numpy()
    img = img.clip(0, 1.0)
    uint16_image = np.round(img * 65535.0).astype(np.uint16)
    np.save(img_path, uint16_image)
    return None

def imwrite_ISPraw(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)

    img = img[0][[0,1,3]].permute(1,2,0).numpy()*255
    img = img.clip(0, 255.0)

    cv2.imwrite(img_path,img[...,::-1])
    return None

def rgb_to_ycbcr(image, bgr2rgb):
    """
    将 RGB 图像转换为 YCbCr 色彩空间。

    :param image: 输入的 RGB 图像 (H, W, 3)，像素值范围为 0-255。
    :return: YCbCr 图像 (H, W, 3)。
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must have dtype of uint8")
    
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    offset = np.array([0, 128, 128])
    
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ycbcr = np.dot(image, transform_matrix.T) + offset
    ycbcr = np.clip(ycbcr, 0, 255).astype(np.float32)
    return ycbcr

def read_img(img_path):
    img = cv2.imread(img_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ycbcr_image = rgb_to_ycbcr(img.astype(np.uint8), bgr2rgb=False)
    return img.astype(np.float32) / 255., ycbcr_image / 255.



def pack_rggb_raw(path, index='patch_data'):
    raw_img = np.load(path)
    raw_data = raw_img[index]
    im = raw_data / 4095.0
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)
    return out

def bayer_raw(path, index='patch_data'):
    raw_img = np.load(path)
    raw_data = raw_img[index]
    im = raw_data / 4095.0
    return im

def ReadRaw(path, index):
    raw_img = np.load(path)
    raw_data = raw_img[index]
    im = raw_data / 4095.0
    return im

def random_crop(gt_rgb, lq_rgb, opt):
    assert lq_rgb.shape == gt_rgb.shape
    C, H, W = lq_rgb.shape
    lq_patch_size = opt['crop_size']
    top = np.random.randint(0, H - lq_patch_size)
    left = np.random.randint(0, W - lq_patch_size)

    lq_rgb = lq_rgb[ :, top:top + lq_patch_size, left:left + lq_patch_size]
    gt_rgb = gt_rgb[ :, top:top + lq_patch_size, left:left + lq_patch_size]

    return gt_rgb, lq_rgb
def crop(crop_size_x, crop_size_y, x, y, img, type_f):
    if type_f=='rgb':
        return img[2*x:2*x + crop_size_x*2, 2*y:2*y + crop_size_y*2, :]
    if type_f=='raw':
        return img[x:x + crop_size_x,y:y + crop_size_y, :]


def crop_loader(crop_size_x, crop_size_y, x, y, path_set, type_f, pad_size=100, pad=False, use_awb=True, cat_type='cat'):
    imgs = []
    for path in path_set:
        if type_f=='rgb':
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            img = img[2*x:2*x + crop_size_x*2, 2*y:2*y + crop_size_y*2]
            img = default_toTensor(img)
        elif type_f=='raw':
            img = load_raw(path, use_awb)
            img = img[:,x:x + crop_size_x,y:y + crop_size_y]
            img = torch.from_numpy(img)
        '''
        if pad:
            img = add_margin(img, pad_size, pad_size, pad_size, pad_size, (123, 117, 104))
        '''
        imgs.append(img)
    if cat_type == 'cat':
        return torch.cat(imgs, dim=0)
    elif cat_type == 'stack':
        return torch.stack(imgs, dim=0)
    else:
        raise ('cat type error')

def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    return composed_transform(img)

def load_raw(path,use_awb=True):
    raw_img = np.load(path)
    try:
        raw_data = raw_img['data'].transpose((2,0,1))
    except:
        raw_data = raw_img.transpose((2,0,1))

    try:
        bl = raw_img['black_level_per_channel'][0]
        wl = raw_img['white_level']
    except:
        bl = 0
        wl = 65535
    norm_factor = wl - bl
    raw_data = (raw_data- bl)/norm_factor
    raw_data = raw_data.astype(np.float32)
    # add camera_whitebalance
    try:
        cwb = raw_img['camera_whitebalance']
        cwb_rggb = np.expand_dims(np.expand_dims(np.array([cwb[0],cwb[1],cwb[1],cwb[2]]), axis=1), axis=2)
        if use_awb:
            raw_data = raw_data*cwb_rggb
    except:
        pass
    raw_data = raw_data.astype(np.float32)

    return raw_data

def default_loader(path_set,type_f,use_awb=True, cat_type='cat'):
    imgs = []
    for path in path_set:
        if type_f=='rgb':
            img = Image.open(path).convert('RGB')
            img = default_toTensor(img)
        elif type_f=='raw':
            img = load_raw(path, use_awb)
            img = torch.from_numpy(img)
        imgs.append(img)

    if cat_type == 'cat':
        return torch.cat(imgs, dim=0)
    elif cat_type == 'stack':
        return torch.stack(imgs, dim=0)
    else:
        raise ('cat type error')

