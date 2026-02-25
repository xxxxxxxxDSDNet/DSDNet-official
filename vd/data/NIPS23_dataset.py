import torch
from torch.utils import data as data
import random
from vd.data.data_util import *
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from vd.data.data_util import *
import numpy as np

@DATASET_REGISTRY.register()
class NIPS23ImageYCbCrDataset(data.Dataset):
    def __init__(self, opt):
        super(NIPS23ImageYCbCrDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_rgb_folder, self.gt_raw_folder, self.lq_rgb_folder, self.lq_raw_folder = opt['dataroot_gt_rgb'], opt['dataroot_gt_raw'], opt['dataroot_lq_rgb'], opt['dataroot_lq_raw']

        if self.opt['phase'] == 'train':
            self.paths = nips_paired_paths_from_folders([self.lq_rgb_folder, self.lq_raw_folder, self.gt_rgb_folder, self.gt_raw_folder])
        else:
            self.paths = nips_paired_paths_from_folders([self.lq_rgb_folder, self.lq_raw_folder, self.gt_rgb_folder, self.gt_raw_folder])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.paths[index]['key']

        gt_rgb_path = self.paths[index]['gt_rgb_path']

        gt_raw_path = self.paths[index]['gt_raw_path']

        moire_rgb_path = self.paths[index]['lq_rgb_path']

        moire_raw_path = self.paths[index]['lq_raw_path']


        if  self.opt.get('crop_size', False) :
            x = random.randint(0, self.opt['width']//2 - self.opt['crop_size'])
            y = random.randint(0, self.opt['height']//2 - self.opt['crop_size'])

            gt_rgb = crop_loader(self.opt['crop_size'], self.opt['crop_size'], x, y, [gt_rgb_path],'rgb',use_awb=self.opt.get('use_awb', True))
            gt_raw = crop_loader(self.opt['crop_size'], self.opt['crop_size'], x, y, [gt_raw_path],'raw',use_awb=self.opt.get('use_awb', True))
            moire_rgb = crop_loader(self.opt['crop_size'], self.opt['crop_size'], x, y, [moire_rgb_path],'rgb',use_awb=self.opt.get('use_awb', True))
            moire_raw = crop_loader(self.opt['crop_size'], self.opt['crop_size'], x, y, [moire_raw_path],'raw',use_awb=self.opt.get('use_awb', True))
        else:
            gt_rgb = default_loader([gt_rgb_path],'rgb', use_awb=self.opt.get('use_awb', True))
            gt_raw = default_loader([gt_raw_path],'raw', use_awb=self.opt.get('use_awb', True))
            moire_rgb = default_loader([moire_rgb_path],'rgb', use_awb=self.opt.get('use_awb', True))
            moire_raw = default_loader([moire_raw_path],'raw', use_awb=self.opt.get('use_awb', True)) 
        gt_Y = rgb_to_ycbcr(gt_rgb)
        return {'gt_raw': gt_raw, 'lq_raw':moire_raw,'gt_rgb': gt_rgb,'gt_Y':gt_Y, 'lq_rgb':moire_rgb,'name': key}

    def __len__(self):
        return len(self.paths)


def rgb_to_ycbcr(image, bgr2rgb=False):
    image = image.numpy() * 255
    image = image.transpose((1,2,0))
    
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
    return torch.from_numpy(ycbcr/255).permute(2,0,1)