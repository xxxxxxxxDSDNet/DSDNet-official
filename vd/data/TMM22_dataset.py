from torch.utils import data as data

from vd.data.data_util import *
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class TMM22ImageYCbCrDataset(data.Dataset):
    def __init__(self, opt):
        super(TMM22ImageYCbCrDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_rgb_folder, self.gt_raw_folder, self.lq_rgb_folder, self.lq_raw_folder = opt['dataroot_gt_rgb'], opt['dataroot_gt_raw'], opt['dataroot_lq_rgb'], opt['dataroot_lq_raw']

        if self.opt['phase'] == 'train':
            self.paths = tmm22_rawrgb_paired_paths_from_folders([self.lq_rgb_folder, self.lq_raw_folder, self.gt_rgb_folder, self.gt_raw_folder])
        else:
            self.paths = tmm22_rawrgb_paired_paths_from_folders([self.lq_rgb_folder, self.lq_raw_folder, self.gt_rgb_folder, self.gt_raw_folder])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.paths[index]['key']

        gt_rgb_path = self.paths[index]['gt_rgb_path']
        img_gt_rgb, gt_Y = read_img(gt_rgb_path)         

        gt_raw_path = self.paths[index]['gt_raw_path']
        img_gt_raw = pack_rggb_raw(gt_raw_path)

        lq_rgb_path = self.paths[index]['lq_rgb_path']
        img_lq_rgb, _ = read_img(lq_rgb_path)

        lq_raw_path = self.paths[index]['lq_raw_path']
        img_lq_raw = pack_rggb_raw(lq_raw_path)

        if self.opt['phase'] == 'train':
            img_gt_rgb, img_gt_raw, img_lq_rgb, img_lq_raw, gt_Y = augment([img_gt_rgb, img_gt_raw, img_lq_rgb, img_lq_raw, gt_Y], self.opt['use_hflip'], self.opt['use_rot'])

        img_gt_rgb, img_gt_raw, img_lq_rgb, img_lq_raw, gt_Y = img2tensor([img_gt_rgb, img_gt_raw, img_lq_rgb, img_lq_raw, gt_Y], bgr2rgb=False, float32=True)     

        return {'lq_rgb': img_lq_rgb, 'lq_raw': img_lq_raw, 'gt_rgb': img_gt_rgb, 'gt_raw': img_gt_raw, 'gt_Y':gt_Y, 'name': key}

    def __len__(self):
        return len(self.paths)
    


