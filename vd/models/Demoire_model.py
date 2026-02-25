import torch
import os.path as osp
import time
from tqdm import tqdm
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
import lpips
from vd.data.data_util import tensor2numpy, imwrite_gt
import os,cv2
# from deepspeed.profiling.flops_profiler import get_model_profile


@MODEL_REGISTRY.register()
class ImageDemoire(SRModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        if self.opt['val']['metrics'].get('lpips', False) :
            self.net_metric_alex = lpips.LPIPS(net=self.opt['val']['metrics']['lpips']['type']).cuda()

        self.net_g = build_network(opt['network_g'])

        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  
            self.net_g_ema.eval()

        self.loss_dict = OrderedDict()
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.loss_dict['l_pix_rgb'] = 0
        else:
            self.cri_pix = None

        if train_opt.get('Y_opt'):
            self.cri_Y = build_loss(train_opt['Y_opt']).to(self.device)
            self.loss_dict['l_Y'] = 0
        else:
            self.cri_Y = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.loss_dict['l_percep'] = 0

        else:
            self.cri_perceptual = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
            self.loss_dict['l_color'] = 0

        else:
            self.cri_color = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
            self.loss_dict['l_ssim'] = 0
        else:
            self.cri_ssim = None

        if train_opt.get('ms_ssim_opt'):
            self.cri_ms_ssim = build_loss(train_opt['ms_ssim_opt']).to(self.device)
            self.loss_dict['l_ms_ssim'] = 0
        else:
            self.cri_ms_ssim = None

        
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        self.get_train_visuals(current_iter)
        self.optimizer_g.zero_grad()
    
        self.demoire_rgb, self.demoire_Y  = self.net_g(self.input)
            
        l_total = 0
        if self.cri_pix:
            l_pix_rgb = self.cri_pix(self.demoire_rgb, self.gt_rgb)
            l_total += l_pix_rgb
            self.loss_dict['l_pix_rgb'] += l_pix_rgb.item()/self.opt['logger']['print_freq']

        if self.cri_Y:
            l_Y = self.cri_Y(self.demoire_Y, self.gt_Y)
            l_total += l_Y
            self.loss_dict['l_Y'] += l_Y.item()/self.opt['logger']['print_freq']

        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.demoire_rgb, self.gt_rgb, feature_layers=[2])
            l_total += l_percep
            self.loss_dict['l_percep'] += l_percep.item()/self.opt['logger']['print_freq']

        if self.cri_color:
            l_color = self.cri_color(self.demoire_rgb, self.gt_rgb)
            l_total += l_color
            self.loss_dict['l_color'] += l_color.item()/self.opt['logger']['print_freq']

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.demoire_rgb, self.gt_rgb)
            l_total += l_ssim
            self.loss_dict['l_ssim'] += l_ssim.item()/self.opt['logger']['print_freq']

        if self.cri_ms_ssim:
            l_ms_ssim = self.cri_ms_ssim(self.demoire_rgb, self.gt_rgb)
            l_total += l_ms_ssim
            self.loss_dict['l_ms_ssim'] += l_ms_ssim.item()/self.opt['logger']['print_freq']

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(self.loss_dict)
        if current_iter % self.opt['logger']['print_freq'] == 0:
            if self.cri_pix:
                self.loss_dict['l_pix_rgb'] = 0
            if self.cri_perceptual:
                self.loss_dict['l_percep'] = 0
            if self.cri_color:
                self.loss_dict['l_color'] = 0
            if self.cri_ssim:
                self.loss_dict['l_ssim'] = 0
            if self.cri_ms_ssim:
                self.loss_dict['l_ms_ssim'] = 0
            if self.cri_Y:
                self.loss_dict['l_Y'] = 0                
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.demoire_rgb, _ = self.net_g_ema(self.input)

        else:
            self.net_g.eval()
            with torch.no_grad():
                self.demoire_rgb, _ = self.net_g(self.input)
     
            self.net_g.train()
            
    def feed_data(self, data):
        self.lq_raw = data['lq_raw'].to(self.device)
        self.lq_rgb = data['lq_rgb'].to(self.device)
        self.gt_raw = data['gt_raw'].to(self.device)
        self.gt_rgb = data['gt_rgb'].to(self.device)
        self.gt_Y = data['gt_Y'].to(self.device)
        self.input = {'MoireRaw':self.lq_raw,
                      'MoireRGB':self.lq_rgb,
                      'CleanRaw':self.gt_raw,
                      'CleanRGB':self.gt_rgb,}
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        time_inf_total = 0.

        for idx, val_data in enumerate(dataloader):
            img_name = val_data['name'][0]
            self.feed_data(val_data)
            st = time.time()
            self.test()
            st1 = time.time() - st
            time_inf_total += st1

            visuals = self.get_current_visuals()
   
            if self.opt['is_train']:
                sr_img_tensors = self.demoire_rgb.detach()
                metric_data['img'] = sr_img_tensors
                if 'gt' in visuals:
                    gt_img_tensors = self.gt_rgb.detach()
                    metric_data['img2'] = gt_img_tensors
                    del self.gt_rgb
            else:
                sr_img = tensor2numpy(visuals['result'])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    gt_img = tensor2numpy(visuals['gt'])
                    metric_data['img2'] = gt_img

            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    pass
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}.png')
                    imwrite_gt(sr_img, save_img_path)


            if with_metrics:

                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'lpips':
                        continue  
                    if self.opt['is_train']:
                        self.metric_results[name] += calculate_metric(metric_data, opt_).detach().cpu().numpy().sum()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

                if self.opt['val']['metrics'].get('lpips', False):   
                    self.metric_results['lpips'] += self.net_metric_alex.forward(self.demoire_rgb, self.gt_rgb, normalize=True).detach().cpu().numpy().sum()
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        time_avg = time_inf_total / (idx + 1)
        print('The average test time is %.3f' % time_avg)
        if with_metrics:
            for metric in self.metric_results.keys():
                if self.opt['is_train']:
                    self.metric_results[metric] /= (idx + 1) 
                else:
                    self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq_raw'] = self.lq_raw.detach().cpu()
        out_dict['result'] = self.demoire_rgb.detach().cpu()
        out_dict['gt'] = self.gt_rgb.detach().cpu()
        return out_dict
    
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

  
    def get_train_visuals(self, current_iter):
        if current_iter < 5:
            inp_batchs = self.lq_raw.detach().cpu()
            gt_batchs = self.gt_rgb.detach().cpu()
            Y_batchs = self.gt_Y.detach().cpu()
            for batch_id in range(gt_batchs.shape[0]):
                input_path = os.path.join(self.opt['path']['experiments_root'], 'train_vis', f'{current_iter}_{batch_id}_input.jpg')
                gt_path = os.path.join(self.opt['path']['experiments_root'], 'train_vis', f'{current_iter}_{batch_id}_gt.jpg')
                Y_path = os.path.join(self.opt['path']['experiments_root'], 'train_vis', f'{current_iter}_{batch_id}_gt_Y.jpg')

                inp = inp_batchs[batch_id][[0,1,3]].permute(1,2,0).numpy()*255
                gt = gt_batchs[batch_id].permute(1,2,0).numpy()*255
                gt_Y = Y_batchs[batch_id].permute(1,2,0).numpy()*255
                if not os.path.exists(os.path.dirname(gt_path)):
                    os.makedirs(os.path.dirname(gt_path))

                cv2.imwrite(gt_path,gt[...,::-1])
                cv2.imwrite(input_path,inp[...,::-1])
                cv2.imwrite(Y_path, gt_Y[:,:,0])