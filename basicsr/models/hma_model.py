import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp

import numpy as np
import cv2
import copy 

@MODEL_REGISTRY.register()
class HMAModel(SRModel):
    def pre_process(self):
        # pad to a multiple of window_size using reflection padding
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        _, _, h, w = self.lq.size()
        pad_h = (math.ceil(h / window_size) * window_size) - h
        pad_w = (math.ceil(w / window_size) * window_size) - w
        self.mod_pad_h, self.mod_pad_w = pad_h, pad_w
        # Use reflection padding
        self.img = F.pad(self.lq, (0, pad_w, 0, pad_h), mode='reflect')

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)

                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]


    def ensemble_inference(self, val_data, model_paths, weights=None):
        if weights is None:
            weights = [1.0] * len(model_paths)
        assert len(weights) == len(model_paths), \
            "Length of weights must match length of model_paths."

        ensemble_output = None
        total_weight = sum(weights)

        # Save original nets
        original_net_g = self.net_g
        original_net_g_ema = getattr(self, 'net_g_ema', None)  # might be None

        for idx, (model_path, w) in enumerate(zip(model_paths, weights)):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'params_ema' in checkpoint:
                # If net_g_ema doesn't exist, create it
                if not hasattr(self, 'net_g_ema') or self.net_g_ema is None:
                    self.net_g_ema = copy.deepcopy(self.net_g)
                self.net_g_ema.load_state_dict(checkpoint['params_ema'], strict=True)
            else:
                self.net_g.load_state_dict(checkpoint['params'], strict=True)

            # Re-feed data after loading weights
            self.feed_data(val_data)

            # Self-ensemble or normal inference
            if self.opt['val'].get("self_ensemble", False):
                print('self ensemble with N models...')
                self.test_selfensemble()
            else:
                self.pre_process()
                if 'tile' in self.opt:
                    self.tile_process()
                else:
                    self.process()
                self.post_process()

            output = self.output.detach()
            if ensemble_output is None:
                ensemble_output = output * w
            else:
                ensemble_output += output * w

        ensemble_output = ensemble_output / total_weight

        self.net_g = original_net_g
        self.net_g_ema = original_net_g_ema

        return ensemble_output


    def test_selfensemble(self):
        """
        Perform x8 self-ensemble, ensuring we still do reflection-padding so that
        the input size is valid for window-based attention.
        """
        self.pre_process() 

        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                # Transpose height <-> width
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            else:
                tfnp = v2np  

            ret = torch.from_numpy(tfnp).to(self.device)
            return ret

        img_list = [self.img]
        for tf in ['v', 'h', 't']:
            img_list.extend([_transform(t, tf) for t in img_list])

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in img_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in img_list]

        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')

        stacked = torch.stack(out_list, dim=0)
        output = torch.mean(stacked, dim=0, keepdim=False)
        self.output = output

        # Remove padding
        self.post_process()


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        real_dataset = dataloader.dataset
        if hasattr(real_dataset, 'dataset'):
            real_dataset = real_dataset.dataset
        dataset_name = real_dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        img_format = self.opt['val'].get('img_format', 'png')
        competition = self.opt['val'].get('competition', False)
        
        ensemble_paths = self.opt['val'].get('ensemble_models', None)
        ensemble_weights = self.opt['val'].get('ensemble_weights', None)

        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
                
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)  

            if ensemble_paths is not None:
                print(f'[INFO] Using model ensemble with {len(ensemble_paths)} models...')
                self.output = self.ensemble_inference(val_data, ensemble_paths, ensemble_weights)
            else:
                # normal single-model path
                if self.opt['val'].get("self_ensemble", False):
                    print('self ensemble with one model')
                    self.test_selfensemble()
                else:
                    self.pre_process()
                    if 'tile' in self.opt:
                        self.tile_process()
                    else:
                        self.process()
                    self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            if tb_logger is not None:
                result = (sr_img / 255.).astype(np.float32)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                tb_logger.add_image(img_name, result, global_step=current_iter, dataformats='HWC')

            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.{img_format}')
                    elif self.opt['val']['competition']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}.{img_format}')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.{img_format}')
                imwrite(sr_img, save_img_path)
                
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
