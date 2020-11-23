import os
import yaml
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch import nn

from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler
from datasets.coco import COCODataSets
from nets.faster_rcnn import FasterRCNN
from torch.utils.data.dataloader import DataLoader
from commons.model_utils import rand_seed, ModelEMA, freeze_bn, reduce_sum
from metrics.map import coco_map
from commons.optims_utils import IterWarmUpCosineDecayMultiStepLRAdjust, split_optimizer

rand_seed(1024)


class DDPMixProcessor(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.data_cfg = self.cfg['data']
        self.model_cfg = self.cfg['model']
        self.optim_cfg = self.cfg['optim']
        self.val_cfg = self.cfg['val']
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.val_cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['gpus']
        self.gpu_num = len(str(self.cfg['gpus']).split(","))
        dist.init_process_group(backend='nccl')
        ### dataloader -------------------------------------------------------------------------------------------------
        self.tdata = COCODataSets(img_root=self.data_cfg['train_img_root'],
                                  annotation_path=self.data_cfg['train_annotation_path'],
                                  min_thresh=self.data_cfg['min_thresh'],
                                  max_thresh=self.data_cfg['max_thresh'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=True,
                                  remove_blank=self.data_cfg['remove_blank']
                                  )
        self.tloader = DataLoader(dataset=self.tdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.tdata, shuffle=True))
        self.vdata = COCODataSets(img_root=self.data_cfg['val_img_root'],
                                  annotation_path=self.data_cfg['val_annotation_path'],
                                  min_thresh=self.data_cfg['min_thresh'],
                                  max_thresh=self.data_cfg['max_thresh'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=False,
                                  remove_blank=False
                                  )
        self.vloader = DataLoader(dataset=self.vdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.vdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.vdata, shuffle=False))
        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata), " | ",
              "empty_data: ", self.tdata.empty_images_len)
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))

        ##### model set ------------------------------------------------------------------------------------------------
        model = FasterRCNN(cfg=self.model_cfg)
        self.scaler = amp.GradScaler(enabled=True)
        self.best_map = 0.
        self.best_map50 = 0.
        optimizer = split_optimizer(model, self.optim_cfg)
        local_rank = dist.get_rank()
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        model.to(self.device)
        if self.optim_cfg['sync_bn']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = nn.parallel.distributed.DistributedDataParallel(model,
                                                                     device_ids=[local_rank],
                                                                     output_device=local_rank)
        self.optimizer = optimizer
        self.ema = ModelEMA(self.model)
        self.lr_adjuster = IterWarmUpCosineDecayMultiStepLRAdjust(init_lr=self.optim_cfg['lr'],
                                                                  warm_up_epoch=self.optim_cfg['warm_up_epoch'],
                                                                  iter_per_epoch=len(self.tloader),
                                                                  epochs=self.optim_cfg['epochs'],
                                                                  alpha=self.optim_cfg['alpha'],
                                                                  gamma=self.optim_cfg['gamma'],
                                                                  bias_idx=2,
                                                                  milestones=self.optim_cfg['milestones']
                                                                  )

    def train(self, epoch):
        self.model.train()
        if self.model_cfg['freeze_bn']:
            self.model.apply(freeze_bn)
        if self.local_rank == 0:
            pbar = tqdm(self.tloader)
        else:
            pbar = self.tloader
        loss_list = [list(), list(), list(), list(), list()]
        ulr = 0
        dlr = 0
        for i, (img_tensor, targets_tensor, _) in enumerate(pbar):
            _, _, h, w = img_tensor.shape
            with torch.no_grad():
                targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
                img_tensor = img_tensor.to(self.device)
                targets_tensor = targets_tensor.to(self.device)
            self.optimizer.zero_grad()
            with amp.autocast(enabled=True):
                _, total_loss = self.model(img_tensor, None, targets_tensor)
                rpn_cls_loss = total_loss['rpn_cls_loss']
                rpn_box_loss = total_loss['rpn_box_loss']
                roi_cls_loss = total_loss['roi_cls_loss']
                roi_box_loss = total_loss['roi_box_loss']
                loss = rpn_cls_loss + rpn_box_loss + roi_cls_loss + roi_box_loss
            self.scaler.scale(loss).backward()
            self.lr_adjuster(self.optimizer, i, epoch)
            ulr = self.optimizer.param_groups[0]['lr']
            dlr = self.optimizer.param_groups[2]['lr']
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            loss_list[0].append(loss.item())
            loss_list[1].append(rpn_cls_loss.item())
            loss_list[2].append(rpn_box_loss.item())
            loss_list[3].append(roi_cls_loss.item())
            loss_list[4].append(roi_box_loss.item())
            if self.local_rank == 0:
                pbar.set_description(
                    "epoch:{:2d}|loss:{:6.4f}|{:6.4f}|{:6.4f}|{:6.4f}|{:6.4f}|ulr:{:8.6f},dlr:{:8.6f}".format(
                        epoch + 1,
                        loss.item(),
                        rpn_cls_loss.item(),
                        rpn_box_loss.item(),
                        roi_cls_loss.item(),
                        roi_box_loss.item(),
                        ulr,
                        dlr
                    ))
        self.ema.update_attr(self.model)
        mean_loss_list = [np.array(item).mean() for item in loss_list]
        print(
            "epoch:{:3d}|local:{:3d}|loss:{:6.4f}|{:6.4f}|:{:6.4f}|{:6.4f}|{:6.4f}|ulr:{:8.6f}|dlr:{:8.6f}"
                .format(epoch + 1,
                        self.local_rank,
                        mean_loss_list[0],
                        mean_loss_list[1],
                        mean_loss_list[2],
                        mean_loss_list[3],
                        mean_loss_list[4],
                        ulr,
                        dlr)
        )

    @torch.no_grad()
    def val(self, epoch):
        self.ema.ema.eval()
        predict_list = list()
        target_list = list()
        # self.model.eval()
        if self.local_rank == 0:
            pbar = tqdm(self.vloader)
        else:
            pbar = self.vloader
        for i, (img_tensor, targets_tensor, _) in enumerate(pbar):
            _, _, h, w = img_tensor.shape
            targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
            img_tensor = img_tensor.to(self.device)
            targets_tensor = targets_tensor.to(self.device)
            predicts, _ = self.ema.ema(img_tensor, None)
            for i, predict in enumerate(predicts):
                predict_list.append(predict)
                targets_sample = targets_tensor[targets_tensor[:, 0] == i][:, 2:]
                target_list.append(targets_sample)
        mp, mr, map50, map = coco_map(predict_list, target_list)
        mp = reduce_sum(torch.tensor(mp, device=self.device)).item() / self.gpu_num
        mr = reduce_sum(torch.tensor(mr, device=self.device)).item() / self.gpu_num
        map50 = reduce_sum(torch.tensor(map50, device=self.device)).item() / self.gpu_num
        map = reduce_sum(torch.tensor(map, device=self.device)).item() / self.gpu_num
        if self.local_rank == 0:
            print("epoch: {:2d}|gpu_num:{:d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
                  .format(epoch + 1,
                          self.gpu_num,
                          mp * 100,
                          mr * 100,
                          map50 * 100,
                          map * 100))
        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:s}_last.pth"
                                        .format(self.cfg['model_name']))
        best_map_weight_path = os.path.join(self.val_cfg['weight_path'],
                                            "{:s}_best_map.pth"
                                            .format(self.cfg['model_name']))
        best_map50_weight_path = os.path.join(self.val_cfg['weight_path'],
                                              "{:s}_best_map50.pth"
                                              .format(self.cfg['model_name']))
        ema_static = self.ema.ema.state_dict()
        cpkt = {
            "ema": ema_static,
            "map": map * 100,
            "epoch": epoch,
            "map50": map50 * 100
        }
        if self.local_rank != 0:
            return
        torch.save(cpkt, last_weight_path)
        if map > self.best_map:
            torch.save(cpkt, best_map_weight_path)
            self.best_map = map
        if map50 > self.best_map50:
            torch.save(cpkt, best_map50_weight_path)
            self.best_map50 = map50

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
        dist.destroy_process_group()
        torch.cuda.empty_cache()
