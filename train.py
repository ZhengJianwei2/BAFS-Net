import os
import pprint
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import warnings
from tqdm import tqdm
from dataloader import test_image_preprocess, train_image_preprocess, get_loader
from util import Optimizer, Scheduler
from metrics import AvgMeter
from model import Net
from config import getConfig
from test import Test

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2' 

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=20):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr*decay))

def iou(pred, mask):
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = args.img_size
        self.train_img_dir = os.path.join(args.data_path, 'image')
        self.train_mask_dir = os.path.join(args.data_path, 'mask')
        self.train_edge_dir = os.path.join(args.data_path, 'edge2')

        self.train_transform = train_image_preprocess(
            args.img_size, args.aug_option)
        self.test_transform = test_image_preprocess(img_size=args.img_size)

        self.train_loader = get_loader(img_dir=self.train_img_dir, mask_dir=self.train_mask_dir, edge_dir=self.train_edge_dir,
                                       phase='train', batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       transform=self.train_transform, seed=args.seed)
        # Model
        self.model = Net(args).to(self.device)

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = Scheduler(args, self.optimizer)

        #Train and Validate
        min_mae = 1000000000
        early_terminate = 0
        t = time.time()
        for epoch in range(1, args.epochs + 1):
            if (epoch-1)%30==0:
                adjust_lr(self.optimizer, self.args.lr, epoch, decay_epoch=30)

            self.epoch = epoch
            train_loss, train_mae = self.training(args)
            val_loss, val_mae = self.validating()

            self.scheduler.step(val_loss)

            if val_mae < min_mae:
                early_terminate = 0
                best_epoch = epoch
                best_mae = val_mae
                min_mae = val_mae
                # torch.save(self.model.module.state_dict(),os.path.join(args.model_path,'best_model.pth'))  # multi gpus
                torch.save(self.model.state_dict(), os.path.join(
                    args.model_path, 'best_model.pth'))    # single gpu

                print(
                    f'-----------------SAVE:{best_epoch}epoch------------------')
            else:
                early_terminate += 1
                
        print(f'\n\nBest Val Epoch is {best_epoch} | Val_loss:{min_mae:.5f} | Val_MAE:{best_mae:.5f}'                        
              f'time: {(time.time() - t)/60:.3f} Mins')

    def training(self, args):
        self.model.train()
        train_loss = AvgMeter()
        train_mae = AvgMeter()

        for images, masks, edges in tqdm(self.train_loader):
            images = torch.tensor(
                images, device=self.device, dtype=torch.float32)
            masks = torch.tensor( 
                masks, device=self.device, dtype=torch.float32)
            edges = torch.tensor(
                edges, device=self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            out1, out2, out3, edge = self.model(images)
            
            loss1 = iou(out1, masks) + F.binary_cross_entropy_with_logits(out1, masks)
            loss2 = iou(out2, masks) + F.binary_cross_entropy_with_logits(out2, masks)
            loss3 = iou(out3, masks) + F.binary_cross_entropy_with_logits(out3, masks)
            losse = iou(edge, edges) + F.binary_cross_entropy_with_logits(edge, edges)

            loss = loss3*0.7 + losse * 0.3 + loss1*0.25 + loss2*0.5

            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
            self.optimizer.step() 

            # Metric
            mae = 0.0
            for xx in range(masks.size(0)):
                outx = F.sigmoid(out3[xx].unsqueeze(0))
                mae = mae + torch.mean(torch.abs(outx-masks[xx].unsqueeze(0)))
            mae /= masks.size(0)

            train_loss.update(loss.item(), n=images.size(0))
            train_mae.update(mae.item(), n=images.size(0))
            # print(f'loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}, loss3: {loss3.item():.4f}, losse: {losse.item():.4f}, loss: {loss.item():.4f}')

        print(f'Epoch: [ {self.epoch:03d} / {args.epochs:03d} ]')
        print(
            f'Train Loss: [ {train_loss.avg:.5f} | MAE: {train_mae.avg:.5f} ]')

        return train_loss.avg, train_mae.avg
        
def main(cfg):
    print(">>>>>>>>>Blow are the Training Parameters<<<<<<<<<<")
    pprint.pprint(cfg)

    seed = cfg.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    if cfg.action == 'train':
        Trainer(cfg)
    else:
        Test(cfg)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    cfg = getConfig()
    main(cfg=cfg)
