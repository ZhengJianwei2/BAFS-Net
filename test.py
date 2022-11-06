#!/usr/bin/python3
# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import test_image_preprocess, get_loader
from model import Net
from config import getConfig


class Test(object):
    def __init__(self, cfg):
        ## dataset
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_transform = test_image_preprocess(img_size=cfg.img_size)
        self.save_path = cfg.save_map_path 

        self.model = Net(cfg).to(self.device)
        
        path = os.path.join(self.cfg.model_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(path))

        te_img_folder ='./SodDataset/EORSSD/testset/image'
        te_gt_folder ='./SodDataset/EORSSD/testset/mask'
        self.test_loader = get_loader(img_dir=te_img_folder, mask_dir=te_gt_folder, edge_dir=None, phase='test',
                                      batch_size=cfg.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers, transform=self.test_transform)
        
        self.test_processing()

    def test_processing(self):
        self.model.eval()
        with torch.no_grad():
            for i, (images, masks, original_size, image_name) in enumerate(tqdm(self.test_loader)):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                _,_,out5,e = self.model(images)

                H, W = original_size
                for i in range(images.size(0)):
                    h, w = H[i].item(), W[i].item()
                    

                    output = F.interpolate(out5[i].unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
                    output = (output.sigmoid().data.cpu().numpy().squeeze()*255.0).round().astype(np.uint8)

                    cv2.imwrite(os.path.join('./eval/maps/pred/EORSSD/Ours', image_name[i]+'.png'), output)

        
#run test
cfg = getConfig()
Test(cfg)
