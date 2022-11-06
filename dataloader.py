import cv2
import glob
import torch
import numpy as np
import albumentations as albu
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def np_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device = 'cuda', dtype = torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt

def get_loader(img_dir, mask_dir, edge_dir, phase, batch_size, shuffle,
               num_workers, transform, seed = None):
    if phase == 'test':
        dataset = TestDatasetFactory(img_dir,mask_dir,transform)
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
        num_workers=num_workers)
    else:
        dataset = DatasetFactory(img_dir=img_dir,mask_dir=mask_dir,edge_dir=edge_dir,phase=phase,transform=transform,seed=seed)
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True) 
    
    print(f'{phase} length : {len(dataset)}')

    return data_loader


class DatasetFactory(Dataset):
    def __init__(self,img_dir,mask_dir,edge_dir,phase,transform=None,seed=None):
        super().__init__()
        self.image = sorted(glob.glob(img_dir+'/*'))
        self.mask = sorted(glob.glob(mask_dir+'/*'))
        self.edge = sorted(glob.glob(edge_dir+'/*'))
        self.transform = transform
                                                                                                       
    
    def __getitem__(self,idx):
        image = cv2.imread(self.image[idx])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask[idx])
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        edge = cv2.imread(self.edge[idx])
        edge = cv2.cvtColor(edge,cv2.COLOR_BGR2GRAY)

        if self.transform is not None:  # augmentation
            enhanced = self.transform(image=image, masks=[mask, edge])
            image = enhanced['image']
            mask = np.expand_dims(enhanced['masks'][0],axis=0)   # [H,W] -> [1,H,W]
            edge = np.expand_dims(enhanced['masks'][1],axis=0)   # [H,W] -> [1,H,W]
            mask = mask / 255.0
            edge = edge / 255.0

        return image, mask, edge

    def __len__(self):
        return len(self.image)

class TestDatasetFactory(Dataset):
    def __init__(self,img_dir,mask_dir,transform=None) -> None:
        self.image = sorted(glob.glob(img_dir + '/*'))
        self.mask = sorted(glob.glob(mask_dir + '/*'))
        self.transform = transform 

    def __getitem__(self, idx):
        img_name = Path(self.image[idx]).stem
        image = cv2.imread(self.image[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        initial_size = image.shape[:2]
        
        if self.transform is not None:
            enhanced = self.transform(image=image)
            image = enhanced['image']
        
        return image, self.mask[idx], initial_size, img_name
    
    def __len__(self):
        return len(self.image)


def test_image_preprocess(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        ),
        ToTensorV2(),
    ]
    )
    return transforms

def train_image_preprocess(img_size, aug_option):
    if aug_option == 1:
        transforms = albu.Compose([
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485,0.456,0.406],
                           [0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    elif aug_option ==2:
        transforms = albu.Compose(
            [
                albu.OneOf([
                    albu.HorizontalFlip(),
                    albu.VerticalFlip(),
                    albu.RandomRotate90(),
                ], p=0.5),
                albu.OneOf([
                    albu.RandomContrast(),
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                ], p= 0.5),
                albu.Resize(img_size, img_size, always_apply=True),
                albu.Normalize([0.485,0.456,0.406],
                               [0.229,0.224,0.225]),
                ToTensorV2(),
            ]
        )
    else:
        pass
    return transforms


