#coding=utf-8
import os
import cv2
import numpy as np

def generate_edge():
    for name in os.listdir('./SodDataset/EORSSD/trainset/mask'):
        mask = cv2.imread('./SodDataset/EORSSD/trainset/mask/'+name,0)
        body = mask
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]

        if len(tmp)!=0:
            a = np.where(body>2)
            body[a] = 255

        savepath = './SodDataset/EORSSD/trainset/edge2/'
        os.makedirs(savepath,exist_ok=True)
        res = mask-body
        res = cv2.blur(res,(3,3))

        cv2.imwrite(savepath+name, res)

if __name__=='__main__':
    generate_edge()



