import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
from glob import glob
import os.path
import re
import random
import cv2
from torch.utils import data

STF_CLASSES= [
    'PassengerCar', 'LargeVehicle', 'RidableVehicle',
    'Pedestrian', 'DontCare',
    'Pedestrian_is_group', 'Obstacle', 'Vehicle'
    ]


class Class_to_ind_stf(object):
    def __init__(self,binary,binary_item):
        self.binary=binary
        self.binary_item=binary_item
        self.classes=STF_CLASSES

    def __call__(self, name):
        if not name in self.classes:
            raise ValueError('No such class name : {}'.format(name))
        else:
            if self.binary:
                if name==self.binary_item:
                    return True
                else:
                    return False
            else:
                return self.classes.index(name)

class AnnotationTransform_stf(object):
    '''
    Transform Kitti detection labeling type to norm type:
    source: Car          0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00  1.75 13.22 1.62
    STF:    PassengerCar 0.00 2 -1   421.56 486.74 693.47 599.85 1.35 1.75 3.72 -5.54 1.11 30.04 0.852 0.000 0.000 2.423 1.00 0.0000000000 0.0000000000 0.9361579423 0.3515797308 True True True False

    target: [xmin,ymin,xmax,ymax,label_ind]

    levels=['easy','medium']
    '''
    def __init__(self,Class_to_ind_stf=Class_to_ind_stf(True,'Car'),levels=['easy','medium','hard']):
        self.Class_to_ind_stf=Class_to_ind_stf
        self.levels=levels if isinstance(levels,list) else [levels]

    def __call__(self,target_lines,width,height):

        res=list()
        for line in target_lines:
            xmin,ymin,xmax,ymax=tuple(line.strip().split(' ')[4:8])
            bnd_box=[xmin,ymin,xmax,ymax]
            new_bnd_box=list()
            for i,pt in enumerate(range(4)):
                cur_pt=float(bnd_box[i])
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                new_bnd_box.append(cur_pt)
            label_idx=self.Class_to_ind_stf(line.split(' ')[0])
            new_bnd_box.append(label_idx)
            res.append(new_bnd_box)
        return res

class STFLoader(data.Dataset):
    # TODO: Is img_size in STF 512?
    def __init__(self, root, split="training",
                 img_size=512, transforms=None,target_transform=None):
        self.root = root
        self.split = split
        self.target_transform = target_transform
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892]) # TODO
        self.files = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.transforms = transforms
        self.name='stf'

        # TODO these are all files;
        # You need to split them in train & test
        file_list = glob(os.path.join(root, 'cam_stereo_left_lut', '*.png'))
        self.files["training"] = file_list

        label_list=glob(os.path.join(root, 'cam_left_labels_TMP', '*.txt'))
        self.labels["training"] = label_list


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = img_name

        #img = m.imread(img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        #img = np.array(img, dtype=np.uint8)

        if self.split != "testing":
            lbl_path = self.labels[self.split][index]
            lbl_lines=open(lbl_path,'r').readlines()
            if self.target_transform is not None:
                target = self.target_transform(lbl_lines, width, height)
        else:
            lbl = None

        # if self.is_transform:
        #     img, lbl = self.transform(img, lbl)

        if self.transforms is not None:
            target = np.array(target)
            img, boxes, labels = self.transforms(img, target[:, :4], target[:, 4])
            #img, lbl = self.transforms(img, lbl)
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))            

        if self.split != "testing":
            #return img, lbl
            return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        else:
            return img

