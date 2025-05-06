from skimage.io import imread
from torch.utils.data import Dataset
from os.path import join
import numpy as np
import glob
from collections import defaultdict
from parse import parse
import torch

class LatentDataset(Dataset):
    """
        A class representing a dataset of latents info
    """

    def __init__(self, dataset_path):
       self.EP, self.HP, self.EN, self.HN = build_latents_data(dataset_path)
       self.dataset_size = len(self.EN)
       

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        EP = self.EP[idx%len(self.EP)]
        HP = self.HP[idx%len(self.HP)]
        EN = self.EN[idx]
        HN = self.HN[idx%len(self.HN)]
        return EP, HP, EN, HN        


def build_latents_data(dataset_path):
    files = glob.glob(dataset_path+'/*.pt')
    EP, HP, EN, HN = [], [], [], []
    iou_thr = 0.5
    score_thr = 0.2
    target_class = 7
    #load all files to dict
    for file in  files:
        info = torch.load(file)
        #info fields: {'embeds', 'scores', "ious", "gt_classes"}    
        scores = info['scores']
        ious_all = torch.from_numpy(info['ious'])
        embeds = info['embeds']
        gt_classes = info['gt_class']
        #set embeds to buckets of HP, EP, HN, EN according to ious and scores        
        for i in range(ious_all.shape[1]):        
            ious = ious_all[:,i]
            #if we are on target class images
            if gt_classes[i] == target_class:
                EP.append(embeds[(ious>iou_thr)&(scores>score_thr)])
                HP.append(embeds[(ious>iou_thr)&(scores<=score_thr)])
                HN.append(embeds[(ious<=iou_thr)&(scores>score_thr)])
                EN.append(embeds[(ious<=iou_thr)&(scores<=score_thr)])
            else:
                #look for HN in non target class
                if max(scores) > score_thr:
                    HN.append(embeds[scores>score_thr])                    
                    EN.append(embeds[scores<score_thr])                    
                else:
                    EN.append(embeds)                    
                    
    
    EP = torch.cat(EP, dim=0)
    HP = torch.cat(HP, dim=0)
    EN = torch.cat(EN, dim=0)
    HN = torch.cat(HN, dim=0)
    
    return EP, HP, EN, HN