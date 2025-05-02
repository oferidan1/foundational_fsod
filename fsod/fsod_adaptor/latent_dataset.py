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
    #load all files to dict
    for file in  files:
        info = torch.load(file)
        #info fields: {'embeds', 'scores', "ious"}    
        scores = info['scores']
        ious = torch.from_numpy(info['ious'])
        embeds = info['embeds']
        #set embeds to buckets of HP, EP, HN, EN according to ious and scores        
        iou_thr = 0.5
        score_thr = 0.05
        EP.append(embeds[(ious>iou_thr)&(scores>score_thr)])
        HP.append(embeds[(ious>iou_thr)&(scores<=score_thr)])
        HN.append(embeds[(ious<=iou_thr)&(scores>score_thr)])
        EN.append(embeds[(ious<=iou_thr)&(scores<=score_thr)])
    
    EP = torch.cat(EP, dim=0)
    HP = torch.cat(HP, dim=0)
    EN = torch.cat(EN, dim=0)
    HN = torch.cat(HN, dim=0)
    
    return EP, HP, EN, HN