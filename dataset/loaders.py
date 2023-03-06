
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch


class Prepare_dataset(Dataset):
    def __init__(self, data_path, slices):
        
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        self.slices = slices
        self.get_samples()
            
    def __len__(self):
        return len(self.inds_record)
    
    def get_samples(self):
        self.inds_record = []
        
        for ind, sample in enumerate(self.data):
            t1, t1ce, t2, flair, mask = sample
            
            un_inds = np.unique(np.where(mask > 0)[2])
            for sel_ind in un_inds:
                if sel_ind - ((self.slices-1)/2) >= 0 and sel_ind + ((self.slices-2)/2):
                    
                    self.inds_record.append(sel_ind)
                    
    def __getitem__(self, index):
        
        
            
            