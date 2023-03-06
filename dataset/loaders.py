
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms

class Prepare_dataset(Dataset):
    def __init__(self, data_path, slices):
        
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        self.slices = slices
        self.get_samples()
        
        self.transform = transforms.ToTensor()
        
        
    def __len__(self):
        return len(self.inds_record)
    
    def get_samples(self):
        self.inds_record = []
        
        for ind, sample in enumerate(self.data):
            t1, t1ce, t2, flair, mask_path = sample
            mask = nib.load(mask_path).get_fdata()
            
            un_inds = np.unique(np.where(mask > 0)[2])
            for sel_ind in un_inds:
                if sel_ind - ((self.slices-1)/2) >= 0 and sel_ind + ((self.slices-2)/2):
                    
                    self.inds_record.append([ind, sel_ind])
                    
    def get_combined_slices(self, t1, t1ce, t2, flair, mask, selected_ind):
        
        t1_sliced = t1[:, :, selected_ind - ((self.slices-1)/2):selected_ind - ((self.slices-1)/2)]
        t1ce_sliced = t1ce[:, :, selected_ind - ((self.slices-1)/2):selected_ind - ((self.slices-1)/2)]
        t2_sliced = t2[:, :, selected_ind - ((self.slices-1)/2):selected_ind - ((self.slices-1)/2)]
        flair_sliced = flair[:, :, selected_ind - ((self.slices-1)/2):selected_ind - ((self.slices-1)/2)]
        mask_sliced = mask[:, :, selected_ind - ((self.slices-1)/2):selected_ind - ((self.slices-1)/2)]
        
        concatenated = np.concatenate((t1_sliced, t1ce_sliced, t2_sliced, flair_sliced))
        
        return concatenated, mask_sliced
        
        
                    
    def __getitem__(self, index):
        ind, selected_ind = self.inds_record[index]
        
        t1_path, t1ce_path, t2_path, flair_path, mask_path = self.data[ind]
        
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        t1 = (t1 - t1.min()) / (t1.max() - t1.min())
        t1ce = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min())
        t2 = (t2 - t2.min()) / (t2.max() - t2.min())
        flair = (flair - flair.min()) / (flair.max() - flair.min())
        
        
        input_sample, mask_sliced = self.get_combined_slices(t1, t1ce, t2, flair, mask)
        
        input_sample = self.transform(input_sample)
        mask_sliced = self.transform(mask_sliced)
        
        return input_sample, mask_sliced
        