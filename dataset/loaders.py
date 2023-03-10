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
        print('Prepared samples')
        
        self.transform = transforms.ToTensor()
        #self.prepare_dataset()
        
        
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
        print('Data len: ', len(self.inds_record))
                    
    def get_combined_slices(self, t1, t1ce, t2, flair, mask, selected_ind):
        
        t1_sliced = t1[:, :, int(selected_ind - ((self.slices - 1)/2)) : int(selected_ind + ((self.slices - 1)/2) + 1)]
        t1ce_sliced = t1ce[:, :, int(selected_ind - ((self.slices - 1)/2)) : int(selected_ind + ((self.slices - 1)/2) + 1)]
        t2_sliced = t2[:, :, int(selected_ind - ((self.slices - 1)/2)) : int(selected_ind + ((self.slices - 1)/2) + 1)]
        flair_sliced = flair[:, :, int(selected_ind - ((self.slices - 1)/2)) : int(selected_ind + ((self.slices - 1)/2) + 1)]
        mask_sliced = mask[:, :, selected_ind]
        
        concatenated1 = np.concatenate((t1ce_sliced, flair_sliced), axis = 2)
        
        t1ce_sliced_2 = t1ce[:, :, int(selected_ind - ((self.slices-1)/2)):selected_ind + 1]
        flair_sliced_2 = flair[:, :, int(selected_ind - ((self.slices-1)/2)):selected_ind + 1]
        mask_sliced_2 = mask[:, :, int(selected_ind - ((self.slices-1)/2)):selected_ind]
        
        concatenated2 = np.concatenate((t1ce_sliced_2, flair_sliced_2, mask_sliced_2), axis = 2)
        
        return concatenated1, concatenated2, mask_sliced
        
    def get_combined_slices_order(self, t1, t1ce, t2, flair, mask, selected_ind):
        start = int(selected_ind - ((self.slices - 1)/2))
        end = int(selected_ind + ((self.slices - 1)/2) + 1)
        
        input1 = []
        input2 = []
        gt_masks = []
        
        for z in range(start, end):
            t1_slice = t1[:, :, z]
            t1ce_slice = t1ce[:, :, z]
            t2_slice = t2[:, :, z]
            flair_slice = flair[:, :, z]
            mask_slice = mask[:, :, z]
            
            input1.append(t1ce_slice)
            input1.append(flair_slice)
            
            if z < selected_ind:
                input2.append(t1ce_slice)
                input2.append(flair_slice)
                input2.append(mask_slice)
            
            if z == selected_ind:
                input2.append(t1ce_slice)
                input2.append(flair_slice)
                gt_masks.append(mask_slice)
        
        
        input1 = np.array(input1)
        input2 = np.array(input2)
        gt_masks = np.array(gt_masks)
        
        input1 = np.transpose(input1, (1, 2, 0))
        input2 = np.transpose(input2, (1, 2, 0))
        gt_masks = np.transpose(gt_masks, (1, 2, 0))
        
        
        return input1, input2, gt_masks
    
    def prepare_dataset(self):
        self.prepared_data = []
        
        for i in range(len(self.inds_record)):
            ind, selected_ind = self.inds_record[i]
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
            
            
            input1_sample, input2_sample, mask_sliced = self.get_combined_slices_order(t1, t1ce, t2, flair, mask, selected_ind)
            
            self.prepared_data.append([input1_sample, input2_sample, mask_sliced])
        
    
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
        
        
        input1_sample, input2_sample, mask_sliced = self.get_combined_slices_order(t1, t1ce, t2, flair, mask, selected_ind)
        
        #input1_sample, input2_sample, mask_sliced = self.prepared_data[index]
        #print('inter input1: ', np.array(input1_sample).shape)
        #print('inter input2: ', np.array(input2_sample).shape)
        #print('mask sliced: ', np.array(mask_sliced).shape)
        
        mask_sliced[mask_sliced == 4] = 3
        
        input1_sample = self.transform(input1_sample)
        input2_sample = self.transform(input2_sample)
        mask_sliced = self.transform(mask_sliced)
        
        return input1_sample, input2_sample, mask_sliced
        