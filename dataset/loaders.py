import os
import numpy as np
import cv2
import math
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler
import random

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class Data_slicer():
    def __init__(self, data_path, slices):
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        self.slices = slices
        self.get_samples_primary()
        self.get_samples_secondary()
        
        self.inds_record_primary = np.array(self.inds_record_primary)
        self.inds_record_secondary = np.array(self.inds_record_secondary)
        
        self.Combine_inds()
    
    def get_samples_primary(self):
        self.inds_record_primary = []
        
        for ind, sample in enumerate(self.data):
            t1, t1ce, t2, flair, mask_path = sample
            mask = nib.load(mask_path).get_fdata()
            
            un_inds = np.unique(np.where(mask == 4)[2])
            prev = 0
            for sel_ind in un_inds:
                if sel_ind - ((self.slices-1)/2) >= 0 and sel_ind + ((self.slices-2)/2) and sel_ind > prev+3:
                    self.inds_record_primary.append([ind, sel_ind])
                    prev = sel_ind
        #print('Primary Data len: ', len(self.inds_record_primary))

    def get_samples_secondary(self):
        self.inds_record_secondary = []
        
        for ind, sample in enumerate(self.data):
            t1, t1ce, t2, flair, mask_path = sample
            mask = nib.load(mask_path).get_fdata()
            
            c1_inds = np.where(mask == 1, 1, 0)
            nc4_inds = np.where(mask == 4, 0, 1)
            
            mul_inds = c1_inds * nc4_inds
            
            un_inds = np.unique(np.where(mul_inds == 1)[2])
            prev = 0
            for sel_ind in un_inds:
                if sel_ind - ((self.slices-1)/2) >= 0 and sel_ind + ((self.slices-2)/2) and sel_ind > prev+3:
                    self.inds_record_secondary.append([ind, sel_ind])
                    prev = sel_ind
        #print('Secondary Data len: ', len(self.inds_record_secondary))

    def Combine_inds(self):
        #print('primary shape: ', self.inds_record_primary.shape)
        #print('secondary shape: ', self.inds_record_secondary.shape)
        
        random.shuffle(self.inds_record_primary)
        random.shuffle(self.inds_record_secondary)
        
        self.combined_data = np.concatenate((self.inds_record_primary, self.inds_record_secondary), axis = 0)
        
        self.primary_indices = range(0, self.inds_record_primary.shape[0])
        self.secondary_indices = range(self.inds_record_primary.shape[0], self.inds_record_primary.shape[0]+self.inds_record_secondary.shape[0])
        # print('in primary: ', self.primary_indices)
        # print('in secondary: ', self.secondary_indices)
        
    def get_inds(self):
        return self.primary_indices, self.secondary_indices
    
    def get_data(self):
        return self.combined_data


class Prepare_dataset(Dataset):
    def __init__(self, data_path, inds_record, slices):
        
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        self.slices = slices
        self.inds_record = inds_record
        print('Prepared samples')
        
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.inds_record)
    
        
    def preprocess(self, b_slice):
        b_slice = (b_slice / np.max(b_slice))*255
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl1 = clahe.apply(b_slice.astype(np.uint8))
        
        mid = 0.09
        mean = np.mean(cl1)
        gamma = math.log(mid*255)/math.log(mean)
        img_gamma1 = np.power(cl1, gamma).clip(0,255).astype(np.uint8)
        
        return img_gamma1

    def get_combined_slices_order(self, t1ce, flair, mask, selected_ind):
        #print('loader shapes: ', t1ce.shape, flair.shape)
        start = int(selected_ind - ((self.slices - 1)/2))
        end = int(selected_ind + ((self.slices - 1)/2) + 1)
        
        input1 = torch.zeros((2, 240, 240))
        input2 = torch.zeros((2, self.slices, 240, 240))
        gt_masks = torch.zeros((1, 240, 240))
        
        for k, z in enumerate(range(start, end)):
            #t1_slice = t1[z, :, :]
            t1ce_slice = t1ce[z, :, :]
            #t2_slice = t2[z, :, :]
            flair_slice = flair[z, :, :]
            mask_slice = mask[z, :, :]
            
            #input2[0, k, :, :] = t1_slice
            input2[0, k, :, :] = t1ce_slice
            #input2[2, k, :, :] = t2_slice
            input2[1, k, :, :] = flair_slice
            
            if z == selected_ind:
                #input1[0, :, :] = t1_slice
                input1[0, :, :] = t1ce_slice
                #input1[2, :, :] = t2_slice
                input1[1, :, :] = flair_slice
                gt_masks[0, :, :] = mask_slice
        
        return input1, input2, gt_masks
        
    
    def __getitem__(self, index):
        
        ind, selected_ind = self.inds_record[index]
        t1_path, t1ce_path, t2_path, flair_path, mask_path = self.data[ind]
        
        #t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        #t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        #t1 = self.transform(t1)
        t1ce = self.transform(t1ce)
        #t2 = self.transform(t2)
        flair = self.transform(flair)
        mask = self.transform(mask)
        
        input1_sample, input2_sample, mask_sliced = self.get_combined_slices_order(t1ce, flair, mask, selected_ind)
        
        mask_sliced[mask_sliced == 4] = 3
        
        
        input1_sample = (input1_sample - input1_sample.min()) / (input1_sample.max() - input1_sample.min())
        input2_sample = (input2_sample - input2_sample.min()) / (input2_sample.max() - input2_sample.min())
        
        return input1_sample, input2_sample, mask_sliced
        

class Prepare_full_volume_dataset(Dataset):
    def __init__(self, data_path, slices):
        
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        self.slices = slices
        #self.get_samples()
        
        self.transform = transforms.ToTensor()
        #self.prepare_dataset()
        
        
    def __len__(self):
        return len(self.data)
        
    
    def __getitem__(self, index):
        
        t1_path, t1ce_path, t2_path, flair_path, mask_path = self.data[index]
        
        t1ce = nib.load(t1ce_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        #t1ce = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min())
        #flair = (flair - flair.min()) / (flair.max() - flair.min())
        
        mask[mask == 4] = 3
        #print('mask shape: ', mask.shape)
        
        t1ce = torch.FloatTensor(t1ce)
        flair = torch.FloatTensor(flair)
        mask = torch.Tensor(mask)
        t1ce = torch.permute(t1ce, (2, 0, 1))
        flair = torch.permute(flair, (2, 0, 1))
        mask = torch.permute(mask, (2, 0, 1))
        
        return t1ce, flair, mask

class Prepare_test_dataset(Dataset):
    def __init__(self, data_path):
        
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        t1_path, t1ce_path, t2_path, flair_path, mask_path = self.data[index]
        
        #t1 = nib.load(t1_path).get_fdata()
        t1cee = nib.load(t1ce_path).get_fdata()
        #t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        # t1ce = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min())
        # flair = (flair - flair.min()) / (flair.max() - flair.min())
        
        mask[mask == 4] = 3
        #t1 = self.transform(t1)
        t1ce = self.transform(t1cee)
        #t2 = self.transform(t2)
        flair = self.transform(flair)
        mask = self.transform(mask)
        
        return t1_path, t1ce_path, t2_path, flair_path, t1ce, flair, mask