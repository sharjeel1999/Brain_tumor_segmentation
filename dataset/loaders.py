import os
import numpy as np
import cv2
import math
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms as t
import itertools
from torch.utils.data.sampler import Sampler
import random
from torch.nn.functional import interpolate
from monai import data, transforms

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

def pre_process(in_slice):
    print('in shape: ', in_slice.shape)
    # print('in min max: ', np.min(in_slice), np.max(in_slice))
    in_slice = ((in_slice/np.max(in_slice))*255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    cl = clahe.apply(in_slice)
    return cl

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
                    # print(ind)
                    self.inds_record_primary.append([ind, sel_ind])
                    prev = sel_ind
        #print('Primary Data len: ', len(self.inds_record_primary))
        
    
    def get_samples_secondary(self):
        self.inds_record_secondary = []
        
        for ind, sample in enumerate(self.data):
            t1, t1ce, t2, flair, mask_path = sample
            mask = nib.load(mask_path).get_fdata()
            
            # c1_inds = np.where(mask == 1, 1, 0)
            # nc4_inds = np.where(mask == 4, 0, 1)
            # mul_inds = c1_inds * nc4_inds
            
            inds4 = np.array(np.unique(np.where(mask == 4)[2])).tolist()
            inds2 = np.array(np.unique(np.where(mask == 2)[2])).tolist()
            inds1 = np.array(np.unique(np.where(mask == 1)[2])).tolist()
            
            diff_ab = np.setdiff1d(inds4, np.intersect1d(inds4, inds2, inds2))
            diff_bc = np.setdiff1d(inds2, np.intersect1d(inds4, inds2, inds2))
            diff_ca = np.setdiff1d(inds2, np.intersect1d(inds4, inds2, inds2))
            un_inds = np.union1d(np.union1d(diff_ab, diff_bc), diff_ca)
            
            # un_inds = np.unique(np.where(mul_inds == 1)[2])
            
            prev = 0
            for sel_ind in un_inds:
                if sel_ind - ((self.slices-1)/2) >= 0 and sel_ind + ((self.slices-2)/2) and sel_ind > prev+3:
                    # print(ind)
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
        
        self.transform = t.ToTensor()
        
        self.train_transform = transforms.Compose(
        [
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

        ])
        
    def __len__(self):
        return len(self.inds_record)
    
        
    # def preprocess(self, b_slice):
    #     b_slice = (b_slice / np.max(b_slice))*255
    #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    #     cl1 = clahe.apply(b_slice.astype(np.uint8))
        
    #     mid = 0.09
    #     mean = np.mean(cl1)
    #     gamma = math.log(mid*255)/math.log(mean)
    #     img_gamma1 = np.power(cl1, gamma).clip(0,255).astype(np.uint8)
        
    #     return img_gamma1
    
    def hist_match(self, source, template):

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)
    
    def get_combined_slices_order(self, t1, t1ce, t2, flair, mask, selected_ind):
        #print('loader shapes: ', t1ce.shape, flair.shape)
        start = int(selected_ind - ((self.slices - 1)/2))
        end = int(selected_ind + ((self.slices - 1)/2) + 1)
        
        input1 = torch.zeros((4, 128, 128))
        input2 = torch.zeros((4, self.slices, 128, 128))
        gt_masks = torch.zeros((1, 128, 128))
        
        for k, z in enumerate(range(start, end)):
            t1_slice = t1[:, :, z]
            t1ce_slice = t1ce[:, :, z]
            t2_slice = t2[:, :, z]
            flair_slice = flair[:, :, z]
            mask_slice = mask[:, :, z]
            
            # print('flair shape before: ', flair_slice.shape)
            # temp = np.load('D:\\brain_tumor_segmentation\\rough_4\\template.npy', allow_pickle = True)
            # flair_slice = self.transform(self.hist_match(flair_slice, temp))
            # print('flair shape after: ', flair_slice.shape)
            
            input2[0, k, :, :] = t1_slice
            input2[1, k, :, :] = t1ce_slice
            input2[2, k, :, :] = t2_slice
            input2[3, k, :, :] = flair_slice
            
            if z == selected_ind:
                input1[0, :, :] = t1_slice
                input1[1, :, :] = t1ce_slice
                input1[2, :, :] = t2_slice
                input1[3, :, :] = flair_slice
                gt_masks[0, :, :] = mask_slice
        
        return input1, input2, gt_masks
        
    
    def __getitem__(self, index):
        
        ind, selected_ind = self.inds_record[index]
        ind, selected_ind = int(ind), int(selected_ind)
        # print('ind and lenght: ', ind, len(self.data))
        t1_path, t1ce_path, t2_path, flair_path, mask_path = self.data[ind]
        
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        print('before shape: ', t1.shape)
        t1 = self.transform(t1)
        t1ce = self.transform(t1ce)
        t2 = self.transform(t2)
        flair = self.transform(flair)
        mask = self.transform(mask)
        
        t1 = reshape_3d(t1, 128, 128, 64, mode = 'trilinear')
        t1ce = reshape_3d(t1ce, 128, 128, 64, mode = 'trilinear')
        t2 = reshape_3d(t2, 128, 128, 64, mode = 'trilinear')
        flair = reshape_3d(flair, 128, 128, 64, mode = 'trilinear')
        mask = reshape_3d(mask.float(), 128, 128, 64).long()
        
        print('after shape: ', t1.shape)
        
        input1_sample, input2_sample, mask_sliced = self.get_combined_slices_order(t1, t1ce, t2, flair, mask, selected_ind)
        
        mask_sliced[mask_sliced == 4] = 3
        
        
        input1_sample = (input1_sample - input1_sample.min()) / (input1_sample.max() - input1_sample.min())
        input2_sample = (input2_sample - input2_sample.min()) / (input2_sample.max() - input2_sample.min())
        
        # input1_sample = (input1_sample - input1_sample.mean()) / input1_sample.std()
        # input2_sample = (input2_sample - input2_sample.mean()) / input2_sample.std()
        
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
        
        t1ce = nib.load(t1_path).get_fdata()
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

# class reshape_3d(torch.nn.Module):

#     def init(self, height, width, depth, mode='nearest'):
#         super(reshape_3d, self).init()
#         self.height = height
#         self.width = width
#         self.depth = depth
#         self.mode = mode

#     def forward(self, x):

#         if len(x.shape) == 4:
#             x = x.unsqueeze(0)
#             x = interpolate(x,size=(self.height, self.width, self.depth), mode=self.mode, )
#             x = x.squeeze(0)
#         else:
#             x = x.unsqueeze(0).unsqueeze(0)
#             x = interpolate(x,size=(self.height, self.width, self.depth), mode=self.mode, )
#             x = x.squeeze(0).squeeze(0)
#         return x

def reshape_3d(x, height, width, depth, mode = 'nearest'):
    if len(x.shape) == 4:
        x = x.unsqueeze(0)
        x = interpolate(x,size=(height, width, depth), mode=mode, )
        x = x.squeeze(0)
    else:
        x = x.unsqueeze(0).unsqueeze(0)
        x = interpolate(x,size=(height, width, depth), mode=mode, )
        x = x.squeeze(0).squeeze(0)
        
    return x

import torchio.transforms as tio

class Prepare_test_dataset(Dataset):
    def __init__(self, data_path):
        
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        
        transforms_dict = {
            tio.RandomAffine(scales = (0.9, 1.1), degrees = 15):0.75,
                tio.RandomElasticDeformation(num_control_points = 7, locked_borders = 2):0.25,
            }
        self.aug_transform = tio.OneOf(transforms_dict)
        
        # self.affine
        
        self.transform = transforms.ToTensor()
        # self.resize = reshape_3d(128, 128, 128)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        t1_path, t1ce_path, t2_path, flair_path, mask_path = self.data[index]
        # print('data path: ', t1_path)
        t1 = nib.load(t1_path).get_fdata()
        t1cee = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        # proc_t1 = np.load(proc_t1_path, allow_pickle=True)
        # proc_t1ce = np.load(proc_t1ce_path, allow_pickle=True)
        # proc_t2 = np.load(proc_t2_path, allow_pickle=True)
        # proc_flair = np.load(proc_flair_path, allow_pickle=True)
        
        # print('shapes: ', t1.shape, proc_t1.shape)

        rint = np.random.randint(0, 4)
        
        if rint == 1:
           t1 = np.flipud(t1)
           t1cee = np.flipud(t1cee)
           t2 = np.flipud(t2)
           flair = np.flipud(flair)
           mask = np.flipud(mask)
       
        if rint == 2:
            t1 = np.fliplr(t1)
            t1cee = np.fliplr(t1cee)
            t2 = np.fliplr(t2)
            flair = np.fliplr(flair)
            mask = np.fliplr(mask)
            
        if rint == 3:
            t1 = np.flipud(t1)
            t1cee = np.flipud(t1cee)
            t2 = np.flipud(t2)
            flair = np.flipud(flair)
            mask = np.flipud(mask)
            
            t1 = np.fliplr(t1)
            t1cee = np.fliplr(t1cee)
            t2 = np.fliplr(t2)
            flair = np.fliplr(flair)
            mask = np.fliplr(mask)
        
        mask[mask == 4] = 3
        t1 = self.transform(t1)
        t1cee = self.transform(t1cee)
        t2 = self.transform(t2)
        flair = self.transform(flair)
        mask = self.transform(mask)
        # print('before shape: ', flair.shape)
        t1 = reshape_3d(t1, 128, 128, 64, mode = 'trilinear')
        t1cee = reshape_3d(t1cee, 128, 128, 64, mode = 'trilinear')
        t2 = reshape_3d(t2, 128, 128, 64, mode = 'trilinear')
        flair = reshape_3d(flair, 128, 128, 64, mode = 'trilinear')
        mask = reshape_3d(mask.float(), 128, 128, 64).long()
        
        # return t1_path, t1ce_path, t2_path, flair_path, t1cee, flair, mask
        return t1, t1cee, t2, flair, mask