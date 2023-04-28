import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import math
from tqdm import tqdm

# data_path = 'D:\\brain_tumor_segmentation\\rough\\Brain_data_paths_array.npy'

# data_array = np.load(data_path, allow_pickle=True)

# print(len(data_array))

# t1_path, t1ce_path, t2_path, flair_path, mask_path = data_array[1]

# t1 = nib.load(t1_path).get_fdata()
# t1ce = nib.load(t1ce_path).get_fdata()
# t2 = nib.load(t2_path).get_fdata()
# flair = nib.load(flair_path).get_fdata()
# mask = nib.load(mask_path).get_fdata()

# print('All shapes: ', t1.shape, t1ce.shape, t2.shape, flair.shape, mask.shape)

# print('Mask unique: ', np.unique(mask))
# print('slices with mask: ', np.unique(np.where(mask == 4)[2]))

# print('t1 min max: ', np.min(t1), np.max(t1))
# print('t1ce min max: ', np.min(t1ce), np.max(t1ce))
# print('t2 min max: ', np.min(t2), np.max(t2))
# print('flair min max: ', np.min(flair), np.max(flair))


#mask[mask != 4] = 0
#mask[mask == 4] = 1
#t1[t1 > 1000] = 1000
#t1[t1 < 400] = 400

#plt.imshow(t1[:, :, 70])
#plt.show()

#t1ce[t1ce < 1000] = 0
#plt.imshow(t1ce[:, :, 70])
#plt.show()

#t2[t2 > 2000] = 0
#t2[t2 < 200] = 0
#plt.imshow(t2[:, :, 70])
#plt.show()

#flair[flair > 2000] = 0
#plt.imshow(flair[:, :, 70])
#plt.show()

# for z in range(74, 75):
    
#     b_slice = t1ce[:, :, z]
#     plt.imshow(b_slice)
#     plt.show()
    
    
        
#     b_slice = (b_slice / np.max(b_slice))*255
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     cl1 = clahe.apply(b_slice.astype(np.uint8))
    
#     mid = 0.09
#     mean = np.mean(cl1)
#     gamma = math.log(mid*255)/math.log(mean)
#     img_gamma1 = np.power(cl1, gamma).clip(0,255).astype(np.uint8)
    
#     plt.imshow(cl1)
#     plt.show()
    
#     plt.imshow(img_gamma1)
#     plt.show()
    
#     plt.imshow(mask[:, :, z])
#     plt.show()
    

# plt.imshow(mask[:, :, 70])
# plt.show()

# b_slice = t1ce[:, :, 70]
# b_slice = (b_slice / np.max(b_slice))*255
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl1 = clahe.apply(b_slice.astype(np.uint8))

# plt.imshow(cl1[:, :, 70])
# plt.show()

data_path = 'D:\\brain_tumor_segmentation\\rough\\Brain_data_paths_array.npy'

data_array = np.load(data_path, allow_pickle=True)
data_len = len(data_array)
data_array = data_array[data_len-35:data_len-30]#data_len-35]
print(len(data_array))

c0_sum = 0
c1_sum = 0
c2_sum = 0
c3_sum = 0

for sample in tqdm(data_array):
    t1_path, t1ce_path, t2_path, flair_path, mask_path = sample
    
    mask = nib.load(mask_path).get_fdata()
    mask = mask.astype(np.uint8)
    print('mask unique: ', np.unique(mask))
    c0 = np.zeros_like(mask)
    c1 = np.zeros_like(mask)
    c2 = np.zeros_like(mask)
    c3 = np.zeros_like(mask)
    
    c0 = np.where(mask == 0, 1, 0)
    c1 = np.where(mask == 1, 1, 0) #[mask == 1] == 1
    c2 = np.where(mask == 2, 1, 0) #[mask == 2] == 1
    c3 = np.where(mask == 4, 1, 0) #[mask == 4] == 1
    
    c0_sum += np.sum(c0)
    c1_sum += np.sum(c1)
    c2_sum += np.sum(c2)
    c3_sum += np.sum(c3)
    
print(c0_sum)
print(c1_sum)
print(c2_sum)
print(c3_sum)