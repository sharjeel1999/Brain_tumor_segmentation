import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2

data_path = 'D:\\brain_tumor_segmentation\\rough\\Brain_data_paths_array.npy'

data_array = np.load(data_path, allow_pickle=True)

print(len(data_array))

t1_path, t1ce_path, t2_path, flair_path, mask_path = data_array[1]

t1 = nib.load(t1_path).get_fdata()
t1ce = nib.load(t1ce_path).get_fdata()
t2 = nib.load(t2_path).get_fdata()
flair = nib.load(flair_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

print('All shapes: ', t1.shape, t1ce.shape, t2.shape, flair.shape, mask.shape)

print('Mask unique: ', np.unique(mask))
print('slices with mask: ', np.unique(np.where(mask == 4)[2]))

print('t1 min max: ', np.min(t1), np.max(t1))
print('t1ce min max: ', np.min(t1ce), np.max(t1ce))
print('t2 min max: ', np.min(t2), np.max(t2))
print('flair min max: ', np.min(flair), np.max(flair))

#t1[t1 > 1000] = 1000
#t1[t1 < 400] = 400

plt.imshow(t1[:, :, 70])
plt.show()

#t1ce[t1ce < 1000] = 0
plt.imshow(t1ce[:, :, 70])
plt.show()

#t2[t2 > 2000] = 0
#t2[t2 < 200] = 0
plt.imshow(t2[:, :, 70])
plt.show()

#flair[flair > 2000] = 0
plt.imshow(flair[:, :, 70])
plt.show()

plt.imshow(mask[:, :, 70])
plt.show()

b_slice = t1ce[:, :, 70]
b_slice = (b_slice / np.max(b_slice))*255
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl1 = clahe.apply(b_slice.astype(np.uint8))

plt.imshow(cl1)
plt.show()