import SimpleITK as sitk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

data_path = 'D:\\brain_tumor_segmentation\\rough_4\\Brain_data_paths_array_2018.npy'
total_data = np.load(data_path, allow_pickle = True)
t1_path, t1ce_path, t2_path, flair_path, mask_path = total_data[1]
t1cee_p = nib.load(t1ce_path).get_fdata()
flair_p = nib.load(flair_path).get_fdata()
mask_p = nib.load(mask_path).get_fdata()

maskImage = sitk.OtsuThreshold(flair_p, 0, 1, 200)
maskImagePath = input('Enter the name of the mask image to be saved : ')
sitk.WriteImage(maskImage, maskImagePath)
print("Mask image is saved.")

inputImage = sitk.Cast(flair_p,sitk.sitkFloat32)

corrector = sitk.N4BiasFieldCorrectionImageFilter();

output = corrector.Execute(inputImage,maskImage)

print(output.shape)
print(flair_p.shape)

plt.imshow(flair_p[:, :, 19])
plt.show()

plt.imhow(output)
plt.show()
