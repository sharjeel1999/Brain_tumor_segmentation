import os
import zipfile
import nibabel as nib
import numpy as np

data_paths_array = []


data_folder_path = 'E:\\Datasets\\BraTS_Dataset'

for sub_folder in os.listdir(data_folder_path):
    #print(subfolder)
    for file in os.listdir(os.path.join(data_folder_path, sub_folder)):
        
        basename = os.path.splitext(file)[0]
        basename = os.path.splitext(basename)[0]
        #print(file)
        
        if basename[-3:] =='seg':
            print('Seg: ', basename)
            #mask = nib.load(os.path.join(data_folder_path, sub_folder, file))
            mask = os.path.join(data_folder_path, sub_folder, file)
        
        if basename[-2:] == 't1':
            print('t1: ', basename)
            #t1 = nib.load(os.path.join(data_folder_path, sub_folder, file))
            t1 = os.path.join(data_folder_path, sub_folder, file)
        
        if basename[-4:] == 't1ce':
            print('t1ce: ', basename)
            #t1ce = nib.load(os.path.join(data_folder_path, sub_folder, file))
            t1ce = os.path.join(data_folder_path, sub_folder, file)
            
        if basename[-2:] == 't2':
            print('t2: ', basename)
            #t2 = nib.load(os.path.join(data_folder_path, sub_folder, file))
            t2 = os.path.join(data_folder_path, sub_folder, file)
        
        if basename[-5:] == 'flair':
            print('flair: ', basename)
            #flair = nib.load(os.path.join(data_folder_path, sub_folder, file))
            flair = os.path.join(data_folder_path, sub_folder, file)
        
        #print(image.shape)
        #sx, sy, sz = image.header.get_zooms()
        #image = nib.load(os.path.join(data_folder_path, sub_folder, file))
        #print('pixel spacing: ', sx, sy, sz)
        
    data_paths_array.append([t1, t1ce, t2, flair, mask])
    
    #print('\n')

data_paths_array = np.array(data_paths_array)
print('Data length: ', len(data_paths_array))

np.save('Brain_data_paths_array.npy', data_paths_array)