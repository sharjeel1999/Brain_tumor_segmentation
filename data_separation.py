import numpy as np

data_path = 'D:\\brain_tumor_segmentation\\rough\\Brain_data_paths_array.npy'
#data_path = 'C:\\Users\\Sharjeel\\Desktop\\Brain_data_paths_array.npy'
total_data = np.load(data_path, allow_pickle = True)

selected_inds = [0,1,2,3,4,279,280,281,282,283,284,783,784,785,786,787,869,870,871,872]
print(len(selected_inds))

training_arr = []
testing_arr = []

for i, sample in enumerate(total_data):
    t1_path, t1ce_path, t2_path, flair_path, mask_path = sample
    
    if i in selected_inds:
        testing_arr.append([t1_path, t1ce_path, t2_path, flair_path, mask_path])
    else:
        training_arr.append([t1_path, t1ce_path, t2_path, flair_path, mask_path])
    
training_arr = np.array(training_arr)
testing_arr = np.array(testing_arr)
print('training shape: ', training_arr.shape)
print('testing shape: ', testing_arr.shape)

np.save('Training_val.npy', training_arr)
np.save('Testing.npy', testing_arr)