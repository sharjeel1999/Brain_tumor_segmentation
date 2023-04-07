import numpy as np
import torch

from models import UNet, UNet_2D, UNet_3D, Discriminator, Decoder_2D, Decoder_3D, UNet_2D_red, UNet_3D_red, Decoder_3D_red, Dense_encoder_2d, Dense_encoder_3d, Dense_Decoder_2D, CRF
from dataset import Prepare_dataset, Prepare_test_dataset, Prepare_full_volume_dataset, Data_slicer, TwoStreamBatchSampler
#from core import Run_Model
from core_full_volume import Run_Model

batch = 8
epochs = 100
base_lr = 0.001

data_path = 'D:\\brain_tumor_segmentation\\rough_4\\Training_val.npy'
#data_path = 'C:\\Users\\Sharjeel\\Desktop\\Brain_data_paths_array.npy'
total_data = np.load(data_path, allow_pickle = True)

data_len = len(total_data)
print('Data len: ', data_len)
print('Train len: ', data_len - 15)

train_data = total_data[0:data_len-15]
validation_data = total_data[data_len-15:data_len] # 20

''' Train data prep '''
train_slicer = Data_slicer(train_data, slices = 5)
primary_indices, secondary_indices = train_slicer.get_inds()

train_sampler = TwoStreamBatchSampler(primary_indices, secondary_indices, batch, 1)
train_set = Prepare_dataset(train_data, train_slicer.get_data(), slices = 5)

Train_loader = torch.utils.data.DataLoader(train_set, num_workers = 3, pin_memory = True, batch_sampler = train_sampler)

'''validation data prep '''
validation_slicer = Data_slicer(validation_data, slices = 5)
val_primary_indices, val_secondary_indices = validation_slicer.get_inds()

validation_sampler = TwoStreamBatchSampler(val_primary_indices, val_secondary_indices, batch, 1)
validation_set = Prepare_dataset(validation_data, validation_slicer.get_data(), slices = 5)

Validation_loader = torch.utils.data.DataLoader(validation_set, num_workers = 3, pin_memory = True, batch_sampler = validation_sampler)


device = torch.device('cuda')

encoder_2d = UNet_2D(2)#.cuda()
encoder_3d = UNet_3D(2)#.cuda()
discriminator_1 = Discriminator(1024, 2, mode = '2D')#.cuda()
discriminator_2 = Discriminator(1024, 2, mode = '3D')#.cuda()
decoder = Decoder_2D(1024, 4)#.cuda() #2048
#decoder = Decoder_3D_red(512, 4).cuda()
#crf = CRF(2)

weight_save_path = ['D:\\brain_tumor_segmentation\\weight_saves\\experiment_8\\initial_training',
                    'D:\\brain_tumor_segmentation\\weight_saves\\experiment_8\\regularization',
                    'D:\\brain_tumor_segmentation\\weight_saves\\experiment_8\\combined',
                    'D:\\brain_tumor_segmentation\\weight_saves\\experiment_8\\crf']

record_save_path = ['D:\\brain_tumor_segmentation\\record_saves\\experiment_8\\initial_training.txt',
                    'D:\\brain_tumor_segmentation\\record_saves\\experiment_8\\regularization.txt',
                    'D:\\brain_tumor_segmentation\\record_saves\\experiment_8\\combined.txt',
                    'D:\\brain_tumor_segmentation\\record_saves\\experiment_8\\crf.txt']

trainer = Run_Model(weight_save_path, record_save_path, encoder_2d, encoder_3d, decoder, discriminator_1, discriminator_2)

# if __name__ == '__main__':
#     trainer.train_loop_mixed(60, base_lr, Train_loader, Validation_loader)
    # trainer.Regularization_Loop(5, base_lr, Train_loader, Validation_loader)
    # trainer.Combined_loop(20, base_lr, Train_loader, Validation_loader)


test_path = 'D:\\brain_tumor_segmentation\\rough_4\\Testing.npy'
test_data = np.load(test_path, allow_pickle = True) #total_data[0:data_len]#[data_len-20:data_len]
test_set = Prepare_test_dataset(test_data)
Test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, pin_memory = True)
trainer.testing_whole_samples(Test_loader, 5)


