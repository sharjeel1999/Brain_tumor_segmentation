import numpy as np
import torch

from models import UNet, UNet_2D, UNet_3D, Discriminator, Decoder_2D, Decoder_3D, UNet_2D_red, UNet_3D_red, Decoder_3D_red, Dense_encoder_2d, Dense_encoder_3d, Dense_Decoder_2D, CRF
from dataset import Prepare_dataset, Prepare_test_dataset, Prepare_full_volume_dataset, Data_slicer, TwoStreamBatchSampler
#from core import Run_Model
from core_full_volume import Run_Model

import random
from sklearn.model_selection import KFold

def set_divisions(selected, total_data, fold):
    training = []
    validation = []
    testing = []
    prev_inds = [6, 16, 20, 1, 23, 24]
    
    while True:
        rint = random.randint(0, 35)
        sb = prev_inds[fold] #rint
        if sb not in selected:
            selected.append(sb)
            break
    
    for sf in range(35):
        if sf == sb:
            testing = total_data[sf*35 : sf*35 + 20]
            validation = total_data[sf*35 + 20 : (sf+1)*35]
        else:
            k = total_data[sf*35:(sf+1)*35]
            for s in k:
                a1, a2, a3, a4, a5 = s
                training.append([a1, a2, a3, a4, a5])
            #training.append([total_data[sf*35:(sf+1)*35]])
    
    k = total_data[1225:1251]
    for s in k:
        a1, a2, a3, a4, a5 = s
        training.append([a1, a2, a3, a4, a5])
    training = np.array(training)
    
    return training, validation, testing, selected, sb

batch = 8
epochs = 100
base_lr = 0.001

# data_path = 'D:\\brain_tumor_segmentation\\rough_4\\Training_val.npy'
data_path = 'C:\\Users\\Sharjeel\\Desktop\\Brain_data_paths_array.npy'
total_data = np.load(data_path, allow_pickle = True)
print('total data shape: ', total_data.shape)

device = torch.device('cuda')

encoder_2d = UNet_2D(2)#.cuda()
encoder_3d = UNet_3D(2)#.cuda()
discriminator_1 = Discriminator(1024, 2, mode = '2D')#.cuda()
discriminator_2 = Discriminator(1024, 2, mode = '3D')#.cuda()
decoder = Decoder_2D(1024, 4)#.cuda() #2048

k = 5
splits = KFold(n_splits=k, shuffle=True, random_state=42)

selected = []

if __name__ == '__main__':
    for fold in range(6):
        
        train_data, validation_data, testing_data, selected, sb = set_divisions(selected, total_data, fold)
        
        print('train shape: ', train_data.shape)
        print('val shape: ', validation_data.shape)
        # train_data = train_data[0:10]
        # validation_data = validation_data[0:5]
        # testing_data = testing_data[0:2]
        
        ''' Train data prep '''
        train_slicer = Data_slicer(train_data, slices = 7)
        primary_indices, secondary_indices = train_slicer.get_inds()
        
        train_sampler = TwoStreamBatchSampler(primary_indices, secondary_indices, batch, 1)
        train_set = Prepare_dataset(train_data, train_slicer.get_data(), slices = 7)
        
        Train_loader = torch.utils.data.DataLoader(train_set, num_workers = 3, pin_memory = True, batch_sampler = train_sampler)
        
        '''validation data prep '''
        validation_slicer = Data_slicer(validation_data, slices = 7)
        val_primary_indices, val_secondary_indices = validation_slicer.get_inds()
        
        validation_sampler = TwoStreamBatchSampler(val_primary_indices, val_secondary_indices, batch, 1)
        validation_set = Prepare_dataset(validation_data, validation_slicer.get_data(), slices = 7)
        
        Validation_loader = torch.utils.data.DataLoader(validation_set, num_workers = 3, pin_memory = True, batch_sampler = validation_sampler)
        
        
        weight_save_path = ['D:\\brain_tumor_segmentation\\weight_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\initial_training',
                            'D:\\brain_tumor_segmentation\\weight_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\regularization',
                            'D:\\brain_tumor_segmentation\\weight_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\combined',
                            'D:\\brain_tumor_segmentation\\weight_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\crf']
        
        record_save_path = ['D:\\brain_tumor_segmentation\\record_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\initial_training.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\regularization.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\combined.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\experiment_9_crossfolds_7slices\\fold' + str(fold) +'\\testing.txt']
        
        
        trainer = Run_Model(weight_save_path, record_save_path, encoder_2d, encoder_3d, decoder, discriminator_1, discriminator_2)
        
        # trainer.train_loop_mixed(10, base_lr, Train_loader, Validation_loader, sb)
        # trainer.Regularization_Loop(5, base_lr, Train_loader, Validation_loader)
        trainer.Combined_loop(20, base_lr, Train_loader, Validation_loader, sb)
        
        test_set = Prepare_test_dataset(testing_data)
        Test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, pin_memory = True)
        trainer.testing_whole_samples(Test_loader, 7, sb)
    

