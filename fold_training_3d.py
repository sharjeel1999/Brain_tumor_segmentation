import numpy as np
import torch

from models import UNet, Encoder_2d, Encoder_3d, Discriminator, Decoder_2D, Decoder_3D, UNet_2D_red, UNet_3D_red, Decoder_3D_red, Dense_encoder_2d, Dense_encoder_3d, Dense_Decoder_2D, Decoder_2D_mod, ResidualUNet3D, nnUNet, nnUNet_2D
from dataset import Prepare_dataset, Prepare_test_dataset, Prepare_full_volume_dataset, Data_slicer, TwoStreamBatchSampler
#from core import Run_Model
from core_full_volume import Run_Model
# from core_missing import Run_Model

import random
from sklearn.model_selection import KFold

def set_divisions(selected, total_data, f):
    training = []
    validation = []
    testing = []
    prev = [6, 16, 20, 1, 23, 24]
    
    while True:
        rint = random.randint(0, 5)
        sb = f#prev[f] #rint
        if sb not in selected:
            selected.append(sb)
            break
    
    for sf in range(5):
        if sf == sb:
            testing = total_data[sf*250 : sf*250 + 250] # for 2018 = 57, for 2019 = 67, for 2020 = 73 for 2021 = 250
            validation = total_data[sf*250 : sf*250 + 250]
        else:
            k = total_data[sf*250:(sf+1)*250]
            for s in k:
                a1, a2, a3, a4, a5 = s
                training.append([a1, a2, a3, a4, a5])
            #training.append([total_data[sf*35:(sf+1)*35]])
    
    # k = total_data[1225:1251]
    # for s in k:
    #     a1, a2, a3, a4, a5 = s
    #     training.append([a1, a2, a3, a4, a5])
    training = np.array(training)
    
    return training, validation, testing, selected, sb

def combine_datasets(brats_18, brats_19, brats_20):
    combined_data = []
    
    for sample in brats_18:
        a1, a2, a3, a4, a5 = sample
        combined_data.append([a1, a2, a3, a4, a5])
        
    for sample in brats_19:
        a1, a2, a3, a4, a5 = sample
        combined_data.append([a1, a2, a3, a4, a5])
        
    for sample in brats_20:
        a1, a2, a3, a4, a5 = sample
        combined_data.append([a1, a2, a3, a4, a5])
        
    return np.array(combined_data)

# def set_divisions(selected, total_data, fold):
#     training = []
#     validation = []
#     testing = []
#     prev_inds = [6, 16, 20, 1, 23, 24]
#     while True:
#         rint = random.randint(0, 8) # 0 - 35
#         sb = 6 #prev_inds[fold]
#         if sb not in selected:
#             selected.append(sb)
#             break
    
#     for sf in range(35): # 35
#         if sf == sb:
#             testing = total_data[sf*35 : sf*35 + 20]
#             validation = total_data[sf*35 + 20 : (sf+1)*35]
#         else:
#             k = total_data[sf*35:(sf+1)*35]
#             for s in k:
#                 a1, a2, a3, a4, a5 = s
#                 training.append([a1, a2, a3, a4, a5])
#             #training.append([total_data[sf*35:(sf+1)*35]])
    
#     k = total_data[1225:1251]
#     # k = total_data[280:285]
#     for s in k:
#         a1, a2, a3, a4, a5 = s
#         training.append([a1, a2, a3, a4, a5])
#     training = np.array(training)
    
#     return training, validation, testing, selected, sb

# def set_divisions(selected, total_data, fold):
#     training = []
#     validation = []
#     testing = []
#     # prev_inds = [6, 16, 20, 1, 23, 24]
#     # while True:
#     #     rint = random.randint(0, 8) # 0 - 35
#     #     sb = 6 #prev_inds[fold]
#     #     if sb not in selected:
#     #         selected.append(sb)
#     #         break
#     sb = fold
#     for sf in range(5): # 35
#         if sf == sb:
#             testing = total_data[sf*250 : (sf+1)*250]
#             validation = total_data[sf*250 : (sf+1)*250]
#         else:
#             k = total_data[sf*250:(sf+1)*250]
#             for s in k:
#                 a1, a2, a3, a4, a5 = s
#                 training.append([a1, a2, a3, a4, a5])
#             #training.append([total_data[sf*35:(sf+1)*35]])
    
#     # k = total_data[1225:1251]
#     # k = total_data[280:285]
#     for s in k:
#         a1, a2, a3, a4, a5 = s
#         training.append([a1, a2, a3, a4, a5])
#     training = np.array(training)
    
#     return training, validation, testing, selected, sb

batch = 8
epochs = 100
base_lr = 0.0001 #1e-4

# data_path = 'D:\\brain_tumor_segmentation\\rough_4_modified_3d\\Brain_data_paths_array.npy'
# data_path = 'D:\\brain_tumor_segmentation\\rough_4_modified_3d\\Brain_data_paths_array.npy'
# total_data_20 = np.load(data_path, allow_pickle = True)

# data_path = 'D:\\brain_tumor_segmentation\\rough_4_modified_3d\\Brain_data_paths_array_2018.npy'
# total_data = np.load(data_path, allow_pickle = True)
# print('total data shape: ', total_data.shape)
# np.random.shuffle(total_data)
# np.save('shuffled_2018_x.npy', total_data) #2 is best, x is best

total_data_18 = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2018_x.npy', allow_pickle = True)
total_data_19 = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2019_1.npy', allow_pickle = True)
total_data_20 = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2020.npy', allow_pickle = True)
total_data = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2021.npy', allow_pickle = True)

combined_data = combine_datasets(total_data_18, total_data_19, total_data_20)
print('Combined data shape: ', combined_data.shape)

device = torch.device('cuda')

from torch.nn import init
from torch import nn

all = ['initialize_weights']
# torch.autograd.set_detect_anomaly(True)

def initialize_weights(model):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)        
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
            
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)

encoder_2d = Encoder_2d(4, inter_channels = 32)#.cuda()
encoder_3d = Encoder_3d(4, inter_channels = 32)#.cuda()
discriminator_1 = Discriminator(1024, 2, mode = '2D')#.cuda()
discriminator_2 = Discriminator(1024, 2, mode = '3D')#.cuda()
decoder = Decoder_2D(512, inter_channels = 32, n_classes = 4)#.cuda() #2048
# unet_3D = ResidualUNet3D(in_channels = 12, out_channels = 4, final_sigmoid = False, num_groups = 8, f_maps = 32, num_levels = 5, layer_order = 'gcl')
unet_3D = nnUNet(in_channels = 4, num_classes = 4, inter_channels = 32)
# print(unet_3D)

encoder_2d.apply(initialize_weights)
encoder_3d.apply(initialize_weights)
discriminator_1.apply(initialize_weights)
discriminator_2.apply(initialize_weights)
decoder.apply(initialize_weights)
unet_3D.apply(initialize_weights)

k = 5
splits = KFold(n_splits=k, shuffle=True, random_state=42)

selected = []

if __name__ == '__main__':
    for fold in range(0, 1):
        print('selected fold: ', fold)
        train_data, validation_data, testing_data, selected, sb = set_divisions(selected, total_data, fold)
        
        print('train shape: ', train_data.shape)
        print('val shape: ', validation_data.shape)
        # train_data = train_data[0:1]
        # validation_data = validation_data[0:2]
        # testing_data = testing_data[0:2]
        # combined_data = combined_data[0:2]
        
        ''' Train data prep '''
        train_set = Prepare_test_dataset(train_data)
        Train_loader = torch.utils.data.DataLoader(train_set, pin_memory = True, batch_size = 1)
        
        '''validation data prep '''
        validation_set = Prepare_test_dataset(validation_data)
        Validation_loader = torch.utils.data.DataLoader(validation_set, pin_memory = True, batch_size = 1)
        
        
        weight_save_path = ['D:\\brain_tumor_segmentation\\weight_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\initial_training',
                            'D:\\brain_tumor_segmentation\\weight_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\regularization',
                            'D:\\brain_tumor_segmentation\\weight_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\combined',
                            'D:\\brain_tumor_segmentation\\weight_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\crf']
        
        record_save_path = ['D:\\brain_tumor_segmentation\\record_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\initial_training.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\regularization.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\combined.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\z_model_using_prev_weights_21\\fold' + str(fold) +'\\testing.txt']
        
        
        trainer = Run_Model(weight_save_path, record_save_path, encoder_2d, encoder_3d, decoder, unet_3D, discriminator_1, discriminator_2)
        
        trainer.train_loop(1, base_lr, Train_loader, Validation_loader, mode = 'validation')#, sb)
        # trainer.Regularization_Loop(5, base_lr, Train_loader, Validation_loader)
        # trainer.Combined_loop(20, base_lr, Train_loader, Validation_loader, sb)
        
        # test_set = Prepare_test_dataset(testing_data)
        # Test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, pin_memory = True)
        # trainer.testing_whole_samples(Test_loader, 7, sb)
    

