import numpy as np
import torch

from models import UNet, UNet_2D, UNet_3D, Discriminator
from dataset import Prepare_dataset

def Sequencer(init_train, regularizer, final_training, num_epochs):
    sequence = {}
    
    if init_train == True:
        sequence['init'] = int(num_epochs/2)
        
    if final_training == True:
        sequence['regularization'] = int(num_epochs/2)
        
    if final_training == True:
        sequence['combined'] = num_epochs
    
    return sequence

data_path = 'D:\\brain_tumor_segmentation\\rough\\Brain_data_paths_array.npy'
total_data = np.load(data_path, allow_pickle = True)

data_len = len(total_data)
print('Data len: ', data_len)
print('Train len: ', data_len - 15)

train_data = total_data[0:data_len-15]
validation_data = total_data[data_len-15:data_len]

train_set = Prepare_dataset(train_data, slices = 5)
validation_set = Prepare_dataset(validation_data, slices = 5)

batch = 3
epochs = 100
base_lr = 0.001

Train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch, shuffle = True)
Validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch, shuffle = True)

sequence = Sequencer(init_train = True, regularizer = True, final_training = True, num_epochs = epochs)

device = torch.device('cuda')

encoder_2d = UNet_2D(10).cuda()
encoder_3d = UNet_3D(1).cuda()
discriminator_1 = Discriminator().cuda()
discriminator_2 = Discriminator().cuda()
decoder = 0

# for seq in sequence:
#     seq_epochs = sequence[seq]
    
#     if seq == 'init':
#         optimizer = torch.optim.AdamW(list(encoder_2d.parameters()) + list(encoder_3d.parameters()) + list(decoder.parameters()), lr = base_lr, weight_decay = 1e-5)
        
#     if seq == 'regularization':
#         optimizer = torch.optim.AdamW(list(encoder_2d.parameters()) + list(encoder_3d.parameters()) + list(discriminator_1.parameters()) + list(discriminator_2.parameters()), lr = base_lr, weight_decay = 1e-5)

#     if seq == 'combined':
#         optimizer = torch.optim.AdamW(list(encoder_2d.parameters()) + list(encoder_3d.parameters()) + list(discriminator_1.parameters()) + list(discriminator_2.parameters()) + list(decoder.parameters()), lr = base_lr, weight_decay = 1e-5)




for sample in Train_loader:
    input1, input2, gt_masks = sample
    input1, input2 = input1.cuda(), input2.cuda()
    
    input2 = torch.unsqueeze((input2), dim = 1)
    
    print('input1 shape: ', input1.shape)
    print('input2 shape: ', input2.shape)
    print('gt masks shape: ', gt_masks.shape)
    
    out1 = encoder_2d(input1.float())
    out2 = encoder_3d(input2.float())
    
    print('output1 shape: ', out1.shape)
    print('output2 shape: ', out2.shape)