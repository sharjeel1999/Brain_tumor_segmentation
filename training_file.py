import numpy as np
import torch

from models import UNet, UNet_2D, UNet_3D, Discriminator, Decoder_2D
from dataset import Prepare_dataset
from core import Run_Model

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
#data_path = 'C:\\Users\\Sharjeel\\Desktop\\Brain_data_paths_array.npy'
total_data = np.load(data_path, allow_pickle = True)

data_len = len(total_data)
print('Data len: ', data_len)
print('Train len: ', data_len - 15)

train_data = total_data[0:data_len-35]
validation_data = total_data[data_len-35:data_len-20]

train_set = Prepare_dataset(train_data, slices = 5)
validation_set = Prepare_dataset(validation_data, slices = 5)

batch = 6
epochs = 100
base_lr = 0.001

Train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch, shuffle = True, num_workers = 3, pin_memory = True)
Validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch, shuffle = True, num_workers = 3, pin_memory = True)


device = torch.device('cuda')

encoder_2d = UNet_2D(10).cuda()
encoder_3d = UNet_3D(1).cuda()
discriminator_1 = Discriminator(1024, 2).cuda()
discriminator_2 = Discriminator(1024, 2).cuda()
decoder = Decoder_2D(2048, 4).cuda()

weight_save_path = ['D:\\brain_tumor_segmentation\\weight_saves\\experiment_1\\initial_training',
                    'D:\\brain_tumor_segmentation\\weight_saves\\experiment_1\\regularization',
                    'D:\\brain_tumor_segmentation\\weight_saves\\experiment_1\\combined']

record_save_path = ['D:\\brain_tumor_segmentation\\record_saves\\experiment_1\\initial_training.txt',
                    'D:\\brain_tumor_segmentation\\record_saves\\experiment_1\\regularization.txt',
                    'D:\\brain_tumor_segmentation\\record_saves\\experiment_1\\combined.txt']

trainer = Run_Model(weight_save_path, record_save_path, encoder_2d, encoder_3d, decoder, discriminator_1, discriminator_2)

if __name__ == '__main__':
    #trainer.train_loop(15, base_lr, Train_loader, Validation_loader)
    #trainer.Regularization_Loop(10, base_lr, Train_loader, Validation_loader)
    trainer.Combined_loop(100, base_lr, Train_loader, Validation_loader)


# sequence = Sequencer(init_train = True, regularizer = True, final_training = True, num_epochs = epochs)

# for seq in sequence:
#     seq_epochs = sequence[seq]
    
#     if seq == 'init':
#         trainer.train_loop(50, base_lr, Train_loader, Validation_loader)
        
#     if seq == 'regularization':
#         trainer.Regularization_Loop(50, base_lr, Train_loader, Validation_loader)

#     if seq == 'combined':
#         trainer.Combined_loop(100, base_lr, Train_loader, Validation_loader)

# optimizer = torch.optim.AdamW(list(encoder_2d.parameters()) + list(encoder_3d.parameters()) + list(decoder.parameters()), lr = base_lr, weight_decay = 1e-5)

# CUDA_LAUNCH_BLOCKING=1

# from utils import loss_segmentation, loss_detection, dice_coeff

# for sample in Validation_loader:
#     input1, input2, gt_masks = sample
#     input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
    
#     optimizer.zero_grad()
#     input2 = torch.unsqueeze((input2), dim = 1)
    
#     print('input1 shape: ', input1.shape, input1.get_device())
#     print('input2 shape: ', input2.shape, input2.get_device())
#     print('gt masks shape: ', gt_masks.shape, gt_masks.get_device())
    
#     out1 = encoder_2d(input1.float())
#     out2 = encoder_3d(input2.float())
#     print('output1 shape: ', out1.shape, out1.get_device())
#     print('output2 shape: ', out2.shape, out2.get_device())
    
#     #disc1_out = discriminator_1(out1)
#     #disc2_out = discriminator_2(out2)
#     #print('discriminator 1 output: ', disc1_out.shape, disc1_out.get_device())
#     #print('discriminator 2 output: ', disc2_out.shape, disc2_out.get_device())
    
#     combined_out = torch.cat((out1, out2), dim = 1)
#     decoder_out = decoder(combined_out)
#     print('decoder out shape: ', decoder_out.shape, decoder_out.get_device())
    
        

#     gt_masks = torch.squeeze(gt_masks, dim = 1)
    
    
#     print('gt mask: ', gt_masks.shape, torch.unique(gt_masks))
#     loss, dsc, iou = loss_segmentation(decoder_out, gt_masks.long())
    
#     loss.backward()
#     optimizer.step()
    
    
    
    