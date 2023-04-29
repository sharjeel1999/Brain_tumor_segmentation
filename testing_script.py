import numpy as np
import torch
import os

from models import UNet, UNet_2D, UNet_3D, Discriminator, Decoder_2D, Decoder_3D, UNet_2D_red, UNet_3D_red, Decoder_3D_red, Dense_encoder_2d, Dense_encoder_3d, Dense_Decoder_2D, CRF
from dataset import Prepare_dataset, Prepare_test_dataset, Prepare_full_volume_dataset, Data_slicer, TwoStreamBatchSampler
#from core import Run_Model
from core_full_volume import Run_Model

def set_divisions(selected, total_data, sb):
    training = []
    validation = []
    testing = []
    
    # while True:
        # rint = random.randint(0, 35)
        # sb = rint
        # if sb not in selected:
        #     selected.append(sb)
        #     break
    
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


data_path = 'C:\\Users\\Sharjeel\\Desktop\\Brain_data_paths_array.npy'
total_data = np.load(data_path, allow_pickle = True)

batch = 8
epochs = 100
base_lr = 0.001

device = torch.device('cuda')

encoder_2d = UNet_2D(2)#.cuda()
encoder_3d = UNet_3D(2)#.cuda()
discriminator_1 = Discriminator(1024, 2, mode = '2D')#.cuda()
discriminator_2 = Discriminator(1024, 2, mode = '3D')#.cuda()
decoder = Decoder_2D(1024, 4)#.cuda() #2048

f = 6
save_encoder12 = 'Encoder2D' + str(f) + '.pth'
save_encoder22 = 'Encoder3D' + str(f) + '.pth'
save_decoder2 = 'Decoder' + str(f) + '.pth'

folder_path = 'D:\\brain_tumor_segmentation\\weight_saves\\experiment_9_crossfolds_7slices\\fold0\\initial_training'

encoder_2d.load_state_dict(torch.load(os.path.join(folder_path, save_encoder12)))
encoder_3d.load_state_dict(torch.load(os.path.join(folder_path, save_encoder22)))
decoder.load_state_dict(torch.load(os.path.join(folder_path, save_decoder2)))

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

selected = []
train_data, validation_data, testing_data, selected, sb = set_divisions(selected, total_data, f)
testing_data = testing_data[19:20]

# test_path = 'D:\\brain_tumor_segmentation\\rough_4\\Testing.npy'
# test_data = np.load(test_path, allow_pickle = True) #total_data[0:data_len]#[data_len-20:data_len]
test_set = Prepare_test_dataset(testing_data)
Test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, pin_memory = True)
trainer.testing_whole_samples(Test_loader, 7, f)


