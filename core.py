import torch
import torch.nn as nn
import random
import numpy as np
import os

class Run_Model():
    def __init__(self):
        print('..')
        
    
    def train_loop(self, encoder_2d, encoder_3d, decoder, discriminator1, discriminator2, num_epochs, record_save_path, weights_save_path, base_lr, init_train, regularization, comb_training):
        
        for seq in sequence:
            seq_epochs = sequence[seq]
        
            for seq_epochs in num_epochs:
                
                if seq == 'init':
                    optimizer = 
                    
                    decoder_out = Single_pass_initial(encoder_2d, encoder_3d, decoder)
                    
                    