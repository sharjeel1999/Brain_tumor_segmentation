import torch
import torch.nn as nn

def Single_pass_initial(encoder_2d, encoder_3d, decoder, input_1, input_2):
    out_2d = encoder_2d(input_1)
    out_3d = encoder_3d(input_2)
    
    print('2d out shape: ', out_2d.shape)
    print('3d out shape: ', out_3d.shape)
    
    combined_features = torch.cat((out_2d, out_3d), dim = 1)
    
    dec_out = decoder(combined_features)
    
    return dec_out

def Single_pass_regularization(encoder_2d, encoder_3d, discriminator1, discriminator2, input_1, input_2):
    out_2d = encoder_2d(input_1)
    out_3d = encoder_3d(input_2)
    
    disc_out_1 = discriminator1(out_2d)
    disc_out_2 = discriminator2(out_3d)
    
    return disc_out_1, disc_out_2

def Single_pass_complete(encoder_2d, encoder_3d, decoder, discriminator1, discriminator2, input_1, input_2):
    out_2d = encoder_2d(input_1)
    out_3d = encoder_3d(input_2)
    
    disc_out_1 = discriminator1(out_2d)
    disc_out_2 = discriminator2(out_3d)
    
    combined_features = torch.cat((out_2d, out_3d), dim = 1)
    
    dec_out = decoder(combined_features)
    
    return disc_out_1, disc_out_2, dec_out


    