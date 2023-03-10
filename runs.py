import torch
import torch.nn as nn
from utils import loss_segmentation, loss_detection, dice_coeff

def Single_pass_initial(encoder_2d, encoder_3d, decoder, input_1, input_2, gt_mask):
    out_2d = encoder_2d(input_1)
    out_3d = encoder_3d(input_2)
    
    print('2d out shape: ', out_2d.shape)
    print('3d out shape: ', out_3d.shape)
    
    combined_features = torch.cat((out_2d, out_3d), dim = 1)
    
    dec_out = decoder(combined_features)
    
    loss, dsc, iou = loss_segmentation(dec_out, gt_mask)
    
    metrics = {}
    metrics['dice'] = dsc
    metrics['iou'] = iou
    
    return loss, metrics

def Single_pass_regularization(encoder_2d, encoder_3d, discriminator1, discriminator2, input_1, input_2, gt1, gt2):
    out_2d = encoder_2d(input_1)
    out_3d = encoder_3d(input_2)
    
    disc_out_1 = discriminator1(out_2d)
    disc_out_2 = discriminator2(out_3d)
    
    det_loss1 = loss_detection(disc_out_1, gt1)
    det_loss2 = loss_detection(disc_out_2, gt2)
    
    loss = det_loss1 + det_loss2
    
    return loss

def Single_pass_complete(encoder_2d, encoder_3d, decoder, discriminator1, discriminator2, input_1, input_2, gt_mask, gt1, gt2):
    out_2d = encoder_2d(input_1)
    out_3d = encoder_3d(input_2)
    
    disc_out_1 = discriminator1(out_2d)
    disc_out_2 = discriminator2(out_3d)
    
    combined_features = torch.cat((out_2d, out_3d), dim = 1)
    
    dec_out = decoder(combined_features)
    
    seg_loss, dsc, iou = loss_segmentation(dec_out, gt_mask)
    det_loss1 = loss_detection(disc_out_1, gt1)
    det_loss2 = loss_detection(disc_out_2, gt2)
    
    loss = seg_loss + det_loss1 + det_loss2
    
    return loss


    