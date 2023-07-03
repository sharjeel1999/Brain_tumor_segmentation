import torch
import torch.nn as nn
import random
import numpy as np
import os
from utils import loss_segmentation, loss_detection, dice_coeff, class_dice, test_scores_3d, dice_3d, hausdorf_distance, other_metrics, prec_rec, loss_segmentation_hem
from torch.autograd import Variable
import time
from tqdm import tqdm

import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from torchvision import transforms

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Connection(nn.Module):
    def __init__(self):
        super(Connection, self).__init__()
        
        self.connection_layer = nn.Conv3d(8, 32, kernel_size = 3, stride = 1, padding = 1).cuda()
        
    def forward(self, x):
        return self.connection_layer(x)

def deep_supervision_loss_2d(full_input, full_gt):
    # print('loss input shape: ', full_input[0].shape)
    # print('gt input shape: ', full_gt.shape)
    
    loss3d_1, _, _, _, _, _, _ = loss_segmentation(full_input[0], full_gt.long().cuda())
    
    by2_gt = full_gt[:, 0:full_gt.shape[1]:2**1, 0:full_gt.shape[2]:2**1]
    loss3d_2, _, _, _, _, _, _ = loss_segmentation(full_input[1], by2_gt.long().cuda())
    
    by4_gt = full_gt[:, 0:full_gt.shape[1]:2**2, 0:full_gt.shape[2]:2**2]
    loss3d_3, _, _, _, _, _, _ = loss_segmentation(full_input[2], by4_gt.long().cuda())
    
    by8_gt = full_gt[:, 0:full_gt.shape[1]:2**3, 0:full_gt.shape[2]:2**3]
    loss3d_4, _, _, _, _, _, _ = loss_segmentation(full_input[3], by8_gt.long().cuda())
    
    return loss3d_1*0.5 + loss3d_2*0.2 + loss3d_3*0.2 + loss3d_4*0.1

class Run_Model():
    def __init__(self, weight_save_path, record_save_path, encoder_2d, encoder_3d, decoder, unet_3D, discriminator1, discriminator2):
        self.weight_save_path = weight_save_path
        self.record_save_path = record_save_path
        self.encoder_2d = encoder_2d.cuda()
        self.encoder_3d = encoder_3d.cuda()
        self.decoder = decoder.cuda()
        self.unet_3D = unet_3D.cuda()
        self.discriminator1 = discriminator1#.cuda()
        self.discriminator2 = discriminator2#.cuda()
        
        self.transform = transforms.ToTensor()
    
    def Singl_pass_2D_only(self, input_1, gt_mask):
        gt_mask = torch.squeeze(gt_mask, dim = 1)
        
        dec_out = self.unet_3D(input_1)
        
        _, dsc, class_dsc, ious, class_iou, tc_score, whole_scores = loss_segmentation(dec_out[0], gt_mask)
        
        seg_loss = deep_supervision_loss_2d(dec_out, gt_mask)
        
        metrics = {}
        metrics['dice'] = dsc
        metrics['iou'] = ious
        metrics['class_dice'] = class_dsc
        metrics['class_iou'] = class_iou
        metrics['tc_scores'] = tc_score
        metrics['whole_scores'] = whole_scores
        
        # return seg_loss, metrics, dec_out
        return dec_out, seg_loss, metrics
    
    
    def Single_pass_initial(self, input_1, input_2, gt_mask):

        gt_mask = torch.squeeze(gt_mask, dim = 1)
        
        out_2d_ar = self.encoder_2d(input_1)
        out_3d_ar = self.encoder_3d(input_2)
        
        # print('out 2d unique: ', torch.unique(out_2d_ar[0]))
        # print('out 3d unique: ', torch.unique(out_3d_ar[0]))
        
        dec_out = self.decoder(out_2d_ar, out_3d_ar)
        
        return dec_out
    
    def Single_pass_regularization_second(self, input_1, input_2, optimizer_gen, optimizer_disc, mode):
        EPS = 1e-15
        Tensor = torch.cuda.FloatTensor
        #input_2 = torch.unsqueeze((input_2), dim = 1)
        
        
        if mode == 'train':
            self.encoder_2d.train()
            self.encoder_3d.train()
        else:
            self.encoder_2d.eval()
            self.encoder_3d.eval()
            
        self.discriminator1.eval()
        self.discriminator2.eval()
        
        out_2d_ar = self.encoder_2d(input_1)
        out_3d_ar = self.encoder_3d(input_2)
        out_2d = out_2d_ar[0]
        out_3d = out_3d_ar[0]
        
        disc_out_1_fakee = self.discriminator1(out_2d)
        disc_out_2_fakee = self.discriminator2(out_3d)
        
        
        ONES = Variable(Tensor(input_1.shape[0], 1).fill_(1.0), requires_grad=False).long()
        ZEROS = Variable(Tensor(input_1.shape[0], 1).fill_(0.0), requires_grad=False).long()
        ONES = torch.squeeze(ONES, dim = 1)
        ZEROS = torch.squeeze(ZEROS, dim = 1)
        
        det_loss1 = loss_detection(disc_out_1_fakee, ONES)
        det_loss2 = loss_detection(disc_out_2_fakee, ONES)
        
        tot_gen_loss = det_loss1 + det_loss2
        #print('tot gen loss: ', tot_gen_loss.item())
        if mode == 'train':
            optimizer_gen.zero_grad()
            tot_gen_loss.backward()
            optimizer_gen.step()
        
        ######################################################################################
        
        self.encoder_2d.eval()
        self.encoder_3d.eval()
        if mode == 'train':
            self.discriminator1.train()
            self.discriminator2.train()
        else:
            self.discriminator1.eval()
            self.discriminator2.eval()
        
        z = nn.Parameter(Tensor(np.random.normal(0, 1, (input_1.shape[0], out_2d.shape[1], out_2d.shape[2], out_2d.shape[3]))), requires_grad=False).cuda()
        
        disc_out_1_fake = self.discriminator1(out_2d.detach())
        disc_out_2_fake = self.discriminator2(out_3d.detach())
        
        disc_out_1 = self.discriminator1(z)
        disc_out_2 = self.discriminator2(z)
        
        det_loss1 = loss_detection(disc_out_1, ONES)
        det_loss2 = loss_detection(disc_out_2, ONES)
        det_loss3 = loss_detection(disc_out_1_fake, ZEROS)
        det_loss4 = loss_detection(disc_out_2_fake, ZEROS)
        
        tot_disc_loss = det_loss1 + det_loss2 + det_loss3 + det_loss4
        #print('tot disc loss: ', tot_disc_loss.item())
        if mode == 'train':
            optimizer_disc.zero_grad()
            tot_disc_loss.backward()
            optimizer_disc.step()
        
        
        disc_out_1 = torch.argmax(disc_out_1, dim = 1)
        disc_out_2 = torch.argmax(disc_out_2, dim = 1)
        disc_out_1_fake = torch.argmax(disc_out_1_fake, dim = 1)
        disc_out_2_fake = torch.argmax(disc_out_2_fake, dim = 1)
        
        
        acc1 = (sum(disc_out_1 == ONES).item())/disc_out_1.shape[0]
        acc2 = (sum(disc_out_2 == ONES).item())/disc_out_2.shape[0]
        acc3 = (sum(disc_out_1_fake == ZEROS).item())/disc_out_1_fake.shape[0]
        acc4 = (sum(disc_out_1_fake == ZEROS).item())/disc_out_1_fake.shape[0]
        acc = np.mean([acc1, acc2, acc3, acc4])
        
        return tot_gen_loss, tot_disc_loss, acc
    
    
    def Single_pass_complete(self, input_1, input_2, gt_mask, optimizer_gen, optimizer_disc, mode):
        EPS = 1e-15
        Tensor = torch.cuda.FloatTensor
        #input_2 = torch.unsqueeze((input_2), dim = 1)
        gt_mask = torch.squeeze(gt_mask, dim = 1)
        
        if mode == 'train':
            self.encoder_2d.train()
            self.encoder_3d.train()
            self.decoder.train()
        else:
            self.encoder_2d.eval()
            self.encoder_3d.eval()
            self.decoder.eval()
            
        self.discriminator1.eval()
        self.discriminator2.eval()
        
        out_2d_ar = self.encoder_2d(input_1)
        out_3d_ar = self.encoder_3d(input_2)
        out_2d = out_2d_ar[0]
        out_3d = out_3d_ar[0]
        
        #combined_features = torch.cat((out_2d, out_3d), dim = 1)
        
        dec_out = self.decoder(out_2d_ar, out_3d_ar)
        
        seg_loss, dsc, class_dsc, ious, class_iou, tc_score, whole_scores = loss_segmentation(dec_out, gt_mask)
        disc_out_1_fakee = self.discriminator1(out_2d)
        disc_out_2_fakee = self.discriminator2(out_3d)
        
        
        ONES = Variable(Tensor(input_1.shape[0], 1).fill_(1.0), requires_grad=False).long()
        ZEROS = Variable(Tensor(input_1.shape[0], 1).fill_(0.0), requires_grad=False).long()
        ONES = torch.squeeze(ONES, dim = 1)
        ZEROS = torch.squeeze(ZEROS, dim = 1)
        
        det_loss1 = loss_detection(disc_out_1_fakee, ONES)
        det_loss2 = loss_detection(disc_out_2_fakee, ONES)
        
        tot_gen_loss = 0.1*(det_loss1 + det_loss2) + 0.9*seg_loss
        
        if mode == 'train':
            optimizer_gen.zero_grad()
            tot_gen_loss.backward()
            optimizer_gen.step()
        
        ######################################################################################
        
        if mode == 'train':
            self.discriminator1.train()
            self.discriminator2.train()
        else:
            self.discriminator1.eval()
            self.discriminator2.eval()
        
        # z2d = nn.Parameter(Tensor(np.random.normal(0, 1, (input_1.shape[0], out_2d.shape[1], out_2d.shape[2], out_2d.shape[3]))), requires_grad=False).cuda()
        # z3d = nn.Parameter(Tensor(np.random.normal(0, 1, (input_1.shape[0], out_3d.shape[1], out_3d.shape[2], out_3d.shape[3], out_3d.shape[4]))), requires_grad=False).cuda()
        
        z2d = Tensor(np.zeros((input_1.shape[0], out_2d.shape[1], out_2d.shape[2], out_2d.shape[3])))
        z3d = Tensor(np.zeros((input_1.shape[0], out_3d.shape[1], out_3d.shape[2], out_3d.shape[3], out_3d.shape[4])))
        
        for k in range(input_1.shape[0]):
            z2d[k, :, :, :] = Tensor(np.zeros((out_2d.shape[1], out_2d.shape[2], out_2d.shape[3])))
            z3d[k, :, :, :, :] = Tensor(np.zeros((out_3d.shape[1], out_3d.shape[2], out_3d.shape[3], out_3d.shape[4])))
        
        z2d = nn.Parameter(z2d, requires_grad=True).cuda()
        z3d = nn.Parameter(z3d, requires_grad=True).cuda()
        
        # print('normalized 2d shape: ', z2d.shape)
        # print('normalized 3d shape: ', z3d.shape)
        
        disc_out_1_fake = self.discriminator1(out_2d.detach())
        disc_out_2_fake = self.discriminator2(out_3d.detach())
        
        disc_out_1 = self.discriminator1(z2d)
        disc_out_2 = self.discriminator2(z3d)
        
        det_loss1 = loss_detection(disc_out_1, ONES)
        det_loss2 = loss_detection(disc_out_2, ONES)
        det_loss3 = loss_detection(disc_out_1_fake, ZEROS)
        det_loss4 = loss_detection(disc_out_2_fake, ZEROS)
        
        tot_disc_loss = det_loss1 + det_loss2 + det_loss3 + det_loss4
        
        if mode == 'train':
            optimizer_disc.zero_grad()
            tot_disc_loss.backward()
            optimizer_disc.step()
        
        
        disc_out_1 = torch.argmax(disc_out_1, dim = 1)
        disc_out_2 = torch.argmax(disc_out_2, dim = 1)
        disc_out_1_fake = torch.argmax(disc_out_1_fake, dim = 1)
        disc_out_2_fake = torch.argmax(disc_out_2_fake, dim = 1)

        acc1 = (sum(disc_out_1 == ONES).item())/disc_out_1.shape[0]
        acc2 = (sum(disc_out_2 == ONES).item())/disc_out_2.shape[0]
        acc3 = (sum(disc_out_1_fake == ZEROS).item())/disc_out_1_fake.shape[0]
        acc4 = (sum(disc_out_1_fake == ZEROS).item())/disc_out_1_fake.shape[0]
        acc = np.mean([acc1, acc2, acc3, acc4])
        
        ########################################################################
        
        return tot_gen_loss, dsc, class_dsc, ious, class_iou, tc_score, whole_scores, dec_out
    
    def pre_process(self, in_slice):
        in_slice = ((in_slice/np.max(in_slice))*255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        cl = clahe.apply(in_slice)
        return cl
    
    def hist_match(self, source, template):

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)


        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)
    
    def hist_match_torch(self, source, template):
        # source = torch.from_numpy(source).cuda()
        # template = torch.from_numpy(template).cuda()
        
        oldshape = source.shape
        source = torch.ravel(source)
        template = torch.ravel(template)

        s_values, bin_idx, s_counts = torch.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = torch.unique(template, return_counts=True)
        
        s_quantiles = s_counts.cumsum(dim = 0).type(torch.float64)
        s_quantiles = s_quantiles.clone() / s_quantiles[-1]
        
        t_quantiles = t_counts.cumsum(dim = 0).type(torch.float64)
        t_quantiles = t_quantiles.clone() / t_quantiles[-1]

        interp_t_values = np.interp(s_quantiles.cpu().numpy(), t_quantiles.cpu().numpy(), t_values.cpu().numpy())

        return interp_t_values[bin_idx].reshape(oldshape)
    
    def dynamic_slicer(self, t1_volume, t1ce_volume, t2_volume, flair_volume, gt_mask_volume, slices, selected_ind, t1_temp, t1ce_temp, t2_temp, flair_temp):
        start = int(selected_ind - ((slices - 1)/2))
        end = int(selected_ind + ((slices - 1)/2) + 1)
        # print('start - end: ', selected_ind, start, end)
        
        max_depth = t1ce_volume.shape[3] - 1
        mm = slices // 2
        mms = (slices - mm)*2
        
        input1 = torch.zeros((t1ce_volume.shape[0], 4, 128, 128))
        input2 = torch.zeros((t1ce_volume.shape[0], 4, int(slices), 128, 128))
        gt_masks = torch.zeros((t1ce_volume.shape[0], 1, 128, 128))
        
        # temp = np.load('template.npy', allow_pickle = True)
        
        t1_volume, t1ce_volume, t2_volume, flair_volume = t1_volume.cuda(), t1ce_volume.cuda(), t2_volume.cuda(), flair_volume.cuda()
        
        ent = False
        for k, z in enumerate(range(start, end)):
            
            if z < 0:
                z = 0
            if z > max_depth:
                z = max_depth
            
            t1_slice = t1_volume[:, :, :, z]
            t1ce_slice = t1ce_volume[:, :, :, z]
            t2_slice = t2_volume[:, :, :, z]
            flair_slice = flair_volume[:, :, :, z]
            gt_mask_slice = gt_mask_volume[:, :, :, z]
            
            clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(1,1))
            
            t1_slice_eq = ((t1_slice[0, :, :]/np.max(t1_slice[0, :, :]))*255).astype(np.uint8)
            t1_slice_eq = clahe.apply(t1_slice_eq)
            
            t1ce_slice_eq = ((t1ce_slice[0, :, :]/np.max(t1ce_slice[0, :, :]))*255).astype(np.uint8)
            t1ce_slice_eq = clahe.apply(t1ce_slice_eq)
            
            t2_slice_eq = ((t2_slice[0, :, :]/np.max(t2_slice[0, :, :]))*255).astype(np.uint8)
            t2_slice_eq = clahe.apply(t2_slice_eq)
            
            flair_slice_eq = ((flair_slice[0, :, :]/np.max(flair_slice[0, :, :]))*255).astype(np.uint8)
            flair_slice_eq = clahe.apply(flair_slice_eq)
            

            dummy_tensor = torch.zeros_like(t1_slice)

            input2[:, 0, int(k), :, :] = torch.from_numpy(t1_slice_eq)
            input2[:, 1, int(k), :, :] = torch.from_numpy(t1ce_slice_eq)
            input2[:, 2, int(k), :, :] = torch.from_numpy(t2_slice_eq)
            input2[:, 3, int(k), :, :] = torch.from_numpy(flair_slice_eq)
            
            if z == selected_ind and ent == False:
                input1[:, 0, :, :] = t1_slice
                input1[:, 1, :, :] = t1ce_slice
                input1[:, 2, :, :] = t2_slice
                input1[:, 3, :, :] = flair_slice

                gt_masks[:, 0, :, :] = gt_mask_slice
                ent = True
        
        if input1.max() != 0:
            input1 = (input1 - input1.min()) / (input1.max() - input1.min())
        if input2.max() != 0:
            # print('max: ', input2.min(), input2.max())
            input2 = (input2) / (input2.max())
        
        # print('input 1: ', torch.unique(input1))
        if True in torch.isnan(input2):
            print('now')
        return input1, input2, gt_masks
    
    def deep_supervision_loss(self, full_input, full_gt):
        
        loss3d_1 = loss_segmentation_hem(full_input[0], full_gt.long().cuda())
        
        by2_gt = full_gt[:, 0:full_gt.shape[1]:2**1, 0:full_gt.shape[2]:2**1, 0:full_gt.shape[3]:2**1]
        loss3d_2 = loss_segmentation_hem(full_input[1], by2_gt.long().cuda())
        
        by4_gt = full_gt[:, 0:full_gt.shape[1]:2**2, 0:full_gt.shape[2]:2**2, 0:full_gt.shape[3]:2**2]
        loss3d_3 = loss_segmentation_hem(full_input[2], by4_gt.long().cuda())
        
        by8_gt = full_gt[:, 0:full_gt.shape[1]:2**3, 0:full_gt.shape[2]:2**3, 0:full_gt.shape[3]:2**3]
        loss3d_4 = loss_segmentation_hem(full_input[3], by8_gt.long().cuda())
        
        return loss3d_1*0.5 + loss3d_2*0.2 + loss3d_3*0.2 + loss3d_4*0.1


    def replace_first(self):
        self.unet_3D.encoder_module.initial_normal.conv = Identity()
        # self.unet_3D.encoders.basic_module.conv1 = Identity()
        # print(self.unet_3D)
        
    def train_loop_2D(self, num_epochs, base_lr, train_loader, val_loader, mode):
        
        dice_latch = 0
        slices = 7

        save_unet = 'UNet.pth'
        
        # self.unet_3D.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_unet)), strict=False)
        t1_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\t1_template.npy', allow_pickle = True)
        t1ce_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\t1ce_template.npy', allow_pickle = True)
        t2_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\t2_template.npy', allow_pickle = True)
        flair_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\flair_template.npy', allow_pickle = True)
        
        for epoch in range(0, num_epochs):
            train_loss = []
            train_dice = []
            train_iou = []
            train_class_dice = []
            train_class_iou = []
            train_tc_dice = []
            train_tc_iou = []
            train_whole_dice = []
            train_whole_iou = []

            self.unet_3D.train()

            # optimizer_2 = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()) + list(self.unet_3D.parameters()) + list(self.connection_layer.parameters()), lr = base_lr)#, weight_decay = 1e-7)
            optimizer_2 = torch.optim.Adam(self.unet_3D.parameters(), lr = base_lr)#, weight_decay = 0.00001)

            # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 4)
            scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min', patience = 4)
            
            if mode == 'train':
                for sample in tqdm(train_loader):
                    t1, t1ce, t2, flair, masks = sample
                    vol_depth = t1ce.shape[3]

                    for itter in range(0, vol_depth, 1):
    
                        input1, input2, gt_masks = self.dynamic_slicer(t1, t1ce, t2, flair, masks, slices, itter, t1_temp, t1ce_temp, t2_temp, flair_temp)
                        input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                        
                        pred_mask, seg_loss, metrics = self.Singl_pass_2D_only(input1.float(), gt_masks.long())

                        if len(torch.unique(gt_masks)) > 2:
                            optimizer_2.zero_grad()
                            seg_loss.backward()
                            optimizer_2.step()
                            train_loss.append(seg_loss.item())
                    
                            # print('iou: ', metrics['iou'])
                            train_iou.append(metrics['iou'].detach().cpu().numpy())
                            train_dice.append(metrics['dice'].detach().cpu().numpy())
                            train_class_dice.append(metrics['class_dice'].detach().cpu().numpy())
                            train_class_iou.append(metrics['class_iou'].detach().cpu().numpy())
            
                            train_tc_dice.append(metrics['tc_scores'][0].detach().cpu().numpy())
                            train_tc_iou.append(metrics['tc_scores'][1].detach().cpu().numpy())
                            train_whole_dice.append(metrics['whole_scores'][0].detach().cpu().numpy())
                            train_whole_iou.append(metrics['whole_scores'][1].detach().cpu().numpy())
                

                print('Epoch: ', epoch)
                print('Initial Train Loss: ', np.mean(train_loss))
                print('Initial Train Dice: ', np.mean(train_dice))
                print('Initial Train IoU: ', np.mean(train_iou))
                print('Initial Training NET, Edema, ET dice: ', np.mean(train_class_dice, axis = 0))
                print('Initial Training NET, Edema, ET iou: ', np.mean(train_class_iou, axis = 0))
                print('Initial Training TC dice: ', np.mean(train_tc_dice))
                print('Initial Training TC iou: ', np.mean(train_tc_iou))
                print('Initial Training whole dice: ', np.mean(train_whole_dice))
                print('Initial Training whole iou: ', np.mean(train_whole_iou))
            
            val_loss = []
            val_dice = []
            val_iou = []
            val_class_dice = []
            val_class_iou = []
            val_tc_dice = []
            val_tc_iou = []
            val_whole_dice = []
            val_whole_iou = []

            self.unet_3D.eval()

            for sample in val_loader:
                t1, t1ce, t2, flair, masks = sample

                vol_depth = t1ce.shape[3]
                with torch.no_grad():
                    for itter in range(0, vol_depth, 1):
    
                        input1, input2, gt_masks = self.dynamic_slicer(t1, t1ce, t2, flair, masks, slices, itter, t1_temp, t1ce_temp, t2_temp, flair_temp)
                        input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                        
                        pred_mask, seg_loss, metrics = self.Singl_pass_2D_only(input1.float(), gt_masks.long())

                        if len(torch.unique(gt_masks)) > 2:
                            val_loss.append(seg_loss.item())
                    
                            val_dice.append(metrics['iou'].detach().cpu().numpy())
                            val_iou.append(metrics['dice'].detach().cpu().numpy())
                            val_class_dice.append(metrics['class_dice'].detach().cpu().numpy())
                            val_class_iou.append(metrics['class_iou'].detach().cpu().numpy())
            
                            val_tc_dice.append(metrics['tc_scores'][0].detach().cpu().numpy())
                            val_tc_iou.append(metrics['tc_scores'][1].detach().cpu().numpy())
                            val_whole_dice.append(metrics['whole_scores'][0].detach().cpu().numpy())
                            val_whole_iou.append(metrics['whole_scores'][1].detach().cpu().numpy())
                
            print('Initial Validation Loss: ', np.mean(val_loss))
            print('Initial Validation Dice: ', np.mean(val_dice))
            print('Initial Validation IoU: ', np.mean(val_iou))
            print('Initial Validation NET, Edema, ET dice: ', np.mean(val_class_dice, axis = 0))
            print('Initial Validation NET, Edema, ET iou: ', np.mean(val_class_iou, axis = 0))
            print('Initial Validation TC dice: ', np.mean(val_tc_dice))
            print('Initial Validation TC iou: ', np.mean(val_tc_iou))
            print('Initial Validation whole dice: ', np.mean(val_whole_dice))
            print('Initial Validation whole iou: ', np.mean(val_whole_iou))
            print('\n')
            
            if dice_latch < np.mean(val_dice):
                save_unet_3d2 = 'UNet.pth'
                torch.save(self.unet_3D.state_dict(), os.path.join(self.weight_save_path[0], save_unet_3d2))

                dice_latch = np.mean(val_dice)
            
            with open(self.record_save_path[0], 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(train_loss)} Train Dice: {np.mean(train_dice)} Train IoU: {np.mean(train_iou)}')
                f.write('\n')
                f.write(f'Train NET, Edema, ET dice: {np.mean(train_class_dice, axis = 0)} Train NET, Edema, ET iou: {np.mean(train_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Train TC dice: {np.mean(train_tc_dice)} Train TC iou: {np.mean(train_tc_iou)}')
                f.write('\n')
                f.write(f'Train whole dice: {np.mean(train_whole_dice)} Train whole iou: {np.mean(train_whole_iou)}')
                f.write('\n')
                
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation Dice: {np.mean(val_dice)} Validation IoU: {np.mean(val_iou)}')
                f.write('\n')
                f.write(f'Validation NET, Edema, ET dice: {np.mean(val_class_dice, axis = 0)} Validation NET, Edema, ET iou: {np.mean(val_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Validation TC dice: {np.mean(val_tc_dice)} Validation TC iou: {np.mean(val_tc_iou)}')
                f.write('\n')
                f.write(f'Validation whole dice: {np.mean(val_whole_dice)} Validation whole iou: {np.mean(val_whole_iou)}')
                f.write('\n')
                f.write('\n')
        

    def train_loop(self, num_epochs, base_lr, train_loader, val_loader, mode):
        
        dice_latch = 0
        slices = 7
        save_encoder12 = 'Encoder2D.pth'
        save_encoder22 = 'Encoder3D.pth'
        save_decoder2 = 'Decoder.pth'
        save_unet = 'UNet.pth'
        
        unet_2d = 'UNet_2D.pth'
        unet_3d = 'UNet_3D.pth'
        
        self.replace_first()
        
        self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder12)), strict=False)
        self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder22)), strict=False)
        self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_decoder2)), strict=False)
        
        self.unet_3D.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_unet)), strict=False)
        
        # self.replace_first()
        
        self.connection_layer = nn.Conv3d(8, 32, kernel_size = 3, stride = 1, padding = 1).cuda()
        
        self.connection_layer.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], 'Connection.pth')))
        
        t1_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\t1_template.npy', allow_pickle = True)
        t1ce_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\t1ce_template.npy', allow_pickle = True)
        t2_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\t2_template.npy', allow_pickle = True)
        flair_temp = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\flair_template.npy', allow_pickle = True)
        
        for epoch in range(0, num_epochs):
            train_loss = []
            train_dice = []
            train_iou = []
            train_class_dice = []
            train_class_iou = []
            train_tc_dice = []
            train_tc_iou = []
            train_whole_dice = []
            train_whole_iou = []
            
            train_ET_HD = []
            train_TC_HD = []
            train_whole_HD = []
            
            self.encoder_2d.train()
            self.encoder_3d.train()
            self.decoder.train()
            self.unet_3D.train()
            self.connection_layer.train()
            
            # + list(self.unet_3D.parameters())
            optimizer_2 = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()) + list(self.unet_3D.parameters()) + list(self.connection_layer.parameters()), lr = base_lr)#, weight_decay = 1e-7)
            # optimizer_2 = torch.optim.Adam(self.unet_3D.parameters(), lr = base_lr)#, weight_decay = 0.00001)

            # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 4)
            scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min', patience = 3)
            
            if mode == 'train':
                for sample in tqdm(train_loader):
                    t1, t1ce, t2, flair, masks = sample
                    # print('shapes: ', t1.shape, t1ce.shape)
                    
                    dummy_vol_tensor = torch.zeros_like(t1)
                    
                    input_3d = torch.zeros(t1.shape[0], 4, t1.shape[1], t1.shape[2], t1.shape[3])
                    input_3d[:, 0, :, :, :] = t1
                    input_3d[:, 1, :, :, :] = t1ce
                    input_3d[:, 2, :, :, :] = t2
                    input_3d[:, 3, :, :, :] = flair
                    input_3d = torch.permute(input_3d, (0, 1, 4, 2, 3)).cuda()
                    masks_3d = torch.permute(masks, (0, 3, 1, 2)).cpu()
                    input_3d = (input_3d - input_3d.min()) / (input_3d.max() - input_3d.min())
    
                    vol_depth = t1ce.shape[3]
                    
                    prediction_volume = None
                    gt_volume = None
                    loss1 = 0
                    for itter in range(0, vol_depth, 1):
    
                        input1, input2, gt_masks = self.dynamic_slicer(t1, t1ce, t2, flair, masks, slices, itter, t1_temp, t1ce_temp, t2_temp, flair_temp)
                        input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                        
                        pred_mask = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                        
                        if prediction_volume == None:
                            prediction_volume = torch.unsqueeze(pred_mask, dim = 2)
                            gt_volume = torch.unsqueeze(gt_masks, dim = 2)
                        else:
                            pred_mask = torch.unsqueeze(pred_mask, dim = 2)
                            gt_mask = torch.unsqueeze(gt_masks, dim = 2)
                            prediction_volume = torch.cat((prediction_volume, pred_mask), dim = 2)
                            gt_volume = torch.cat((gt_volume, gt_mask), dim = 2)
                        
        
                    combined_input = torch.cat((input_3d, prediction_volume), dim = 1)
                    
                    combined_input = self.connection_layer(combined_input.cuda())
                    out3d = self.unet_3D(combined_input)

                    # gt_volume = torch.squeeze(gt_volume, dim = 1)
                    
                    # print('output/gt shape: ', out3d[0].shape, gt_volume.shape, masks_3d.shape)
                    # loss3d = loss_segmentation_hem(out3d, masks_3d.long().cuda())
                    loss3d = self.deep_supervision_loss(out3d, masks_3d)
                    # print('loss: ', loss3d)
                    
                    optimizer_2.zero_grad()
                    loss3d.backward()
                    optimizer_2.step()
                    train_loss.append(loss3d.item())

                    arg_out3d = torch.argmax(out3d[0], dim = 1)
                    dsc, class_dsc, ious, class_iou, tc_scores, whole_scores = test_scores_3d(arg_out3d, masks_3d.cuda())
                    
                    hausdorf_scores = hausdorf_distance(arg_out3d, masks_3d)
                    train_ET_HD.append(hausdorf_scores['ET'])
                    train_TC_HD.append(hausdorf_scores['TC'])
                    train_whole_HD.append(hausdorf_scores['WT'])
                    
                    # print('class dice: ', class_dsc)
                    train_iou.append(ious.detach().cpu().numpy())
                    train_dice.append(dsc.detach().cpu().numpy())
                    train_class_dice.append(class_dsc.detach().cpu().numpy())
                    train_class_iou.append(class_iou.detach().cpu().numpy())
    
                    train_tc_dice.append(tc_scores[0].detach().cpu().numpy())
                    train_tc_iou.append(tc_scores[1].detach().cpu().numpy())
                    train_whole_dice.append(whole_scores[0].detach().cpu().numpy())
                    train_whole_iou.append(whole_scores[1].detach().cpu().numpy())
                
                # print('final class dice: ', np.array(train_class_dice).shape)
                # print(train_class_dice)
                print('Epoch: ', epoch)
                print('Initial Train Loss: ', np.mean(train_loss))
                print('Initial Train Dice: ', np.mean(train_dice))
                print('Initial Train IoU: ', np.mean(train_iou))
                print('Initial Training NET, Edema, ET dice: ', np.mean(train_class_dice, axis = 0))
                print('Initial Training NET, Edema, ET iou: ', np.mean(train_class_iou, axis = 0))
                print('Initial Training TC dice: ', np.mean(train_tc_dice))
                print('Initial Training TC iou: ', np.mean(train_tc_iou))
                print('Initial Training whole dice: ', np.mean(train_whole_dice))
                print('Initial Training whole iou: ', np.mean(train_whole_iou))
                
                print('Training ET HD: ', np.nanmean(train_ET_HD))
                print('Training TC HD: ', np.nanmean(train_TC_HD))
                print('Training Whole HD: ', np.nanmean(train_whole_HD))
            
            val_loss = []
            val_dice = []
            val_iou = []
            val_class_dice = []
            val_class_iou = []
            val_tc_dice = []
            val_tc_iou = []
            val_whole_dice = []
            val_whole_iou = []
            
            val_ET_HD = []
            val_TC_HD = []
            val_whole_HD = []
            
            self.encoder_2d.eval()
            self.encoder_3d.eval()
            self.decoder.eval()
            self.unet_3D.eval()
            self.connection_layer.eval()
            
            pred_save_folder = 'D:\\brain_tumor_segmentation\\visual_saves\\final_saves_18\\predictions'
            gt_save_folder = 'D:\\brain_tumor_segmentation\\visual_saves\\final_saves_18\\gt'
            input_folder = 'D:\\brain_tumor_segmentation\\visual_saves\\final_saves_18\\input_samples'
            name_itterator = 0
            
            for sample in val_loader:
                t1, t1ce, t2, flair, masks = sample
                input_3d = torch.zeros(t1.shape[0], 4, t1.shape[1], t1.shape[2], t1.shape[3])
                dummy_vol_tensor = torch.zeros_like(t1)
                input_3d[:, 0, :, :, :] = t1
                input_3d[:, 1, :, :, :] = t1ce
                input_3d[:, 2, :, :, :] = t2
                input_3d[:, 3, :, :, :] = flair
                input_3d = torch.permute(input_3d, (0, 1, 4, 2, 3)).cuda()
                masks_3d = torch.permute(masks, (0, 3, 1, 2)).cpu()
                input_3d = (input_3d - input_3d.min()) / (input_3d.max() - input_3d.min())
                
                vol_depth = t1ce.shape[3]
                # print('t1ce shape: ', t1ce.shape)
                prediction_volume = None
                gt_volume = None
                
                with torch.no_grad():
                    for itter in range(0, vol_depth, 1):
                        input1, input2, gt_masks = self.dynamic_slicer(t1, t1ce, t2, flair, masks, slices, itter, t1_temp, t1ce_temp, t2_temp, flair_temp)
                        input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()

                        pred_mask = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())

                        if prediction_volume == None:
                            prediction_volume = torch.unsqueeze(pred_mask, dim = 2)
                            # print('gt_masks: ', gt_masks.shape)
                            gt_volume = torch.unsqueeze(gt_masks, dim = 2)
                        else:
                            pred_mask = torch.unsqueeze(pred_mask, dim = 2)
                            gt_mask = torch.unsqueeze(gt_masks, dim = 2)
                            # print('exp pred mask: ', pred_mask.shape)
                            prediction_volume = torch.cat((prediction_volume, pred_mask), dim = 2)
                            gt_volume = torch.cat((gt_volume, gt_mask), dim = 2)

                    combined_input = torch.cat((input_3d, prediction_volume), dim = 1)
                    
                    combined_input = self.connection_layer(combined_input.cuda())
                    out3d = self.unet_3D(combined_input)
                    
                    # gt_volume = torch.squeeze(gt_volume, dim = 1)
                    # loss = loss_segmentation_hem(out3d, masks_3d.long().cuda())
                    loss = self.deep_supervision_loss(out3d, masks_3d)
                    scheduler2.step(loss)
                
                
                # sm = torch.nn.Softmax(dim = 1)
                # out3d[0] = sm(out3d[0])
                
                # ets = out3d[0][:, 3, :, :, :]
                # ets = torch.where(ets < 0.95, 0, ets)
                # out3d[0][:, 3, :, :, :] = ets
                
                arg_out3d = torch.argmax(out3d[0], dim = 1)
                save_name = 'sample_' + str(name_itterator) + '.npy'
                
                t1_name = 't1_' + str(name_itterator) + '.npy'
                t1ce_name = 't1ce_' + str(name_itterator) + '.npy'
                t2_name = 't2_' + str(name_itterator) + '.npy'
                flair_name = 'flair_' + str(name_itterator) + '.npy'
                
                # np.save(os.path.join(pred_save_folder, save_name), arg_out3d.detach().cpu().numpy())
                # np.save(os.path.join(gt_save_folder, save_name), gt_volume.detach().cpu().numpy())
                
                # np.save(os.path.join(input_folder, t1_name), t1.detach().cpu().numpy())
                # np.save(os.path.join(input_folder, t1ce_name), t1ce.detach().cpu().numpy())
                # np.save(os.path.join(input_folder, t2_name), t2.detach().cpu().numpy())
                # np.save(os.path.join(input_folder, flair_name), flair.detach().cpu().numpy())
                
                # print('devices: ', arg_out3d.get_device(), masks.get_device())
                
                
                # arg_out3d = self.thresh_process(arg_out3d)
                dsc, class_dsc, ious, class_iou, tc_scores, whole_scores = test_scores_3d(arg_out3d, masks_3d.cuda())
                
                print('Dice score ' + str(name_itterator) + ' :', dsc)
                name_itterator += 1
                
                hausdorf_scores = hausdorf_distance(arg_out3d, masks_3d)
                val_ET_HD.append(hausdorf_scores['ET'])
                val_TC_HD.append(hausdorf_scores['TC'])
                val_whole_HD.append(hausdorf_scores['WT'])
                
                val_loss.append(loss.item())
                val_dice.append(dsc.detach().cpu().numpy())
                val_iou.append(ious.detach().cpu().numpy())
                val_class_dice.append(class_dsc.detach().cpu().numpy())
                val_class_iou.append(class_iou.detach().cpu().numpy())

                val_tc_dice.append(tc_scores[0].detach().cpu().numpy())
                val_tc_iou.append(tc_scores[1].detach().cpu().numpy())
                val_whole_dice.append(whole_scores[0].detach().cpu().numpy())
                val_whole_iou.append(whole_scores[1].detach().cpu().numpy())
                
            print('Initial Validation Loss: ', np.mean(val_loss))
            print('Initial Validation Dice: ', np.mean(val_dice))
            print('Initial Validation IoU: ', np.mean(val_iou))
            print('Initial Validation NET, Edema, ET dice: ', np.mean(val_class_dice, axis = 0))
            print('Initial Validation NET, Edema, ET iou: ', np.mean(val_class_iou, axis = 0))
            print('Initial Validation TC dice: ', np.mean(val_tc_dice))
            print('Initial Validation TC iou: ', np.mean(val_tc_iou))
            print('Initial Validation whole dice: ', np.mean(val_whole_dice))
            print('Initial Validation whole iou: ', np.mean(val_whole_iou))
            print('\n')
            
            print('Validation ET HD: ', np.nanmean(val_ET_HD))
            print('Validation TC HD: ', np.nanmean(val_TC_HD))
            print('Validation Whole HD: ', np.nanmean(val_whole_HD))
            
            if dice_latch < np.mean(val_dice):
                # save_encoder1 = 'Encoder2D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder12 = 'Encoder2D.pth'
                
                # save_encoder2 = 'Encoder3D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder22 = 'Encoder3D.pth'
                
                # save_decoder = 'Decoder_' + str(np.mean(val_dice)) + '.pth'
                save_decoder2 = 'Decoder.pth'
                
                # save_unet_3d = 'UNet_' + str(np.mean(val_dice)) + '.pth'
                save_unet_3d2 = 'UNet.pth'
                
                # torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder1))
                # torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder12))
                
                # # # # # # torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder2))
                # torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder22))
                
                # # # # # # torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder))
                # torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder2))
                
                # torch.save(self.unet_3D.state_dict(), os.path.join(self.weight_save_path[0], save_unet_3d2))
                
                # torch.save(self.connection_layer.state_dict(), os.path.join(self.weight_save_path[0], 'Connection.pth'))
                
                dice_latch = np.mean(val_dice)
            
            with open(self.record_save_path[0], 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(train_loss)} Train Dice: {np.mean(train_dice)} Train IoU: {np.mean(train_iou)}')
                f.write('\n')
                f.write(f'Train NET, Edema, ET dice: {np.mean(train_class_dice, axis = 0)} Train NET, Edema, ET iou: {np.mean(train_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Train TC dice: {np.mean(train_tc_dice)} Train TC iou: {np.mean(train_tc_iou)}')
                f.write('\n')
                f.write(f'Train whole dice: {np.mean(train_whole_dice)} Train whole iou: {np.mean(train_whole_iou)}')
                f.write('\n')
                
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation Dice: {np.mean(val_dice)} Validation IoU: {np.mean(val_iou)}')
                f.write('\n')
                f.write(f'Validation NET, Edema, ET dice: {np.mean(val_class_dice, axis = 0)} Validation NET, Edema, ET iou: {np.mean(val_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Validation TC dice: {np.mean(val_tc_dice)} Validation TC iou: {np.mean(val_tc_iou)}')
                f.write('\n')
                f.write(f'Validation whole dice: {np.mean(val_whole_dice)} Validation whole iou: {np.mean(val_whole_iou)}')
                f.write('\n')
                f.write('\n')
    
    def thresh_process(self, mask):
        
        pred_et = mask.clone()
        print('uniques before: ', torch.unique(pred_et))
        pred_et[pred_et != 3] = 0
        pred_et[pred_et == 3] = 1
        
        pred_et = pred_et.detach().cpu().numpy().astype(np.uint8)
        #print('pred_et shape / unique: ', pred_et.shape, np.unique(pred_et))
        
        temp = np.zeros((1, 64, 128, 128))
        
        for s in range(64):
            pred_temp = pred_et[0, s, :, :]
            #print('contour in shape: ', pred_temp.shape)
            contours, _ = cv2.findContours(pred_temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
            for cont in contours:
                #print('length contours: ', len(cont))
                if len(cont) < 15:
                    #print('ENTERED')
                    pred_temp[cont] = 0
                
            temp[0, s, :, :] = pred_temp
        
        pred_et = torch.Tensor(temp).cuda()
        #print('final shapes: ', pred_et.shape, mask.shape)
        pred_final = torch.where((pred_et == 0) & (mask == 3), 0, mask)
        print('uniques after: ', torch.unique(pred_final))
        return pred_final

    
    def train_loop_mixed(self, num_epochs, base_lr, train_loader, val_loader):
        dice_latch = 0
        slices = 5
        
        # save_encoder12 = 'Encoder2D' + str(sb) +'.pth'
        # save_encoder22 = 'Encoder3D' + str(sb) +'.pth'
        # save_decoder2 = 'Decoder' + str(sb) +'.pth'
        # if sb == 6:
        #     se = 1
        # else:
        #     se = 1
        # print('starting epoch: ', se)
        # self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder12)))
        # self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder22)))
        # self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_decoder2)))
        
        for epoch in range(num_epochs):
            train_loss = []
            train_dice = []
            train_iou = []
            train_class_dice = []
            train_class_iou = []
            train_tc_dice = []
            train_tc_iou = []
            train_whole_dice = []
            train_whole_iou = []
            
            self.encoder_2d.train()
            self.encoder_3d.train()
            self.decoder.train()
            
            # optimizer = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr)#, weight_decay = 1e-5)
            optimizer = torch.optim.Adam(self.unet_3D.parameters(), lr = base_lr)#, weight_decay = 1e-5)
            
            for sample in tqdm(train_loader):
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                loss, metrics, pred_mask = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                
                pred_mask = torch.argmax(pred_mask, dim = 1)
                
                # if bp == True:
                # print('backprop')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                # print(loss.item())
                loss = 0
                
                dsc, class_dsc, ious, class_iou, tc_scores, whole_scores = test_scores_3d(pred_mask, gt_masks)
                # print('here')
                
                train_iou.append(ious.detach().cpu().numpy())
                train_dice.append(dsc.detach().cpu().numpy())
                train_class_dice.append(class_dsc.detach().cpu().numpy())
                train_class_iou.append(class_iou.detach().cpu().numpy())
                
                train_tc_dice.append(tc_scores[0].detach().cpu().numpy())
                train_tc_iou.append(tc_scores[1].detach().cpu().numpy())
                train_whole_dice.append(whole_scores[0].detach().cpu().numpy())
                train_whole_iou.append(whole_scores[1].detach().cpu().numpy())
                
            print('Epoch: ', epoch)
            print('Initial Train Loss: ', np.mean(train_loss))
            print('Initial Train Dice: ', np.mean(train_dice))
            print('Initial Train IoU: ', np.mean(train_iou))
            print('Initial Training NET, Edema, ET dice: ', np.mean(train_class_dice, axis = 0))
            print('Initial Training NET, Edema, ET iou: ', np.mean(train_class_iou, axis = 0))
            print('Initial Training TC dice: ', np.mean(train_tc_dice))
            print('Initial Training TC iou: ', np.mean(train_tc_iou))
            print('Initial Training whole dice: ', np.mean(train_whole_dice))
            print('Initial Training whole iou: ', np.mean(train_whole_iou))
            
            val_loss = []
            val_dice = []
            val_iou = []
            val_class_dice = []
            val_class_iou = []
            val_tc_dice = []
            val_tc_iou = []
            val_whole_dice = []
            val_whole_iou = []
            
            self.encoder_2d.eval()
            self.encoder_3d.eval()
            self.decoder.eval()
            
            for sample in val_loader:
                v_input1, v_input2, v_gt_masks = sample
                v_input1, v_input2, v_gt_masks = v_input1.cuda(), v_input2.cuda(), v_gt_masks.cuda()
                        
                with torch.no_grad():
                    v_loss, v_metrics, v_pred_mask = self.Single_pass_initial(v_input1.float(), v_input2.float(), v_gt_masks.long())
                
                #print('pred out shape: ', v_pred_mask)
                v_pred_mask = torch.argmax(v_pred_mask, dim = 1)
                
                if len(torch.unique(v_gt_masks)) > 1:
                    val_loss.append(v_loss.item())
                    
                dsc, class_dsc, ious, class_iou, tc_scores, whole_scores = test_scores_3d(v_pred_mask, v_gt_masks)
                
                val_dice.append(dsc.detach().cpu().numpy())
                val_iou.append(ious.detach().cpu().numpy())
                val_class_dice.append(class_dsc.detach().cpu().numpy())
                val_class_iou.append(class_iou.detach().cpu().numpy())

                val_tc_dice.append(tc_scores[0].detach().cpu().numpy())
                val_tc_iou.append(tc_scores[1].detach().cpu().numpy())
                val_whole_dice.append(whole_scores[0].detach().cpu().numpy())
                val_whole_iou.append(whole_scores[1].detach().cpu().numpy())
                
            print('Initial Validation Loss: ', np.mean(val_loss))
            print('Initial Validation Dice: ', np.mean(val_dice))
            print('Initial Validation IoU: ', np.mean(val_iou))
            print('Initial Validation NET, Edema, ET dice: ', np.mean(val_class_dice, axis = 0))
            print('Initial Validation NET, Edema, ET iou: ', np.mean(val_class_iou, axis = 0))
            print('Initial Validation TC dice: ', np.mean(val_tc_dice))
            print('Initial Validation TC iou: ', np.mean(val_tc_iou))
            print('Initial Validation whole dice: ', np.mean(val_whole_dice))
            print('Initial Validation whole iou: ', np.mean(val_whole_iou))
            print('\n')
            
            # save_encoder1 = 'Encoder2D_' + str(np.mean(val_dice)) + '.pth'
            # save_encoder2 = 'Encoder3D_' + str(np.mean(val_dice)) + '.pth'
            # save_decoder = 'Decoder_' + str(np.mean(val_dice)) + '.pth'
            # torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder1))
            # torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder2))
            # torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder))
            
            if dice_latch < np.mean(val_dice):
                #save_encoder1 = 'Encoder2D_' + str(np.mean(val_dice)) + '.pth'
                # save_encoder12 = 'Encoder2D' + str(sb) + '.pth'
                
                #save_encoder2 = 'Encoder3D_' + str(np.mean(val_dice)) + '.pth'
                # save_encoder22 = 'Encoder3D' + str(sb) + '.pth'
                
                #save_decoder = 'Decoder_' + str(np.mean(val_dice)) + '.pth'
                # save_decoder2 = 'Decoder' + str(sb) + '.pth'
                
                save_unet = 'UNet_2D.pth'
                
                # #torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder1))
                # torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder12))
                
                # #torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder2))
                # torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder22))
                
                # #torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder))
                # torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder2))
                
                torch.save(self.unet_3D.state_dict(), os.path.join(self.weight_save_path[0], save_unet))
                
                dice_latch = np.mean(val_dice)
            
            with open(self.record_save_path[0], 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(train_loss)} Train Dice: {np.mean(train_dice)} Train IoU: {np.mean(train_iou)}')
                f.write('\n')
                f.write(f'Train NET, Edema, ET dice: {np.nanmean(train_class_dice, axis = 0)} Train NET, Edema, ET iou: {np.nanmean(train_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Train TC dice: {np.mean(train_tc_dice)} Train TC iou: {np.mean(train_tc_iou)}')
                f.write('\n')
                f.write(f'Train whole dice: {np.mean(train_whole_dice)} Train whole iou: {np.mean(train_whole_iou)}')
                f.write('\n')
                
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation Dice: {np.mean(val_dice)} Validation IoU: {np.mean(val_iou)}')
                f.write('\n')
                f.write(f'Validation NET, Edema, ET dice: {np.nanmean(val_class_dice, axis = 0)} Validation NET, Edema, ET iou: {np.nanmean(val_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Validation TC dice: {np.mean(val_tc_dice)} Validation TC iou: {np.mean(val_tc_iou)}')
                f.write('\n')
                f.write(f'Validation whole dice: {np.mean(val_whole_dice)} Validation whole iou: {np.mean(val_whole_iou)}')
                f.write('\n')
                f.write('\n')
    
    def Regularization_Loop(self, num_epochs, base_lr, train_loader, val_loader):
        save_encoder12 = 'Encoder2D.pth'
        save_encoder22 = 'Encoder3D.pth'
        save_disc12 = 'Discriminator1.pth'
        save_disc22 = 'Discriminator2.pth'

        self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder12)))
        self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder22)))
        
        #self.discriminator1.load_state_dict(torch.load(os.path.join(self.weight_save_path[1], save_disc12)))
        #self.discriminator2.load_state_dict(torch.load(os.path.join(self.weight_save_path[1], save_disc22)))
        
        optimizer_gen = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()), lr = base_lr, weight_decay = 1e-5)
        optimizer_disc = torch.optim.Adam(list(self.discriminator1.parameters()) + list(self.discriminator2.parameters()), lr = base_lr, weight_decay = 1e-5)
        
        acc_latch = 0
        
        for epoch in range(num_epochs):
            self.encoder_2d.train()
            self.encoder_3d.train()
            self.discriminator1.train()
            self.discriminator2.train()
            
            train_loss = []
            train_acc = []
            
            for sample in tqdm(train_loader):
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                tot_gen_loss, tot_disc_loss, acc = self.Single_pass_regularization_second(input1.float(), input2.float(), optimizer_gen, optimizer_disc, 'train')
                
                tl = tot_disc_loss.item() + tot_gen_loss.item()
                train_loss.append(tl)
                train_acc.append(acc)
                
            print('Epoch: ', epoch)
            print('Regularization Train Loss: ', np.mean(train_loss))
            print('Regularization Train accuracy: ', np.mean(train_acc))
            
            self.encoder_2d.eval()
            self.encoder_3d.eval()
            self.discriminator1.eval()
            self.discriminator2.eval()
            
            val_loss = []
            val_acc = []
            
            for sample in val_loader:
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                with torch.no_grad():
                    tot_gen_loss, tot_disc_loss, acc = self.Single_pass_regularization_second(input1.float(), input2.float(), optimizer_gen, optimizer_disc, 'val')
                
                tl = tot_disc_loss.item() + tot_gen_loss.item()
                val_loss.append(tl)
                val_acc.append(acc)
                
            print('Regularization Validation Loss: ', np.mean(val_loss))
            print('Regularization Validation accuracy: ', np.mean(val_acc))
            print('\n')
            
            if acc_latch < np.mean(val_acc):
                save_encoder1 = 'Encoder2D_' + str(np.mean(val_acc)) + '.pth'
                save_encoder12 = 'Encoder2D.pth'
                
                save_encoder2 = 'Encoder3D_' + str(np.mean(val_acc)) + '.pth'
                save_encoder22 = 'Encoder3D.pth'
                
                save_discriminator1 = 'Discriminator1_' + str(np.mean(val_acc)) + '.pth'
                save_discriminator12 = 'Discriminator1.pth'
                
                save_discriminator2 = 'Discriminator2_' + str(np.mean(val_acc)) + '.pth'
                save_discriminator22 = 'Discriminator2.pth'
                
                torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[1], save_encoder1))
                torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[1], save_encoder12))
                
                torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[1], save_encoder2))
                torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[1], save_encoder22))
                
                torch.save(self.discriminator1.state_dict(), os.path.join(self.weight_save_path[1], save_discriminator1))
                torch.save(self.discriminator1.state_dict(), os.path.join(self.weight_save_path[1], save_discriminator12))
                
                torch.save(self.discriminator2.state_dict(), os.path.join(self.weight_save_path[1], save_discriminator2))
                torch.save(self.discriminator2.state_dict(), os.path.join(self.weight_save_path[1], save_discriminator22))
                
                acc_latch = np.mean(val_acc)
            
            with open(self.record_save_path[1], 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(train_loss)} Train Accuracy: {np.mean(train_acc)}')
                f.write('\n')
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation Accuracy: {np.mean(val_acc)}')
                f.write('\n')
                f.write('\n')
            
    
    def Combined_loop(self, num_epochs, base_lr, train_loader, val_loader, sb):
        save_encoder12 = 'Encoder2D' + str(sb) +'.pth'
        save_encoder22 = 'Encoder3D' + str(sb) +'.pth'
        save_decoder2 = 'Decoder' + str(sb) +'.pth'
        save_discriminator12 = 'Discriminator1' + str(sb) +'.pth'
        save_discriminator22 = 'Discriminator2' + str(sb) +'.pth'
        
        self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder12)))
        self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder22)))
        # self.discriminator1.load_state_dict(torch.load(os.path.join(self.weight_save_path[2], save_discriminator12)))
        # self.discriminator2.load_state_dict(torch.load(os.path.join(self.weight_save_path[2], save_discriminator22)))
        self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_decoder2)))
        
        optimizer_gen = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr, weight_decay = 1e-5)
        optimizer_disc = torch.optim.Adam(list(self.discriminator1.parameters()) + list(self.discriminator2.parameters()), lr = base_lr, weight_decay = 1e-5)
        
        
        dice_latch = 0
        slices = 5
        
        for epoch in range(0, num_epochs):
            self.encoder_2d.train()
            self.encoder_3d.train()
            self.discriminator1.train()
            self.discriminator2.train()
            self.decoder.train()
            
            train_loss = []
            train_dice = []
            train_iou = []
            train_class_dice = []
            train_class_iou = []
            train_tc_dice = []
            train_tc_iou = []
            train_whole_dice = []
            train_whole_iou = []
            
            for sample in tqdm(train_loader):
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                

                loss, dice, class_dsc, iou, class_iou, tc_score, whole_scores, pred_mask = self.Single_pass_complete(input1.float(), input2.float(), gt_masks.long(), optimizer_gen, optimizer_disc, 'train')
                
                pred_mask = torch.argmax(pred_mask, dim = 1)
                
                class_dsc[class_dsc == 0] = np.nan
                class_iou[class_iou == 0] = np.nan
                
                train_loss.append(loss.item())
                if dice != 0:
                    train_dice.append(dice.detach().cpu().numpy())
                    train_iou.append(iou.detach().cpu().numpy())
                    #print(np.array(class_dsc.detach().cpu().numpy()))
                    train_class_dice.append(np.array(class_dsc.detach().cpu().numpy()))
                    train_class_iou.append(np.array(class_iou.detach().cpu().numpy()))
                    train_tc_dice.append(tc_score[0].detach().cpu().numpy())
                    train_tc_iou.append(tc_score[1].detach().cpu().numpy())
                    train_whole_dice.append(whole_scores[0].detach().cpu().numpy())
                    train_whole_iou.append(whole_scores[1].detach().cpu().numpy())
            
            train_class_dice = np.array(train_class_dice)
            print('Epoch: ', epoch)
            print('class dice shape: ', train_class_dice.shape)
            print('Combined Training Loss: ', np.mean(train_loss))
            print('Combined Training mean dice: ', np.mean(train_dice))
            print('Combined Training mean iou: ', np.mean(train_iou))
            print('Combined Training NET, Edema, ET dice: ', np.nanmean(train_class_dice, axis = 0))
            print('Combined Training NET, Edema, ET iou: ', np.nanmean(train_class_iou, axis = 0))
            print('Combined Training TC dice: ', np.mean(train_tc_dice))
            print('Combined Training TC iou: ', np.mean(train_tc_iou))
            print('Combined Training whole dice: ', np.mean(train_whole_dice))
            print('Combined Training whole iou: ', np.mean(train_whole_iou))
            print('\n')
            
            val_loss = []
            val_dice = []
            val_iou = []
            val_class_dice = []
            val_class_iou = []
            val_tc_dice = []
            val_tc_iou = []
            val_whole_dice = []
            val_whole_iou = []
            
            for sample in val_loader:
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                with torch.no_grad():
                    loss, dice, class_dsc, iou, class_iou, tc_score, whole_scores, pred_mask = self.Single_pass_complete(input1.float(), input2.float(), gt_masks.long(), optimizer_gen, optimizer_disc, 'val')
                
                class_dsc[class_dsc == 0] = np.nan
                class_iou[class_iou == 0] = np.nan
                
                if dice != 0:
                    val_loss.append(loss.item())
                    val_dice.append(dice.detach().cpu().numpy())
                    val_iou.append(iou.detach().cpu().numpy())
                    val_class_dice.append(np.array(class_dsc.detach().cpu().numpy()))
                    val_class_iou.append(np.array(class_iou.detach().cpu().numpy()))
                    val_tc_dice.append(tc_score[0].detach().cpu().numpy())
                    val_tc_iou.append(tc_score[1].detach().cpu().numpy())
                    val_whole_dice.append(whole_scores[0].detach().cpu().numpy())
                    val_whole_iou.append(whole_scores[1].detach().cpu().numpy())
                
            print('Combined Validation Loss: ', np.mean(val_loss))
            print('Combined Validation mean dice: ', np.mean(val_dice))
            print('Combined Validation mean iou: ', np.mean(val_iou))
            print('Combined Validation NET, Edema, ET dice: ', np.nanmean(val_class_dice, axis = 0))
            print('Combined Validation NET, Edema, ET iou: ', np.nanmean(val_class_iou, axis = 0))
            print('Combined Validation TC dice: ', np.mean(val_tc_dice))
            print('Combined Validation TC iou: ', np.mean(val_tc_iou))
            print('Combined Validation whole dice: ', np.mean(val_whole_dice))
            print('Combined Validation whole iou: ', np.mean(val_whole_iou))
            
            if dice_latch < np.mean(val_dice):
                # save_encoder1 = 'Encoder2D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder12 = 'Encoder2D' + str(sb) +'.pth'
                
                # save_encoder2 = 'Encoder3D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder22 = 'Encoder3D' + str(sb) +'.pth'
                
                # save_discriminator1 = 'Discriminator1_' + str(np.mean(val_dice)) + '.pth'
                save_discriminator12 = 'Discriminator1' + str(sb) +'.pth'
                
                # save_discriminator2 = 'Discriminator2_' + str(np.mean(val_dice)) + '.pth'
                save_discriminator22 = 'Discriminator2' + str(sb) +'.pth'
                
                # save_decoder = 'Decoder_' + str(np.mean(val_dice)) + '.pth'
                save_decoder2 = 'Decoder' + str(sb) +'.pth'
                
                # torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder1))
                torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder12))
                
                # torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder2))
                torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder22))
                
                # torch.save(self.discriminator1.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator1))
                torch.save(self.discriminator1.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator12))
                
                # torch.save(self.discriminator2.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator2))
                torch.save(self.discriminator2.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator22))
                
                # torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[2], save_decoder))
                torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[2], save_decoder2))
                
                dice_latch = np.mean(val_dice)
            
            #print('save path: ', self.record_save_path[2])
            with open(self.record_save_path[2], 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(train_loss)} Train Dice: {np.mean(train_dice)} Train IoU: {np.mean(train_iou)}')
                f.write('\n')
                f.write(f'Train NET, Edema, ET dice: {np.nanmean(train_class_dice, axis = 0)} Train NET, Edema, ET iou: {np.nanmean(train_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Train TC dice: {np.mean(train_tc_dice)} Train TC iou: {np.mean(train_tc_iou)}')
                f.write('\n')
                f.write(f'Train whole dice: {np.mean(train_whole_dice)} Train whole iou: {np.mean(train_whole_iou)}')
                f.write('\n')
                
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation Dice: {np.mean(val_dice)} Validation IoU: {np.mean(val_iou)}')
                f.write('\n')
                f.write(f'Validation NET, Edema, ET dice: {np.nanmean(val_class_dice, axis = 0)} Validation NET, Edema, ET iou: {np.nanmean(val_class_iou, axis = 0)}')
                f.write('\n')
                f.write(f'Validation TC dice: {np.mean(val_tc_dice)} Validation TC iou: {np.mean(val_tc_iou)}')
                f.write('\n')
                f.write(f'Validation whole dice: {np.mean(val_whole_dice)} Validation whole iou: {np.mean(val_whole_iou)}')
                f.write('\n')
                f.write('\n')
        
    
    def testing_whole_samples(self, test_loader, slices, sb):
        # save_encoder12 = 'Encoder2D_' + str(sb) + '.pth'
        # save_encoder22 = 'Encoder3D_' + str(sb) + '.pth'
        # save_decoder2 = 'Decoder_' + str(sb) + '.pth'
        
        # self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder12)))
        # self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder22)))
        # self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_decoder2)))
        
        self.encoder_2d.eval()
        self.encoder_3d.eval()
        self.decoder.eval()
        
        test_dice = []
        test_iou = []
        test_class_dice = []
        test_class_iou = []
        test_tc_dice = []
        test_tc_iou = []
        test_whole_dice = []
        test_whole_iou = []
        
        t_dice = []
        
        hd_et = []
        hd_tc = []
        hd_wt = []
        
        sensitivity_et = []
        sensitivity_tc = []
        sensitivity_wt = []
        
        specificity_et = []
        specificity_tc = []
        specificity_wt = []
        
        fp_record_wt = []
        fn_record_wt = []
        
        fp_record_et = []
        fn_record_et = []
        
        fp_record_tc = []
        fn_record_tc = []
        
        nn = 0
        for sample in tqdm(test_loader):
            t1_path, t1ce_path, t2_path, flair_path, t1ce, flair, gt_mask = sample
            #print(t1ce_path)
            #t1cee = nib.load(t1ce_path[0]).get_fdata()
            #t1ce = torch.squeeze(t1ce, dim = 0)
            #flair = torch.squeeze(flair, dim = 0)
            #gt_mask = torch.squeeze(gt_mask, dim = 0)
            # print('t1ce shape: ', t1ce.shape)
            vol_depth = t1ce.shape[3]
            
            prediction_volume = torch.zeros_like(t1ce)
            
            sq = torch.squeeze(gt_mask, 0)
            # print(sq.shape)
            # print(torch.unique(torch.where(sq > 0)[0]))
            
            for itter in range(10, vol_depth-10, 1):
                input1, input2, gt_masks = self.dynamic_slicer(t1ce, flair, prediction_volume, gt_mask, slices, itter)
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                with torch.no_grad():
                    loss, metrics, pred_mask, _ = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                
                
                pred_mask = torch.sigmoid(pred_mask)#, dim = 1) #softmax

                fpred_mask = torch.argmax(pred_mask, dim = 1)
                fpred_mask = torch.unsqueeze(fpred_mask, dim = 0)
                
                z_ch = pred_mask[:, 0, :, :]
                z_ch = torch.unsqueeze(z_ch, dim = 0)
                
                et_ch = pred_mask[:, 3, :, :]
                et_ch = torch.unsqueeze(et_ch, dim = 0)

                prediction_volume[:, :, :, itter] = fpred_mask

                # if len(torch.unique(fpred_mask)) >= 2:# or len(torch.unique(gt_masks)) >= 1:
                #     s_pred_mask = torch.squeeze(fpred_mask).detach().cpu().numpy()
                #     s_gt_mask = torch.squeeze(gt_masks).detach().cpu().numpy()
                #     # s_image = t1ce[0, itter, :, :].detach().cpu().numpy()
                #     # print('s image shape: ', s_image.shape)
                    
                #     rec = []
                #     rec.append([t1_path, t1ce_path, t2_path, flair_path, itter])
                    
                #     #print('pred/gt  mask shape: ', s_pred_mask.shape, s_gt_mask.shape)
                #     save_path = 'D:\\brain_tumor_segmentation\\visual_saves\\after_pre_processing'
                #     name = str(nn) + '.npy'
                #     gt_name = str(nn) + '_gt.npy'
                #     image_name = str(nn) + '_image.npy'
                #     # cv2.imwrite(os.path.join(save_path, name), s_pred_mask)
                #     # cv2.imwrite(os.path.join(save_path, gt_name), s_gt_mask)
                #     # cv2.imwrite(os.path.join(save_path, image_name), s_image)
                #     rec = np.array(rec)
                #     np.save(os.path.join(save_path, name), pred_mask.detach().cpu().numpy())
                #     np.save(os.path.join(save_path, gt_name), s_gt_mask)
                #     np.save(os.path.join(save_path, image_name), rec)
                #     #print('saved ', nn)
                #     nn += 1
            
            
            dice, class_dsc, iou, class_iou, tc_score, whole_scores = test_scores_3d(prediction_volume, gt_mask)
            
            sensitivity, specificity, false_positives, false_negatives = other_metrics(prediction_volume, gt_mask)
            sensitivity_et.append(sensitivity[0])
            sensitivity_tc.append(sensitivity[1])
            sensitivity_wt.append(sensitivity[2])
            
            specificity_et.append(specificity[0])
            specificity_tc.append(specificity[1])
            specificity_wt.append(specificity[2])
            
            fp_record_et.append(false_positives[0])
            fn_record_et.append(false_negatives[0])
            
            fp_record_tc.append(false_positives[1])
            fn_record_tc.append(false_negatives[1])
            
            fp_record_wt.append(false_positives[2])
            fn_record_wt.append(false_negatives[2])
            
            # hd_dict = hausdorf_distance(prediction_volume, gt_mask)
            # hd_et.append(hd_dict['ET'])
            # hd_tc.append(hd_dict['TC'])
            # hd_wt.append(hd_dict['WT'])
            # print('hausdorf distances: ', hd_dict['ET'], hd_dict['TC'], hd_dict['WT'])
            
            # print('pred unique: ', torch.unique(prediction_volume))
            # print('gt unique: ', torch.unique(gt_mask))
            
            dd = dice_3d(gt_mask.long(), prediction_volume)
            t_dice.append(dd)
            
            # print('ET, TC, whole false positives: ', np.array(false_positives)/57600)
            # print('ET, TC, whole false_negatives: ', np.array(false_negatives)/57600)
            
            test_dice.append(dice)
            test_iou.append(iou)
            test_class_dice.append(np.array(class_dsc.detach().cpu().numpy()))
            test_class_iou.append(np.array(class_iou.detach().cpu().numpy()))
            test_tc_dice.append(tc_score[0].detach().cpu().numpy())
            test_tc_iou.append(tc_score[1].detach().cpu().numpy())
            test_whole_dice.append(whole_scores[0].detach().cpu().numpy())
            test_whole_iou.append(whole_scores[1].detach().cpu().numpy())
            
            print('sample dice: ', dice, np.mean(test_dice))
            print('Mean Testing NET, Edema, ET dice: ', np.mean(test_class_dice, axis = 0))
            print('Mean Testing TC dice: ', np.mean(test_tc_dice))
            print('Mean Testing whole dice: ', np.mean(test_whole_dice))
            
        
        print('Combined Testing mean dice: ', np.mean(test_dice))
        print('Combined Testing mean iou: ', np.mean(test_iou))
        print('Combined Testing NET, Edema, ET dice: ', np.mean(test_class_dice, axis = 0))
        print('Combined Testing NET, Edema, ET iou: ', np.mean(test_class_iou, axis = 0))
        print('Combined Testing TC dice: ', np.mean(test_tc_dice))
        print('Combined Testing TC iou: ', np.mean(test_tc_iou))
        print('Combined Testing whole dice: ', np.mean(test_whole_dice))
        print('Combined Testing whole iou: ', np.mean(test_whole_iou))
        
        print('Sensitivity ET, TC, WT: ', np.mean(sensitivity_et), np.mean(sensitivity_tc), np.mean(sensitivity_wt))
        print('Specificity ET, TC, WT: ', np.mean(specificity_et), np.mean(specificity_tc), np.mean(specificity_wt))
        
        # print('Testing ET hausforf: ', np.mean(hd_et))
        # print('Testing TC hausforf: ', np.mean(hd_tc))
        # print('Testing WT hausforf: ', np.mean(hd_wt))
        
        print('Mean/Max/Min FP ET: ', np.mean(fp_record_et), np.max(fp_record_et), np.min(fp_record_et))
        print('Mean/Max/Min FN ET: ', np.mean(fn_record_et), np.max(fn_record_et), np.min(fn_record_et))
        print('Mean/Max/Min FP TC: ', np.mean(fp_record_tc), np.max(fp_record_tc), np.min(fp_record_tc))
        print('Mean/Max/Min FN TC: ', np.mean(fn_record_tc), np.max(fn_record_tc), np.min(fn_record_tc))
        print('Mean/Max/Min FP WT: ', np.mean(fp_record_wt), np.max(fp_record_wt), np.min(fp_record_wt))
        print('Mean/Max/Min FN WT: ', np.mean(fn_record_wt), np.max(fn_record_wt), np.min(fn_record_wt))
        
        
        # with open(self.record_save_path[3], 'a') as f:
        #     f.write(f'Testing Dice: {np.mean(test_dice)} Testing IoU: {np.mean(test_iou)}')
        #     f.write('\n')
        #     f.write(f'Testing NET, Edema, ET dice: {np.mean(test_class_dice, axis = 0)} Testing NET, Edema, ET iou: {np.mean(test_class_iou, axis = 0)}')
        #     f.write('\n')
        #     f.write(f'Testing TC dice: {np.mean(test_tc_dice)} Testing TC iou: {np.mean(test_tc_iou)}')
        #     f.write('\n')
        #     f.write(f'Testing whole dice: {np.mean(test_whole_dice)} Testing whole iou: {np.mean(test_whole_iou)}')
        #     f.write('\n')
        #     f.write(f'Testing Sensitivity ET, TC, WT: {np.mean(sensitivity_et)}, {np.mean(sensitivity_tc)}, {np.mean(sensitivity_wt)}')
        #     f.write('\n')
        #     f.write(f'Testing hausforf ET, TC, WT: {np.mean(hd_et)}, {np.mean(hd_tc)}, {np.mean(hd_wt)}')
        #     f.write('\n')
            