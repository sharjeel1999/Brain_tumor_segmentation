import torch
import torch.nn as nn
import random
import numpy as np
import os
from utils import loss_segmentation, loss_detection, dice_coeff, class_dice, test_scores_3d, dice_3d
from torch.autograd import Variable
import time
from tqdm import tqdm

import cv2
import nibabel as nib

class Run_Model():
    def __init__(self, weight_save_path, record_save_path, encoder_2d, encoder_3d, decoder, discriminator1, discriminator2):
        self.weight_save_path = weight_save_path
        self.record_save_path = record_save_path
        self.encoder_2d = encoder_2d.cuda()
        self.encoder_3d = encoder_3d.cuda()
        self.decoder = decoder.cuda()
        self.discriminator1 = discriminator1.cuda()
        self.discriminator2 = discriminator2.cuda()
    
    def Single_pass_initial(self, input_1, input_2, gt_mask):
        #print('input 1: ', input_1.shape)
        #print('input 2: ', input_2.shape)
        #print('gt mask: ', gt_mask.shape)
        
        #input_2 = torch.unsqueeze((input_2), dim = 1)
        gt_mask = torch.squeeze(gt_mask, dim = 1)
        
        out_2d_ar = self.encoder_2d(input_1)
        out_3d_ar = self.encoder_3d(input_2)
        #out_2d = out_2d_ar[0]
        #out_3d = out_3d_ar[0]
        
        #out_2d = torch.unsqueeze(out_2d, dim = 2)
        #print('out_2d / out_3d shape: ', out_2d.shape, out_3d.shape)
        #combined_features = torch.cat((out_2d, out_3d), dim = 1)
        #print('combined shape: ', combined_features.shape)
        
        dec_out = self.decoder(out_2d_ar, out_3d_ar)
        #dec_out = self.crf(dec_out)
        #print('decoder out shape: ', dec_out.shape, gt_mask.shape)
        seg_loss, dsc, class_dsc, ious, class_iou, tc_score, whole_scores = loss_segmentation(dec_out, gt_mask)
        
        metrics = {}
        metrics['dice'] = dsc
        metrics['iou'] = ious
        metrics['class_dice'] = class_dsc
        metrics['class_iou'] = class_iou
        metrics['tc_scores'] = tc_score
        metrics['whole_scores'] = whole_scores
        
        return seg_loss, metrics, dec_out
    
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
        
        tot_gen_loss = 0.2*(det_loss1 + det_loss2) + seg_loss
        
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
        
        z2d = nn.Parameter(Tensor(np.random.normal(0, 1, (input_1.shape[0], out_2d.shape[1], out_2d.shape[2], out_2d.shape[3]))), requires_grad=False).cuda()
        z3d = nn.Parameter(Tensor(np.random.normal(0, 1, (input_1.shape[0], out_3d.shape[1], out_3d.shape[2], out_3d.shape[3], out_3d.shape[4]))), requires_grad=False).cuda()
        
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
    
    
    def dynamic_slicer(self, t1ce_volume, flair_volume, prev_mask_volume, gt_mask_volume, slices, selected_ind):
        start = int(selected_ind - ((slices - 1)/2))
        end = int(selected_ind + ((slices - 1)/2) + 1)
        
        max_depth = t1ce_volume.shape[1] - 1
        mm = slices // 2
        mms = (slices - mm)*2
        
        input1 = torch.zeros((t1ce_volume.shape[0], 2, 240, 240))
        input2 = torch.zeros((t1ce_volume.shape[0], 2, slices, 240, 240))
        gt_masks = torch.zeros((t1ce_volume.shape[0], 1, 240, 240))
        
        ent = False
        for k, z in enumerate(range(start, end)):
            
            if z < 0:
                z = 0
            if z > max_depth:
                z = max_depth
            
            #print('dynamic all shapes: ', t1ce_volume.shape, flair_volume.shape, gt_mask_volume.shape)
            #t1_slice = t1_volume[:,z, :, :]
            t1ce_slice = t1ce_volume[:,z, :, :]
            #t2_slice = t2_volume[:,z, :, :]
            flair_slice = flair_volume[:, z, :, :]
            gt_mask_slice = gt_mask_volume[:, z, :, :]
            
            #input2[:, 0, k, :, :] = t1_slice
            input2[:, 0, k, :, :] = t1ce_slice
            #input2[:, 2, k, :, :] = t2_slice
            input2[:, 1, k, :, :] = flair_slice
            
            if z == selected_ind and ent == False:
                #print('z and k: ', z, k)
                #print('start, end: ', start, end)
                #input1[:, 0, :, :] = t1_slice
                input1[:, 0, :, :] = t1ce_slice
                #input1[:, 2, :, :] = t2_slice
                input1[:, 1, :, :] = flair_slice
                gt_masks[:, 0, :, :] = gt_mask_slice
                ent = True
        
        input1 = (input1 - input1.min()) / (input1.max() - input1.min())
        input2 = (input2 - input2.min()) / (input2.max() - input2.min())
        
        return input1, input2, gt_masks
    
    
    def train_loop(self, num_epochs, base_lr, train_loader, val_loader):
        
        dice_latch = 0
        slices = 5
        
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
            
            optimizer = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr)#, weight_decay = 1e-5)
            #optimizer = torch.optim.SGD(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr, momentum = 0.9)#, weight_decay = 1e-5)
            
            for sample in tqdm(train_loader):
                t1ce, flair, masks = sample
                #t1ce, flair, masks = t1ce.cuda(), flair.cuda(), masks.cuda()
                
                vol_depth = t1ce.shape[1]
                prediction_volume = torch.zeros_like(t1ce)
                
                
                for itter in range(vol_depth):
                    input1, input2, gt_masks = self.dynamic_slicer(t1ce, flair, prediction_volume, masks, slices, itter)
                
                    if input1.shape[0] > 1 and len(torch.unique(gt_masks)) > 1:
                        input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                        
                        loss, metrics, pred_mask = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                        
                        pred_mask = torch.argmax(pred_mask, dim = 1)
                        prediction_volume[:, itter, :, :] = pred_mask
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())
                        

                #print('prediction shape: ', prediction_volume.shape)
                #print('gt volume shape: ', masks.shape)
                dsc, class_dsc, ious, class_iou, tc_scores, whole_scores = test_scores_3d(prediction_volume, masks)
                
                
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
                v_t1ce, v_flair, v_masks = sample
                vol_depth = v_t1ce.shape[1]
                v_prediction_volume = torch.zeros_like(v_t1ce)
                
                for v_itter in range(vol_depth):
                    v_input1, v_input2, v_gt_masks = self.dynamic_slicer(v_t1ce, v_flair, v_prediction_volume, v_masks, slices, v_itter)
                
                    if input1.shape[0] > 1:
                        v_input1, v_input2, v_gt_masks = v_input1.cuda(), v_input2.cuda(), v_gt_masks.cuda()
                        
                        with torch.no_grad():
                            v_loss, v_metrics, v_pred_mask = self.Single_pass_initial(v_input1.float(), v_input2.float(), v_gt_masks.long())
                        
                        #print('pred out shape: ', v_pred_mask)
                        v_pred_mask = torch.argmax(v_pred_mask, dim = 1)
                        v_prediction_volume[:, v_itter, :, :] = v_pred_mask
                        
                        if len(torch.unique(v_gt_masks)) > 1:
                            val_loss.append(v_loss.item())
                        
 
                dsc, class_dsc, ious, class_iou, tc_scores, whole_scores = test_scores_3d(v_prediction_volume, v_masks)
                
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
            
            if dice_latch < np.mean(val_dice):
                save_encoder1 = 'Encoder2D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder12 = 'Encoder2D.pth'
                
                save_encoder2 = 'Encoder3D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder22 = 'Encoder3D.pth'
                
                save_decoder = 'Decoder_' + str(np.mean(val_dice)) + '.pth'
                save_decoder2 = 'Decoder.pth'
                
                # torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder1))
                # torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder12))
                
                # torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder2))
                # torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder22))
                
                # torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder))
                # torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder2))
                
                dice_latch = np.mean(val_dice)
            
            with open(self.record_save_path[0], 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(train_loss)} Train Dice: {np.mean(train_dice)} Train IoU: {np.mean(train_iou)}')
                f.write('\n')
                f.write(f'Train NET, Edema, ET dice: {np.nanmean(train_class_dice)} Train NET, Edema, ET iou: {np.nanmean(train_class_iou)}')
                f.write('\n')
                f.write(f'Train TC dice: {np.mean(train_tc_dice)} Train TC iou: {np.mean(train_tc_iou)}')
                f.write('\n')
                f.write(f'Train whole dice: {np.mean(train_whole_dice)} Train whole iou: {np.mean(train_whole_iou)}')
                f.write('\n')
                
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation Dice: {np.mean(val_dice)} Validation IoU: {np.mean(val_iou)}')
                f.write('\n')
                f.write(f'Validation NET, Edema, ET dice: {np.nanmean(val_class_dice)} Validation NET, Edema, ET iou: {np.nanmean(val_class_iou)}')
                f.write('\n')
                f.write(f'Validation TC dice: {np.mean(val_tc_dice)} Validation TC iou: {np.mean(val_tc_iou)}')
                f.write('\n')
                f.write(f'Validation whole dice: {np.mean(val_whole_dice)} Validation whole iou: {np.mean(val_whole_iou)}')
                f.write('\n')
                f.write('\n')
    
    def train_loop_mixed(self, num_epochs, base_lr, train_loader, val_loader):
        dice_latch = 0
        slices = 5
        
        save_encoder12 = 'Encoder2D.pth'
        save_encoder22 = 'Encoder3D.pth'
        save_decoder2 = 'Decoder.pth'
        
        self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder12)))
        self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder22)))
        self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_decoder2)))
        
        for epoch in range(7, num_epochs):
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
            
            optimizer = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr)#, weight_decay = 1e-5)
            #optimizer = torch.optim.SGD(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr, momentum = 0.9)#, weight_decay = 1e-5)
            
            for sample in tqdm(train_loader):
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                loss, metrics, pred_mask = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                
                pred_mask = torch.argmax(pred_mask, dim = 1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                
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
            
            if dice_latch < np.mean(val_dice):
                save_encoder1 = 'Encoder2D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder12 = 'Encoder2D.pth'
                
                save_encoder2 = 'Encoder3D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder22 = 'Encoder3D.pth'
                
                save_decoder = 'Decoder_' + str(np.mean(val_dice)) + '.pth'
                save_decoder2 = 'Decoder.pth'
                
                torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder1))
                torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder12))
                
                torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder2))
                torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[0], save_encoder22))
                
                torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder))
                torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[0], save_decoder2))
                
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
            
    
    def Combined_loop(self, num_epochs, base_lr, train_loader, val_loader):
        save_encoder12 = 'Encoder2D.pth'
        save_encoder22 = 'Encoder3D.pth'
        save_discriminator12 = 'Discriminator1.pth'
        save_discriminator22 = 'Discriminator2.pth'
        save_decoder2 = 'Decoder.pth'
        
        self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[2], save_encoder12)))
        self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[2], save_encoder22)))
        self.discriminator1.load_state_dict(torch.load(os.path.join(self.weight_save_path[2], save_discriminator12)))
        self.discriminator2.load_state_dict(torch.load(os.path.join(self.weight_save_path[2], save_discriminator22)))
        self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[2], save_decoder2)))
        
        optimizer_gen = torch.optim.Adam(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr, weight_decay = 1e-5)
        optimizer_disc = torch.optim.Adam(list(self.discriminator1.parameters()) + list(self.discriminator2.parameters()), lr = base_lr, weight_decay = 1e-5)
        
        
        dice_latch = 0
        slices = 5
        
        for epoch in range(2, num_epochs):
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
                save_encoder1 = 'Encoder2D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder12 = 'Encoder2D.pth'
                
                save_encoder2 = 'Encoder3D_' + str(np.mean(val_dice)) + '.pth'
                save_encoder22 = 'Encoder3D.pth'
                
                save_discriminator1 = 'Discriminator1_' + str(np.mean(val_dice)) + '.pth'
                save_discriminator12 = 'Discriminator1.pth'
                
                save_discriminator2 = 'Discriminator2_' + str(np.mean(val_dice)) + '.pth'
                save_discriminator22 = 'Discriminator2.pth'
                
                save_decoder = 'Decoder_' + str(np.mean(val_dice)) + '.pth'
                save_decoder2 = 'Decoder.pth'
                
                torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder1))
                torch.save(self.encoder_2d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder12))
                
                torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder2))
                torch.save(self.encoder_3d.state_dict(), os.path.join(self.weight_save_path[2], save_encoder22))
                
                torch.save(self.discriminator1.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator1))
                torch.save(self.discriminator1.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator12))
                
                torch.save(self.discriminator2.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator2))
                torch.save(self.discriminator2.state_dict(), os.path.join(self.weight_save_path[2], save_discriminator22))
                
                torch.save(self.decoder.state_dict(), os.path.join(self.weight_save_path[2], save_decoder))
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
        
    
    def testing_whole_samples(self, test_loader, slices):
        save_encoder12 = 'Encoder2D.pth'
        save_encoder22 = 'Encoder3D.pth'
        save_decoder2 = 'Decoder.pth'
        
        self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder12)))
        self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_encoder22)))
        self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_decoder2)))
        
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
        
        test_2d_class_dice = []
        test_2d_class_iou = []
        test_2d_tc_dice = []
        test_2d_tc_iou = []
        test_2d_wt_dice = []
        test_2d_wt_iou = []
        t_dice = []
        nn = 0
        for sample in tqdm(test_loader):
            t1_path, t1ce_path, t2_path, flair_path, t1ce, flair, gt_mask = sample
            #print(t1ce_path)
            #t1cee = nib.load(t1ce_path[0]).get_fdata()
            #t1ce = torch.squeeze(t1ce, dim = 0)
            #flair = torch.squeeze(flair, dim = 0)
            #gt_mask = torch.squeeze(gt_mask, dim = 0)
            #print('t1ce shape: ', t1ce.shape)
            vol_depth = t1ce.shape[1]
            
            prediction_volume = torch.zeros_like(t1ce)
            
            for itter in range(10, vol_depth-10, 1):
                input1, input2, gt_masks = self.dynamic_slicer(t1ce, flair, prediction_volume, gt_mask, slices, itter)
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                with torch.no_grad():
                    loss, metrics, pred_mask = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                
                dsc = metrics['dice']
                ious = metrics['iou']
                class_dsc = metrics['class_dice']
                class_iou = metrics['class_iou']
                tc_score = metrics['tc_scores']
                whole_scores = metrics['whole_scores']
                
                test_2d_class_dice.append(np.array(class_dsc.detach().cpu().numpy()))
                test_2d_class_iou.append(np.array(class_iou.detach().cpu().numpy()))
                test_2d_tc_dice.append(tc_score[0].detach().cpu().numpy())
                test_2d_tc_iou.append(tc_score[1].detach().cpu().numpy())
                test_2d_wt_dice.append(whole_scores[0].detach().cpu().numpy())
                test_2d_wt_iou.append(whole_scores[1].detach().cpu().numpy())
                
                
                pred_mask = torch.sigmoid(pred_mask) #softmax
                #print('uniques: ', torch.min(pred_mask), torch.max(pred_mask))
                
                #pred_mask[pred_mask < 0.4] = 0
                fpred_mask = torch.argmax(pred_mask, dim = 1)
                fpred_mask = torch.unsqueeze(fpred_mask, dim = 0)
                
                z_ch = pred_mask[:, 0, :, :]
                z_ch = torch.unsqueeze(z_ch, dim = 0)
                
                et_ch = pred_mask[:, 3, :, :]
                et_ch = torch.unsqueeze(et_ch, dim = 0)
                # print('fpred shape: ', fpred_mask.shape, torch.unique(fpred_mask))
                # print('et ch shape: ', et_ch.shape)
                
                #fpred_mask[z_ch >= 0.4] = 0
                
                #print('fpred/gt: ', torch.unique(fpred_mask), torch.unique(gt_masks))
                
                
                #fpred_mask
                prediction_volume[:, itter, :, :] = fpred_mask
                
                # if len(torch.unique(fpred_mask)) >= 2 or len(torch.unique(gt_masks)):
                #     s_pred_mask = torch.squeeze(fpred_mask).detach().cpu().numpy()
                #     s_gt_mask = torch.squeeze(gt_masks).detach().cpu().numpy()
                #     rec = []
                #     rec.append([t1_path, t1ce_path, t2_path, flair_path, itter])
                    
                #     #print('pred/gt  mask shape: ', s_pred_mask.shape, s_gt_mask.shape)
                #     save_path = 'D:\\brain_tumor_segmentation\\visual_saves\\experiment_3_rough2'
                #     name = str(nn) + '.png'
                #     gt_name = str(nn) + '_gt.png'
                #     image_name = str(nn) + '_image.npy'
                #     cv2.imwrite(os.path.join(save_path, name), s_pred_mask)
                #     cv2.imwrite(os.path.join(save_path, gt_name), s_gt_mask)
                #     #cv2.imwrite(os.path.join(save_path, image_name), s_image)
                #     rec = np.array(rec)
                #     np.save(os.path.join(save_path, image_name), rec)
                #     #print('saved ', nn)
                #     nn += 1
                    
                
            #print('prediction shape: ', prediction_volume.shape)
            #print('gt mask shape: ', gt_mask.shape)
            dice, class_dsc, iou, class_iou, tc_score, whole_scores = test_scores_3d(prediction_volume, gt_mask)
            
            dd = dice_3d(gt_mask.long(), prediction_volume)
            t_dice.append(dd)
            print('sample dice: ', dice)
            test_dice.append(dice)
            test_iou.append(iou)
            test_class_dice.append(np.array(class_dsc.detach().cpu().numpy()))
            test_class_iou.append(np.array(class_iou.detach().cpu().numpy()))
            test_tc_dice.append(tc_score[0].detach().cpu().numpy())
            test_tc_iou.append(tc_score[1].detach().cpu().numpy())
            test_whole_dice.append(whole_scores[0].detach().cpu().numpy())
            test_whole_iou.append(whole_scores[1].detach().cpu().numpy())
            
        # print('Combined 2D Testing NET, Edema, ET dice: ', np.mean(test_2d_class_dice, axis = 0))
        # print('Combined 2D Testing NET, Edema, ET iou: ', np.mean(test_2d_class_iou, axis = 0))
        # print('Combined 2D Testing TC dice: ', np.mean(test_2d_tc_dice))
        # print('Combined 2D Testing TC iou: ', np.mean(test_2d_tc_iou))
        # print('Combined 2D Testing whole dice: ', np.mean(test_2d_wt_dice))
        # print('Combined 2D Testing whole iou: ', np.mean(test_2d_wt_iou))
        
        print('Combined Testing mean dice: ', np.mean(test_dice))
        print('Combined Testing mean iou: ', np.mean(test_iou))
        print('Combined Testing NET, Edema, ET dice: ', np.mean(test_class_dice, axis = 0))
        print('Combined Testing NET, Edema, ET iou: ', np.mean(test_class_iou, axis = 0))
        print('Combined Testing TC dice: ', np.mean(test_tc_dice))
        print('Combined Testing TC iou: ', np.mean(test_tc_iou))
        print('Combined Testing whole dice: ', np.mean(test_whole_dice))
        print('Combined Testing whole iou: ', np.mean(test_whole_iou))
        
        print('test dice: ', np.mean(t_dice))