import torch
import torch.nn as nn
import random
import numpy as np
import os
from runs import Single_pass_initial
from utils import loss_segmentation, loss_detection, dice_coeff, class_dice
from torch.autograd import Variable
import time
from tqdm import tqdm

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
        
        #print('input 2 shape: ', input_2.shape)
        input_2 = torch.unsqueeze((input_2), dim = 1)
        gt_mask = torch.squeeze(gt_mask, dim = 1)
        
        out_2d = self.encoder_2d(input_1)
        out_3d = self.encoder_3d(input_2)
        combined_features = torch.cat((out_2d, out_3d), dim = 1)
        
        dec_out = self.decoder(combined_features)
        #print('gt mask: ', gt_mask.shape, torch.unique(gt_mask))
        loss, dsc, iou = loss_segmentation(dec_out, gt_mask)
        
        metrics = {}
        metrics['dice'] = dsc
        metrics['iou'] = iou
        
        return loss, metrics
    
    def Single_pass_regularization(self, input_1, input_2, optimizer_gen, optimizer_disc, mode):
        EPS = 1e-15
        Tensor = torch.cuda.FloatTensor
        input_2 = torch.unsqueeze((input_2), dim = 1)
        
        self.encoder_2d.eval()
        self.encoder_3d.eval()
        if mode == 'train':
            self.discriminator1.train()
            self.discriminator2.train()
        else:
            self.discriminator1.eval()
            self.discriminator2.eval()
        
        out_2d = self.encoder_2d(input_1)
        out_3d = self.encoder_3d(input_2)
        
        z = Variable(Tensor(np.random.normal(0, 1, (input_1.shape[0], out_2d.shape[1], out_2d.shape[2], out_2d.shape[3])))).cuda()
        
        disc_out_1_fake = self.discriminator1(out_2d)
        disc_out_2_fake = self.discriminator2(out_3d)
        
        disc_out_1 = self.discriminator1(z)
        disc_out_2 = self.discriminator2(z)
        
        #print('first part: ', torch.log(disc_out_1 + EPS))
        #print('second part: ', torch.log(1 - disc_out_1_fake + EPS))
        D_loss_1 = -torch.mean(torch.log(disc_out_1 + EPS) + torch.log(1 - disc_out_1_fake + EPS))
        D_loss_2 = -torch.mean(torch.log(disc_out_2 + EPS) + torch.log(1 - disc_out_2_fake + EPS))
        #print('disc 1: ', D_loss_1.item())
        #print('disc 2: ', D_loss_2.item())
        tot_disc_loss = D_loss_1 + D_loss_2
        #print('tot disc loss: ', tot_disc_loss.item())
        if mode == 'train':
            optimizer_disc.zero_grad()
            tot_disc_loss.backward()
            optimizer_disc.step()
        
        if mode == 'train':
            self.encoder_2d.train()
            self.encoder_3d.train()
        else:
            self.encoder_2d.eval()
            self.encoder_3d.eval()
            
        self.discriminator1.eval()
        self.discriminator2.eval()
        
        out_2d = self.encoder_2d(input_1)
        out_3d = self.encoder_3d(input_2)
        
        disc_out_1_fakee = self.discriminator1(out_2d)
        disc_out_2_fakee = self.discriminator2(out_3d)
        
        G_loss_1 = -torch.mean(torch.log(disc_out_1_fakee + EPS))
        G_loss_2 = -torch.mean(torch.log(disc_out_2_fakee + EPS))
        tot_gen_loss = G_loss_1 + G_loss_2
        #print('tot gen loss: ', tot_gen_loss.item())
        if mode == 'train':
            optimizer_gen.zero_grad()
            tot_gen_loss.backward()
            optimizer_gen.step()
        
        ONES = Variable(Tensor(input_1.shape[0], 1).fill_(1.0), requires_grad=False).long()
        ZEROS = Variable(Tensor(input_1.shape[0], 1).fill_(0.0), requires_grad=False).long()
        
        ONES = torch.squeeze(ONES, dim = 1)
        ZEROS = torch.squeeze(ZEROS, dim = 1)
        
        # det_loss1 = loss_detection(disc_out_1, ONES)
        # det_loss2 = loss_detection(disc_out_2, ONES)
        # det_loss3 = loss_detection(disc_out_1_fake, ZEROS)
        # det_loss4 = loss_detection(disc_out_2_fake, ZEROS)
        
        disc_out_1 = torch.argmax(disc_out_1, dim = 1)
        disc_out_2 = torch.argmax(disc_out_2, dim = 1)
        disc_out_1_fake = torch.argmax(disc_out_1_fake, dim = 1)
        disc_out_2_fake = torch.argmax(disc_out_2_fake, dim = 1)
        
        acc1 = (sum(disc_out_1 == ZEROS).item())/disc_out_1.shape[0]
        acc2 = (sum(disc_out_2 == ZEROS).item())/disc_out_2.shape[0]
        acc3 = (sum(disc_out_1_fake == ONES).item())/disc_out_1_fake.shape[0]
        acc4 = (sum(disc_out_1_fake == ONES).item())/disc_out_1_fake.shape[0]
        acc = np.mean([acc1, acc2, acc3, acc4])
        
        
        
        return tot_gen_loss, tot_disc_loss, acc
    
    def Single_pass_complete(self, input_1, input_2, gt_mask):
        input_2 = torch.unsqueeze((input_2), dim = 1)
        gt_mask = torch.squeeze(gt_mask, dim = 1)
        out_2d = self.encoder_2d(input_1)
        out_3d = self.encoder_3d(input_2)
        
        combined_features = torch.cat((out_2d, out_3d), dim = 1)
        
        dec_out = self.decoder(combined_features)
        
        seg_loss, dsc, iou = loss_segmentation(dec_out, gt_mask)
        
        Tensor = torch.cuda.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, (input_1.shape[0], out_2d.shape[1], out_2d.shape[2], out_2d.shape[3])))).cuda()
        disc_out_1 = self.discriminator1(out_2d)
        disc_out_2 = self.discriminator2(out_3d)
        disc_out_1_fake = self.discriminator1(z)
        disc_out_2_fake = self.discriminator2(z)
        
        ONES = Variable(Tensor(input_1.shape[0], 1).fill_(1.0), requires_grad=False).long()
        ZEROS = Variable(Tensor(input_1.shape[0], 1).fill_(0.0), requires_grad=False).long()
        ONES = torch.squeeze(ONES, dim = 1)
        ZEROS = torch.squeeze(ZEROS, dim = 1)
        
        det_loss1 = loss_detection(disc_out_1, ONES)
        det_loss2 = loss_detection(disc_out_2, ONES)
        det_loss3 = loss_detection(disc_out_1_fake, ZEROS)
        det_loss4 = loss_detection(disc_out_2_fake, ZEROS)
        
        loss = seg_loss + det_loss1 + det_loss2 + det_loss3 + det_loss4
        
        return loss, dsc, iou
    
    def train_loop(self, num_epochs, base_lr, train_loader, val_loader):
        
        optimizer = torch.optim.AdamW(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.decoder.parameters()), lr = base_lr, weight_decay = 1e-5)
        dice_latch = 0
        
        #encoder2d = self.encoder_2d
        #encoder3d = self.encoder_3d
        #decoder = self.decoder
        
        for epoch in range(num_epochs):
            train_loss = []
            train_dice = []
            train_iou = []
            
            self.encoder_2d.train()
            self.encoder_3d.train()
            self.decoder.train()
            
            for sample in train_loader:
                #start = time.time()
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                optimizer.zero_grad()
                loss, metrics = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
                train_iou.append(metrics['iou'].detach().cpu().numpy())
                train_dice.append(metrics['dice'].detach().cpu().numpy())
                #end = time.time()
                #print('time: ', end-start)
                
                
            print('Initial Train Loss: ', np.mean(train_loss))
            print('Initial Train Dice: ', np.mean(train_dice))
            print('Initial Train IoU: ', np.mean(train_iou))
            
            
            val_loss = []
            val_dice = []
            val_iou = []
            
            self.encoder_2d.eval()
            self.encoder_3d.eval()
            self.decoder.eval()
            
            for sample in val_loader:
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                with torch.no_grad():
                    loss, metrics = self.Single_pass_initial(input1.float(), input2.float(), gt_masks.long())
                
                val_loss.append(loss.item())
                val_dice.append(metrics['dice'].detach().cpu().numpy())
                val_iou.append(metrics['iou'].detach().cpu().numpy())
                
            print('Initial Validation Loss: ', np.mean(val_loss))
            print('Initial Validation Dice: ', np.mean(val_dice))
            print('Initial Validation IoU: ', np.mean(val_iou))
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
                f.write(f'Train Loss: {np.mean(train_loss)} Train IOU: {np.mean(train_iou)} Train Dice: {np.mean(train_dice)}')
                f.write('\n')
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation IOU: {np.mean(val_iou)} Validation Dice: {np.mean(val_dice)}')
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
        
        optimizer_gen = torch.optim.AdamW(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()), lr = base_lr, weight_decay = 1e-5)
        optimizer_disc = torch.optim.AdamW(list(self.discriminator1.parameters()) + list(self.discriminator2.parameters()), lr = base_lr, weight_decay = 1e-5)
        
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
                
                #optimizer_gen.zero_grad()
                #optimizer_disc.zero_grad()
                tot_gen_loss, tot_disc_loss, acc = self.Single_pass_regularization(input1.float(), input2.float(), optimizer_gen, optimizer_disc, 'train')
                #print('Acc: ', acc)
                
                # tot_disc_loss.backward()
                # optimizer_disc.step()
                
                # tot_gen_loss.backward()
                # optimizer_gen.step()
                
                
                tl = tot_disc_loss.item() + tot_gen_loss.item()
                train_loss.append(tl)
                train_acc.append(acc)
                
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
                    tot_gen_loss, tot_disc_loss, acc = self.Single_pass_regularization(input1.float(), input2.float(), optimizer_gen, optimizer_disc, 'val')
                
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
                f.write(f'Train Loss: {np.mean(train_loss)} Train Accuracy: {np.mean(train_acc)}')
                f.write('\n')
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation IOU: {np.mean(val_acc)}')
                f.write('\n')
                f.write('\n')
            
    
    def Combined_loop(self, num_epochs, base_lr, train_loader, val_loader):
        save_encoder12 = 'Encoder2D.pth'
        save_encoder22 = 'Encoder3D.pth'
        save_discriminator12 = 'Discriminator1.pth'
        save_discriminator22 = 'Discriminator2.pth'
        save_decoder2 = 'Decoder.pth'
        
        self.encoder_2d.load_state_dict(torch.load(os.path.join(self.weight_save_path[1], save_encoder12)))
        self.encoder_3d.load_state_dict(torch.load(os.path.join(self.weight_save_path[1], save_encoder22)))
        self.discriminator1.load_state_dict(torch.load(os.path.join(self.weight_save_path[1], save_discriminator12)))
        self.discriminator2.load_state_dict(torch.load(os.path.join(self.weight_save_path[1], save_discriminator22)))
        self.decoder.load_state_dict(torch.load(os.path.join(self.weight_save_path[0], save_decoder2)))
        
        optimizer = torch.optim.AdamW(list(self.encoder_2d.parameters()) + list(self.encoder_3d.parameters()) + list(self.discriminator_1.parameters()) + list(self.discriminator_2.parameters()) + list(self.decoder.parameters()), lr = base_lr, weight_decay = 1e-5)
        
        dice_latch = 0
        
        for epoch in range(num_epochs):
            self.encoder_2d.train()
            self.encoder_3d.train()
            self.discriminator1.train()
            self.discriminator2.train()
            self.decoder.train()
            
            train_loss = []
            train_dice = []
            train_iou = []
            
            for sample in train_loader:
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                optimizer.zero_grad()
                loss, dice, iou = self.Single_pass_complete(input1, input2, gt_masks.long())
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
                train_dice.append(dice.detach().cpu().numpy())
                train_iou.append(iou.detach().cpu().numpy())
                
            print('Combined Training Loss: ', np.mean(train_loss))
            print('Combined Training dice: ', np.mean(train_dice))
            print('Combined Training iou: ', np.mean(train_iou))
            print('\n')
            
            val_loss = []
            val_dice = []
            val_iou = []
            
            for sample in val_loader:
                input1, input2, gt_masks = sample
                input1, input2, gt_masks = input1.cuda(), input2.cuda(), gt_masks.cuda()
                
                with torch.no_grad():
                    loss, dice, iou = self.Single_pass_complete(input1, input2, gt_masks)
                    
                val_loss.append(loss.item())
                val_dice.append(dice.detach().cpu().numpy())
                val_iou.append(iou.detach().cpu().numpy())
                
            print('Combined Validation Loss: ', np.mean(val_loss))
            print('Combined Validation dice: ', np.mean(val_dice))
            print('Combined Validation iou: ', np.mean(val_iou))
            
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
            
            with open(self.record_save_path[2], 'a') as f:
                f.write(f'Train Loss: {np.mean(train_loss)} Train Accuracy: {np.mean(train_dice)}')
                f.write('\n')
                f.write(f'Validation Loss: {np.mean(val_loss)} Validation IOU: {np.mean(val_dice)}')
                f.write('\n')
                f.write('\n')
        