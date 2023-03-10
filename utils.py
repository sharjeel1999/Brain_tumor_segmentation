import torch.nn as nn
import torch

def dice_coeff(pred, target):
    #print('pred and target shapes: ', pred.shape, ' ', target.shape)
    smooth = 0.001
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    #print('reshaped shapes: ', m1.shape, ' ', m2.shape)
    intersection = (m1 * m2).sum()
    
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    iou = (intersection + smooth) / (m1.sum() + m2.sum() - intersection + smooth)
    return dice, iou


def class_dice(pred_class, target, epsilon = 1e-6):
    num_classes = len(torch.unique(target))
    #pred_class = torch.argmax(pred, dim = 1)
    #pred_class = torch.argmax(pred.squeeze(), dim=1).detach().cpu().numpy()
    dice = torch.ones(num_classes-1)
    dscore = torch.ones(num_classes-1)
    iou_score = torch.ones(num_classes-1)
    for c in range(0, num_classes):
        p = (pred_class == c)
        t = (target == c)
        #print('p shape: ', p.shape)
        #print('t shape: ', t.shape)
        dc, iou = dice_coeff(p, t)
        #print('dc done')
        dice[c-1] = 1 - dc
        dscore[c-1] = dc
        iou_score[c-1] = iou
        #print('appended')
        dl = torch.sum(dice)
        ds = torch.mean(dscore)
        ious = torch.mean(iou_score)
        
    return ds, ious

def loss_segmentation(pred, target):

    lossf = nn.CrossEntropyLoss()#weight = weights(pred, target).cuda())
    ce = lossf(pred, target)
    
    pred = torch.argmax(pred, dim = 1)
    #print('pred shape: ', pred.shape)
    #print('pred unique: ', torch.unique(pred))
    dsc, ious = class_dice(pred, target)

    loss = ce# + (1-ious)**2
    
    return loss, dsc, ious


def loss_detection(pred, target):
    lossf = nn.CrossEntropyLoss()
    
    ce = lossf(pred, target)
    
    return ce

