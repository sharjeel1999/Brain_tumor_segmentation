import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, average_precision_score
from hausdorff import hd95

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:

        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).cuda()
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)

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


def class_dice(pred_class, target, tot_classes, epsilon = 1e-6):
    num_classes = torch.unique(target)
    #print('num classes: ', num_classes)
    #pred_class = torch.argmax(pred, dim = 1)
    #pred_class = torch.argmax(pred.squeeze(), dim=1).detach().cpu().numpy()
    dice = torch.zeros(3)
    dscore = torch.zeros(3)
    iou_score = torch.zeros(3)
    ds = 0
    ious = 0
    for c in range(1, tot_classes):
        if c in num_classes:
            #print('c: ', c)
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
        
    return ds, dscore, ious, iou_score

def weights(pred, target, epsilon = 1e-6):
    num_classes = 4
    pred_class = torch.argmax(pred, dim = 1)
    #pred_class = torch.argmax(pred.squeeze(), dim=1).detach().cpu().numpy()
    dice = np.ones(num_classes)
    tot = 0
    for c in range(num_classes):
        t = (target == c).sum()
        tot = tot + t
        #print(t.shape)
        dice[c] = t

    dice = dice/dice.sum()
    dice = 1 - dice
    
    return torch.from_numpy(dice).float()

def loss_segmentation(pred, target):
    #print('pred/target shape: ', pred.shape, target.shape)
    #print('pred/target max; ', torch.max(pred), torch.max(target), torch.unique(target))
    
    lossf = nn.CrossEntropyLoss(weight = weights(pred, target).cuda())
    #w = np.array([0.001503597, 0.5840496, 0.20685249, 1])
    #lossf = nn.CrossEntropyLoss(weight = torch.from_numpy(w).float().cuda())
    ce = lossf(pred, target)
    
    # print('target/pred dtypes: ', target.dtype, pred.dtype)
    dsc_l = dice_loss(target, pred)
    iou_l = jaccard_loss(target, pred)
    tl = tversky_loss(target, pred, alpha = 0.7, beta = 0.3)
    
    
    pred_et = pred.clone()
    target_et = target.clone()
    pred_et_0 = torch.unsqueeze(pred_et[:, 0, :, :], dim = 1)
    pred_et_3 = torch.unsqueeze(pred_et[:, 3, :, :], dim = 1)
    pred_et_only = torch.cat((pred_et_0, pred_et_3), dim = 1)
    # print('pred channels only: ', pred_et_only.shape)
    # print('pred/target shape: ', pred_et.shape, target_et.shape)

    target_et[target_et < 3] = 0
    target_et[target_et == 3] = 1
    
    et_dice = dice_loss(target_et, pred_et_only)
    et_iou = jaccard_loss(target_et, pred_et_only)
    
    pred = torch.argmax(pred, dim = 1)
    #print('pred shape: ', pred.shape)
    #print('pred unique: ', torch.unique(pred))
    dsc, class_dsc, ious, class_iou = class_dice(pred, target, 4)
    
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    #print('tc unique: ', torch.unique(target_tc))
    tc_dice, tc_iou = dice_coeff(pred_tc, target_tc)
    #tc_dice, _, tc_iou, _ = class_dice(pred_tc, target_tc, 2)
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    #print('whole unique: ', torch.unique(target_whole))
    whole_dice, whole_iou = dice_coeff(pred_whole, target_whole)
    #whole_dice, _, whole_iou, _ = class_dice(pred_whole, target_whole, 2)
    #print((et_dice))
    
    loss = 0.5*(dsc_l + iou_l) + tl**(2)# + (1 - et_dice) + (1 - et_iou)
    #loss = tl
    
    return loss, dsc, class_dsc, ious, class_iou, [tc_dice, tc_iou], [whole_dice, whole_iou]


def loss_detection(pred, target):
    lossf = nn.CrossEntropyLoss()
    
    ce = lossf(pred, target)
    
    return ce

def prec_rec(pred, gt):
    tn, fp, fn, tp = confusion_matrix(pred.ravel(), gt.ravel()).ravel()
    
    prec = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    specificity = (tn)/(tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    
    iou = (tp)/(tp + fp + fn)
    
    return prec, recall, specificity, accuracy, iou, fp, fn

def other_metrics(pred, target):
    
    pred_et = pred.clone()
    target_et = target.clone()
    pred_et[pred_et != 3] = 0
    pred_et[pred_et == 3] = 1
    target_et[target_et != 3] = 0
    target_et[target_et == 3] = 1
    prec_et, recall_et, specificity_et, acc_et, _, fp_et, fn_et = prec_rec(pred_et.detach().cpu().numpy(), target_et.detach().cpu().numpy())
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    prec_tc, recall_tc, specificity_tc, acc_tc, _, fp_tc, fn_tc = prec_rec(pred_tc.detach().cpu().numpy(), target_tc.detach().cpu().numpy())
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    prec_whole, recall_whole, specificity_whole, acc_whole, _, fp_whole, fn_whole = prec_rec(pred_whole.detach().cpu().numpy(), target_whole.detach().cpu().numpy())
    
    return [recall_et, recall_tc, recall_whole], [specificity_et, specificity_tc, specificity_whole], [fp_et, fp_tc, fp_whole], [fn_et, fn_tc, fn_whole]

def test_scores_3d(pred, target):
    #print('pred shape; ', torch.unique(pred))
    dsc, class_dsc, ious, class_iou = class_dice(pred, target, 4)
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    
    tc_dice, tc_iou = dice_coeff(pred_tc, target_tc)
    #tc_dice, _, tc_iou, _ = class_dice(pred_tc, target_tc, 2)
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    #print('whole unique: ', torch.unique(target_whole))
    whole_dice, whole_iou = dice_coeff(pred_whole, target_whole)
    #whole_dice, _, whole_iou, _ = class_dice(pred_whole, target_whole, 2)
    
    return dsc, class_dsc, ious, class_iou, [tc_dice, tc_iou], [whole_dice, whole_iou]

def hausdorf_distance(pred, target):
    hd95_dict = {}
    hd95_dict['mean'] = 0.0
    hd95_dict['ET'] = 0.0
    hd95_dict['TC'] = 0.0
    hd95_dict['WT'] = 0.0
    
    pred_et = pred.clone()
    target_et = target.clone()
    pred_et[pred_et != 3] = 0
    pred_et[pred_et == 3] = 1
    target_et[target_et != 3] = 0
    target_et[target_et == 3] = 1
    
    if len(torch.unique(pred_et)) == 2:
        hd95_dict['ET'] = hd95(pred_et.detach().cpu().numpy(), target_et.detach().cpu().numpy())
    else:
        hd95_dict['ET'] = 0
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    
    if len(torch.unique(pred_tc)) == 2:
        hd95_dict['TC'] = hd95(pred_tc.detach().cpu().numpy(), target_tc.detach().cpu().numpy())
    else:
        hd95_dict['TC'] = 0
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    
    if len(torch.unique(pred_whole)) == 2:
        hd95_dict['WT'] = hd95(pred_whole.detach().cpu().numpy(), target_whole.detach().cpu().numpy())
    else:
        hd95_dict['WT'] = 0
    
    return hd95_dict

def dice_3d(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:

        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()

        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return dice_loss


