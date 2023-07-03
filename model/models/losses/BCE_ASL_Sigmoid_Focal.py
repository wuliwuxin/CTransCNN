import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import convert_to_one_hot, weight_reduce_loss
# from platypus import NSGAII


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         pos_weight=None):
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    assert pred.dim() == label.dim()

    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred,
        label,
        weight=class_weight,
        pos_weight=pos_weight,
        reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def asymmetric_loss(pred,
                    target,
                    weight=None,
                    gamma_pos=1.0,
                    gamma_neg=4.0,
                    clip=0.05,
                    reduction='mean',
                    avg_factor=None,
                    use_sigmoid=True,
                    eps=1e-8):
    r"""asymmetric loss.

    Please refer to the `paper <https://arxiv.org/abs/2009.14119>`__ for
    details.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma_pos (float): positive focusing parameter. Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We usually set
            gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
           target.shape, 'pred and target should be in the same shape.'

    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = nn.functional.softmax(pred, dim=-1)

    target = target.type_as(pred)

    if clip and clip > 0:
        pt = (1 - pred_sigmoid +
              clip).clamp(max=1) * (1 - target) + pred_sigmoid * target
    else:
        pt = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    asymmetric_weight = (1 - pt).pow(gamma_pos * target + gamma_neg *
                                     (1 - target))
    loss = -torch.log(pt.clamp(min=eps)) * asymmetric_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    assert pred.shape == \
           target.shape, 'pred and target should be in the same shape.'
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def multilabel_categorical_crossentropy(y_true, y_pred, weight, reduction='none', avg_factor=None, ):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = F.binary_cross_entropy_with_logits(
        y_true,
        y_pred,
        weight=weight,
        pos_weight=None,
        reduction='none')
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if y_pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    # loss = neg_loss + pos_loss
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class BCE_ASL_Focal(nn.Module):
    """asymmetric loss.

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.
    """

    def __init__(self,
                 gamma=2.0,  
                 alpha=0.25, 
                 use_soft=False, 
                 class_weight=None,
                 pos_weight=None, 
                 # gamma_pos=1.0,
                 gamma_pos=0.0,
                 gamma_neg=4.0,
                 clip=0.05,
                 reduction='mean',
                 loss_weight=1.0,
                 use_sigmoid=True,  #
                 eps=1e-8):
        super(BCE_ASL_Focal, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.use_soft = use_soft
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
            target = convert_to_one_hot(target.view(-1, 1), pred.shape[-1])
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
        # loss_bce = self.loss_weight * multilabel_categorical_crossentropy(
        #     pred,
        #     target,
        #     weight,
        #     reduction=reduction,
        #     avg_factor=avg_factor)
        loss_asl = self.loss_weight * asymmetric_loss(
            pred,
            target,
            weight,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            clip=self.clip,
            reduction=reduction,
            avg_factor=avg_factor,
            use_sigmoid=self.use_sigmoid,
            eps=self.eps)
        loss_focal = self.loss_weight * sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)
        # pareto_loss = loss_asl
        pareto_loss = torch.stack([loss_asl, loss_focal], dim=0) 
        return pareto_loss
