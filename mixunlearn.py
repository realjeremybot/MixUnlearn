import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional
import PIL.Image
import random
import logging
# from mmcv.runner import auto_fp16, force_fp32, load_checkpoint
from torchvision.utils import save_image
from openmixup.utils import print_log
from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments import cutmix, mixup
from ..utils import PlotTensor
import copy
import torch.nn as nn

def sharpen(p, T=0.3):
    pt = p**(1/T)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    return targets_u


def extractor_loss_functioncosine(s_e, s_e_teacher, u_e, u_e_teacher, temp, lam):
    
    loss1 = nn.CosineEmbeddingLoss(reduction='none')

    COS1 = (1-lam)*loss1(s_e, s_e_teacher, torch.ones(s_e.shape[0]).cuda())
    
    loss2 = nn.CosineEmbeddingLoss(reduction='none')

    COS2 = (lam)*loss2(u_e, u_e_teacher, torch.ones(u_e.shape[0]).cuda())
    
    SIM = torch.sum(torch.log(torch.exp(COS1)/torch.sum(torch.exp(COS2/temp))))
    
    return SIM

# def extractor_loss_functioncosine_k(s_e, s_e_teacher, u_e, u_e_teacher, temp, lam):
    
#     loss1 = nn.CosineEmbeddingLoss(reduction='none')

#     COS1 = (1-lam)*loss1(s_e, s_e_teacher, torch.ones(s_e.shape[0]).cuda())
    
#     loss2 = nn.CosineEmbeddingLoss(reduction='none')

#     COS2 = (lam)*loss2(u_e, u_e_teacher, torch.ones(u_e.shape[0]).cuda())
    
#     SIM = torch.sum(torch.log(torch.exp(COS1/temp)/torch.sum(torch.exp(COS2))))
    
#     return SIM

@MODELS.register_module
class ConUnlearnAdAutoMix(BaseModel):
    def __init__(self,
                 init_model,
                                  
                 temp,
                 temp_adv,
                 sharpen_T,
                 mix_block=None,
                 affine_da=None,
                 head_mix=None,
                 head_one=None,
                 head_mix_k=None,
                 head_one_k=None,
                 head_weights=dict(decent_weight=[], accent_weight=[],
                                   head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
                 alpha=0.75,
#                  thres = 0.5,
                 mix_samples=3,
                 is_random=False,
                 momentum=0.999,
                 lam_margin=-1,
                 switch_off=0.,
                 adv_interval=2,
                 mixup_radio=0.5,

                 beta_radio=0.3,
                 head_one_mix=False,
                 head_ensemble=False,
                 save=False,
                 save_name='MixedSamples',
                 debug=False,
                 mix_shuffle_no_repeat=False,
                 pretrained=None,
                 pretrained_k=None,
                 init_cfg=None,
                 **kwargs):
        super(ConUnlearnAdAutoMix, self).__init__(init_cfg, **kwargs)
        # basic params
        self.temp = temp
        self.temp_adv = temp_adv
        self.sharpen_T = sharpen_T
        self.alpha = float(alpha)
        self.mix_samples = int(mix_samples)
        self.co_mix = 2
        self.is_random = bool(is_random)
        self.momentum = float(momentum)
        self.base_momentum = float(momentum)
        self.lam_margin = float(lam_margin) if float(lam_margin) > 0 else 0
        self.mixup_radio = float(mixup_radio)
        self.beta_radio = float(beta_radio)
        self.switch_off = float(switch_off) if float(switch_off) > 0 else 0
        self.head_one_mix = bool(head_one_mix)
        self.head_ensemble = bool(head_ensemble)
        self.save = bool(save)
        self.save_name = str(save_name)
        self.ploter = PlotTensor(apply_inv=True)
        self.debug = bool(debug)
        self.mix_shuffle_no_repeat = bool(mix_shuffle_no_repeat)
        self.adv_interval = int(adv_interval)
        self.iter = 0
        self.i = 0
#         self.thres = thres
        assert 0 <= self.momentum and self.lam_margin < 1


        assert head_mix is None or isinstance(head_mix, dict)
        assert head_one is None or isinstance(head_one, dict)
        assert head_mix_k is None or isinstance(head_mix_k, dict)
        assert head_one_k is None or isinstance(head_one_k, dict)
        head_mix_k = head_mix if head_mix_k is None else head_mix_k
        head_one_k = head_one if head_one_k is None else head_one_k
        # mixblock
        self.mix_block = builder.build_head(mix_block).cuda()
        
        # backbone
        self.backbone_q = init_model.cuda()

        print("sharpen mixup loss")
        
        self.backbone_k = copy.deepcopy(init_model)
        
        
        # mixup cls head
        assert "head_mix_q" in head_weights.keys() and "head_mix_k" in head_weights.keys()
        self.head_mix_q = builder.build_head(head_mix)
        self.head_mix_k = builder.build_head(head_mix_k)
        # onehot cls head
        if "head_one_q" in head_weights.keys():
            self.head_one_q = builder.build_head(head_one)
        else:
            self.head_one_q = None
        if "head_one_k" in head_weights.keys():
            self.head_one_k = builder.build_head(head_one_k)
        else:
            self.head_one_k = None
        # for feature extract
        self.head = self.head_one_k if self.head_one_k is not None else self.head_one_q
        # onehot and mixup heads for training
        self.weight_mix_q = head_weights.get("head_mix_q", 1.)
        self.weight_mix_k = head_weights.get("head_mix_k", 1.)
        self.weight_one_q = head_weights.get("head_one_q", 1.)
        assert self.weight_mix_q > 0 and (self.weight_mix_k > 0 or backbone_k is not None)
        self.head_weights = head_weights
        self.head_weights['decent_weight'] = head_weights.get("decent_weight", list())
        self.head_weights['accent_weight'] = head_weights.get("accent_weight", list())
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine

        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):

        # init mixblock
        if self.mix_block is not None:
            self.mix_block.init_weights(init_linear='normal')
        
        
        # copy backbone param from q to k
        if pretrained_k is None and self.momentum < 1:
            for param_q, param_k in zip(self.backbone_q.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False  # stop grad k

        # init head
        if self.head_mix_q is not None:
            self.head_mix_q.init_weights()
        if self.head_one_q is not None:
            self.head_one_q.init_weights()

        # copy head one param from q to k
        if (self.head_one_q is not None and self.head_one_k is not None) and \
                (pretrained_k is None and self.momentum < 1):
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(),
                                                self.head_one_k.parameters()):
                param_one_k.data.copy_(param_one_q.data)
                param_one_k.requires_grad = False  # stop grad k

        # copy head mix param from q to k
        if (self.head_mix_q is not None and self.head_mix_k is not None) and \
                (pretrained_k is None and self.momentum < 1):
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(),
                                                self.head_mix_k.parameters()):
                param_mix_k.data.copy_(param_mix_q.data)
                param_mix_k.requires_grad = False  # stop grad k

                
    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the k form q by hook, including the backbone and heads """
        # we don't update q to k when momentum > 1
        if self.momentum >= 1.:
            return
        # update k's backbone and cls head from q
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

                

    def forward_train(self, original_feat_extractor,unlearn_img, retain_img, gt_label, unlearn_student, teacher,**kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of a batch of images, (N, C, H, W).
            gt_label (Tensor): Groundtruth onehot labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        batch_size = retain_img.size(0)
#         self._update_loss_weights()


        index_bb = torch.randperm(batch_size).cuda()
        index_mb = torch.arange(batch_size).cuda()
      
        
#         print(self.alpha)
        l_0 = np.random.beta(self.alpha, self.alpha)

#         l_0 = max(l_0, 1-l_0)
        
        l_1 = np.random.beta(self.alpha, self.alpha)

#         l_1 = max(l_1, 1-l_1)
        
        lam = [l_0, l_1]
        index = [index_mb, index_bb]

        
        
        ori_retain_y, _ =  teacher(retain_img)
        
        
        
        ori_unlearn_y, _ = teacher(unlearn_img)
        
        
#         random_unlearn_y, _ = unlearn_student(unlearn_img)
#         random_retain_y, _ = unlearn_student(retain_img)

        
        unlearn_feature = original_feat_extractor(unlearn_img, if_return_feat=True)
        retain_feature = original_feat_extractor(retain_img, if_return_feat=True)

        results = self.mixup(unlearn_img, retain_img, lam, index, unlearn_feature, retain_feature)
        

        
        if self.iter % self.adv_interval == 0:
            cos_simi_weight, loss_mix_k = self.forward_k(unlearn_img, retain_img, results["img_mix_mb"], ori_unlearn_y, ori_retain_y, index[0], lam[0], temp=self.temp_adv)
            print("forward k")
        else:
            loss_mix_k = 0

        loss_one_q, loss_mix_q = self.forward_q(unlearn_img, retain_img, results["img_mix_bb"], ori_unlearn_y, ori_retain_y, index[1], lam[1], temp=self.temp)

            
        #  loss summary
        losses = {
            'loss': loss_mix_q,
        }
        

        if loss_mix_k is not None and self.weight_mix_k > 0:
            losses["loss"] += loss_mix_k * self.weight_mix_k
            print(self.weight_mix_k)

        self.iter += 1
        
        
        return losses

    

    def forward_q(self, unlearn_img, retain_img, mixed_x, ori_unlearn_y, ori_retain_y, index, lam, temp):
        """
        Args:
            x (Tensor): Input of a batch of images, (N, C, H, W).
            mixed_x (Tensor): Mixup images of x, (N, C, H, W).
            y (Tensor): Groundtruth onehot labels, coresponding to x.
            index (List): Input list of shuffle index (tensor) for mixup.
            lam (List): Input list of lambda (scalar).

        Returns:
            dict[str, Tensor]: loss_one_q and loss_mix_q are losses from q.
        """


        loss_mix_q = None

        out_mix_q, _ = self.backbone_q(mixed_x)
        pred_mix_q = torch.softmax(out_mix_q, dim=-1)


        ori_unlearn_y = sharpen(torch.softmax(ori_unlearn_y, dim=-1), self.sharpen_T)
        ori_retain_y = sharpen(torch.softmax(ori_retain_y, dim=-1), self.sharpen_T)        
        
#         y_mix_q = (lam * F.softmax(random_unlearn_y, dim=-1) + (1-lam) * F.softmax(ori_retain_y[index], dim=-1)).detach()
        
        
        
        loss_mix_q = extractor_loss_functioncosine(pred_mix_q, ori_retain_y, pred_mix_q, ori_unlearn_y, temp, lam)

        if torch.isnan(loss_mix_q):
            print_log("Warming NAN in loss_mix_q. Please use FP32!", logger='root')
            loss_mix_q = dict(loss=None)

        return 0, loss_mix_q

    
    
    
    
    def forward_k(self, unlearn_img, retain_img, mixed_x, ori_unlearn_y, ori_retain_y, index, lam, temp):
        """ forward k with the mixup sample """
        loss_mix_k = dict(loss=None, pre_mix_loss=None)
        cos_simi_weight = 0.0

        pred_mix_k, _ = self.backbone_k(mixed_x)


        # mixup loss
        pred_mix_k = torch.softmax(pred_mix_k, dim=-1)

#         y_mix_k = (lam * F.softmax(ori_unlearn_y, dim=-1) + (1-lam) * F.softmax(random_retain_y[index], dim=-1)).detach()



        ori_unlearn_y = sharpen(torch.softmax(ori_unlearn_y, dim=-1), self.sharpen_T)
        ori_retain_y = sharpen(torch.softmax(ori_retain_y, dim=-1), self.sharpen_T)
    
        loss_mix_k = -extractor_loss_functioncosine(pred_mix_k, ori_retain_y, pred_mix_k, ori_unlearn_y, temp, lam)
        print(temp)
        if torch.isnan(loss_mix_k):
            print_log("Warming NAN in loss_mix_k. Please use FP32!", logger='root')
            loss_mix_k = None

        cos_simi_weight = 0
                

        
        return cos_simi_weight, loss_mix_k



    def mixup(self, unlearn_img, retain_img, lam, index, unlearn_feature, retain_feature):
        """ pixel-wise input space mixup"""
#         results = dict()
#         # lam info
#         lam_mb = lam[0]  # lam is a scalar
#         lam_bb = lam[1]

#         results["vallina_img_mix_mb"] = \
#             unlearn_img * lam_mb + retain_img[index[0], :] * (1-lam_mb)
#         results["vallina_img_mix_bb"] = \
#             unlearn_img * lam_bb + retain_img[index[1], :] * (1-lam_bb)
        
        
        results = dict()
        # lam info
        lam_mb = lam[0]  # lam is a scalar
        lam_bb = lam[1]

#         results["img_mix_mb"] = \
#             unlearn_img * lam_mb + retain_img[index[0], :] * (1-lam_mb)
#         results["img_mix_bb"] = \
#             unlearn_img * lam_bb + retain_img[index[1], :] * (1-lam_bb)

#         rand_int = random.uniform(0, 1)

#         if rand_int > self.thres:
# #             print("direct return")
#             return results
            
#         else:
#             test = 1
# #             print("learn mix")
    
        # get mixup mask
        mb = [unlearn_feature, retain_feature[index[0], :]]
        bb = [unlearn_feature, retain_feature[index[1], :]]

        mask_mb = self.mix_block(mb, lam_mb)
        mask_bb = self.mix_block(bb, lam_bb)

        if self.debug:
            results["debug_plot"] = mask_mb["debug_plot"]
        else:
            results["debug_plot"] = None

        mask_mb = mask_mb["mask"]
        mask_bb = mask_bb["mask"].clone().detach()

#         # lam_margin for backbone training
#         if self.lam_margin >= lam_bb or self.lam_margin >= 1 - lam_bb:
#             mask_bb[:, 0, :, :] = lam_bb
#             mask_bb[:, 1, :, :] = 1 - lam_bb
            
        # mix, apply mask on x and x_
        assert mask_mb.shape[1] == 2
        assert mask_mb.shape[2:] == unlearn_img.shape[2:], f"Invalid mask shape={mask_mb.shape}"
        results["img_mix_mb"] = \
            unlearn_img * mask_mb[:, 0, :, :].unsqueeze(1) + retain_img[index[0], :] * mask_mb[:, 1, :, :].unsqueeze(1)
        results["img_mix_bb"] = \
            unlearn_img * mask_bb[:, 0, :, :].unsqueeze(1) + retain_img[index[1], :] * mask_bb[:, 1, :, :].unsqueeze(1)
        
        
        
        
        return results

    
    def forward_vis(self, original_feat_extractor,unlearn_img, retain_img, gt_label, unlearn_student, **kwargs):
        batch_size = retain_img.size(0)


        index_bb = torch.randperm(batch_size).cuda()


        index_mb = torch.arange(batch_size).cuda()



        l_0 = 0.5
        

        l_1 = 0.5
        
        lam = [l_0, l_1]
        index = [index_mb, index_bb]

        
        
        retain_y, _ = self.backbone_q(retain_img)
        unlearn_y, _ = unlearn_student(unlearn_img)
        
        
        ori_y, _ = original_feat_extractor(unlearn_img)
        
        
        unlearn_feature = original_feat_extractor(unlearn_img, if_return_feat=True)
        retain_feature = original_feat_extractor(retain_img, if_return_feat=True)

        results = self.mixup(unlearn_img, retain_img, lam, index, unlearn_feature, retain_feature)
        
        return {'mix_bb': results["img_mix_bb"], 'mix_mb': results["img_mix_mb"]}

        
