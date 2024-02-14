# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from copy import deepcopy
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid,Store)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .segmentation import sigmoid_focal_loss as seg_sigmoid_focal_loss
from .deformable_transformer import build_deforamble_transformer
import copy
from torchmetrics.functional import pairwise_cosine_similarity
import geoopt


from models.pmath import dist_matrix, _dist_matrix , RiemannianGradient, poincare_mean, dist0
from functools import partial
from models import pmath

import numpy as np
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c
        
        self.clip_r = clip_r
        
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            #ForkedPdb().set_trace()
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac =  torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
            
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)
    
def contrastive_loss(x0, x1, tau, hyp_c):
        # x0 and x1 - positive pair
        # tau - temperature
        # hyp_c - hyperbolic curvature, "0" enables sphere mode

        if hyp_c == 0:
            # ForkedPdb().set_trace()
            # x = F.normalize(x0, dim=-1, p=2)
            # y = F.normalize(x1, dim=-1, p=2)
            # ForkedPdb().set_trace()
        # return 2-2*(x @ y.t())
            dist_f = lambda x, y: 2-2*(F.normalize(x, dim=-1, p=2) @ F.normalize(y, dim=-1, p=2).t())
            #dist_f = lambda x, y: x @ y.t()
        else:
            dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = dist_f(x0, x0) / tau - eye_mask
        logits01 = dist_f(x0, x1) / tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        stats = {
            "logits/min": logits01.min().item(),
            "logits/mean": logits01.mean().item(),
            "logits/max": logits01.max().item(),
            "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
        }
        return loss, stats
    
    
def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if x.dim() == 0:
        return x.unsqueeze(0).nonzero().unbind(1)
    return x.nonzero().unbind(1)
            
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, num_classes: int = 81, empty_weight: float = 0.1,args=None,use_focal=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    
    prob = inputs.sigmoid()
  
    W = torch.ones(num_classes, dtype=prob.dtype, layout=prob.layout, device=prob.device)
    W[-1] = empty_weight
    
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=W)
 
 
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
  
    return loss.mean(1).sum() / num_boxes
  



    
class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        
        
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
   
        self.args=args
      

        if self.args.use_hyperbolic:
            self.manifold = geoopt.PoincareBall(c=self.args.hyperbolic_c)
            
            # self.fc = nn.Linear(hidden_dim, int(hidden_dim/2))
            self.tpc=ToPoincare(c=self.args.hyperbolic_c,ball_dim=hidden_dim,riemannian=False,clip_r=self.args.clip_r)
            #self.tpc=ToPoincare(c=self.args.hyperbolic_c,ball_dim=args.hidden_dim,riemannian=False,clip_r=self.args.clip_r)
     
        
        
        
   
        self.class_embed = nn.Linear(hidden_dim, num_classes)
            
        
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
       

        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        
        # ForkedPdb().set_trace()
        # self.class_embed.bias.shape
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
      
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
          
                
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            
            self.transformer.decoder.bbox_embed = None
        
        # ForkedPdb().set_trace()
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
                
                
  

    def forward(self, samples: NestedTensor,relevant_matrix=None,eval=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        
        
        
        
        if self.args.use_hyperbolic:
            hyperbolic_emb=[]
        else:
            hyperbolic_emb=None
      
        

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # ForkedPdb().set_trace()
       
            outputs_class = self.class_embed[lvl](hs[lvl])
            
            
           
            if self.args.use_hyperbolic:
                
                if self.args.hyperbolic_c>0: 
                    hyperbolic_emb.append(self.tpc(hs[lvl]))  
                   
                else:
                    # print('oui')
                    if self.args.normalize:
                        normalize_weight = hs[lvl].weight.data.clone()
                        normalize_weight= F.normalize(normalize_weight, dim=1, p=2)
                    else:
                        normalize_weight=hs[lvl]
                    
                    hyperbolic_emb.append(normalize_weight)  
          

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
          
           
        
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        #######################################################################################
       
        
        if self.args.use_hyperbolic:
            hyperbolic_emb=torch.stack(hyperbolic_emb)
            
     
        #######################################################################################
    
            
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
                                       
      
        if self.args.use_hyperbolic:
            out['hyperbolic_emb']=hyperbolic_emb[-1]
        if eval and self.args.use_hyperbolic_temp and relevant_matrix is not None:
         
            out['relevant_matrix']=relevant_matrix
 
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, hyperbolic_emb=hyperbolic_emb)
                    

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
     
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, hyperbolic_emb=None):
        return [{'pred_logits': a, 'pred_boxes': c,'hyperbolic_emb':d}
                    for a, c,d in zip(outputs_class[:-1], outputs_coord[:-1],hyperbolic_emb[:-1])]

            


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25, empty_weight=0.1,args=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        
        self.args=args
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        self.empty_weight=empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        
        self.valid_classes=list(range(0,args.PREV_INTRODUCED_CLS+args.CUR_INTRODUCED_CLS))
        # ForkedPdb().set_trace()
        self.min_obj=-hidden_dim*math.log(0.9)
        

       
        self.iteration=0
        self.epoch=0
        
        if self.args.use_hyperbolic:
            self.hyperbolic_emb=Store(81, self.args.emb_per_class,cfg=args)
            self.loss_f = partial(contrastive_loss, tau=self.args.hyperbolic_temp, hyp_c=self.args.hyperbolic_c)
            self.loss_family = partial(contrastive_loss, tau=2*self.args.hyperbolic_temp, hyp_c=self.args.hyperbolic_c)
            
   
      
     
        if 'HIERARCHICAL' in args.dataset:
            if 't1' in args.train_set:
                self.label_mapping={0:0,1:0,\
                                    2:1,3:1,\
                                    4:2,5:2,6:2,\
                                   7:3,\
                                   8:4  ,9:4,\
                                   10:5, 11:5,\
                                    12:6 , 13:6,\
                                    14:7, 15:7,\
                                    16:8,
                                    17:9,18:9,
                                    19:10
                                   }
                self.family_grouping={0:torch.Tensor([0,1]).long(),\
                                    1:torch.Tensor([2,3]).long(),  \
                                    2:torch.Tensor([4,5,6]).long(),\
                                      3:torch.Tensor([7]).long(),\
                                      4:torch.Tensor([8,9]).long(),\
                                      5:torch.Tensor([10,11]).long(),\
                                      6:torch.Tensor([12,13]).long(),\
                                      7:torch.Tensor([14,15]).long(),\
                                      8:torch.Tensor([16]).long(),\
                                       9:torch.Tensor([17,18]).long(),\
                                      10:torch.Tensor([19]).long(),\
                                   }
            elif 't2' in args.train_set:
                
                self.label_mapping={0:0,1:0,      21:0,22:0,\
                                    2:1,3:1,      23:1,\
                                    4:2,5:2,6:2,  24:2,25:2, \
                                   7:3,           26:3,\
                                   8:4  ,9:4,     27:4,28:4,      \
                                   10:5, 11:5,       29:5   ,      \
                                    12:6 , 13:6,      30:6,31:6,32:6,      \
                                    14:7, 15:7,       33:7,34:7  ,   \
                                    16:8,           37:8,\
                                    17:9,18:9,         35:9,36:9,
                                    19:10,                     38:10,39:10,\
                                    20:11,
                                   }
                
                
                
                self.family_grouping={0:torch.Tensor([0,1,  21,22  ]).long(),\
                                    1:torch.Tensor([2,3,   23]).long(),  \
                                    2:torch.Tensor([4,5,6,    24,25]).long(),\
                                      3:torch.Tensor([7,   26]).long(),\
                                      4:torch.Tensor([8,9,   27,28]).long(),\
                                      5:torch.Tensor([10,11,   29]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32]).long(),\
                                      7:torch.Tensor([14,15,   33,34]).long(),\
                                      8:torch.Tensor([16,       37]).long(),\
                                       9:torch.Tensor([17,18,    35,36]).long(),\
                                      10:torch.Tensor([19,   38,39]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
            elif 't3' in args.train_set:
                
                self.label_mapping={0:0,1:0,            21:0,22:0,        40:0, 41:0,\
                                    2:1,3:1,            23:1,             42:1,\
                                    4:2,5:2,6:2,        24:2,25:2,        43:2,44:2,45:2, \
                                   7:3,                 26:3,             46:3,           \
                                   8:4  ,9:4,           27:4,28:4,        47:4,48:4,49:4, \
                                   10:5, 11:5,          29:5   ,          50:5,51:5,    \
                                    12:6 , 13:6,        30:6,31:6,32:6,   52:6,53:6,   \
                                    14:7, 15:7,       
                                    33:7,34:7  ,      54:7,              \
                                    16:8,               37:8,             55:8,56:8,           \
                                    17:9,18:9,          35:9,36:9,        57:9,      \
                                    19:10,              38:10,39:10,       58:10,59:10,   \
                                    20:11,
                                   }
                
                
                self.family_grouping={0:torch.Tensor([0,1,      21,22 ,      40,41 ]).long(),\
                                    1:torch.Tensor([2,3,       23,          42]).long(),  \
                                    2:torch.Tensor([4,5,6,     24,25,       43,44,45]).long(),\
                                      3:torch.Tensor([7,       26,          46]).long(),\
                                      4:torch.Tensor([8,9,     27,28,       47,48,49]).long(),\
                                      5:torch.Tensor([10,11,   29,          50,51]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32,    52,53]).long(),\
                                      7:torch.Tensor([14,15,   33,34,       54]).long(),\
                                      8:torch.Tensor([16,      37,          55,56]).long(),\
                                       9:torch.Tensor([17,18,  35,36,       57]).long(),\
                                      10:torch.Tensor([19,     38,39,        58,59]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
           
            elif 't4' in args.train_set:
                self.label_mapping={0:0,1:0,            21:0,22:0,        40:0, 41:0,        60:0,61:0,\
                                    2:1,3:1,            23:1,             42:1,              62:1,\
                                    4:2,5:2,6:2,        24:2,25:2,        43:2,44:2,45:2,    63:2,64:2,   \
                                   7:3,                 26:3,             46:3,              65:3,66:3,\
                                   8:4  ,9:4,           27:4,28:4,        47:4,48:4,49:4,    67:4,68:4,69:4,      \
                                   10:5, 11:5,          29:5   ,          50:5,51:5,         70:5,71:5,\
                                    12:6 , 13:6,        30:6,31:6,32:6,   52:6,53:6,         72:6,73:6,74:6,\
                                    14:7, 15:7,         33:7,34:7  ,      54:7,              75:7,  \
                                    16:8,               37:8,             55:8,56:8,         76:8,  \
                                    17:9,18:9,          35:9,36:9,        57:9,              77:9,\
                                    19:10,              38:10,39:10,       58:10,59:10,      78:10,79:10,\
                                    20:11,
                                   }
                
                self.family_grouping={0:torch.Tensor([0,1,      21,22 ,      40,41,          60,61 ]).long(),\
                                    1:torch.Tensor([2,3,       23,          42,             62]).long(),  \
                                    2:torch.Tensor([4,5,6,     24,25,       43,44,45,       63,64]).long(),\
                                      3:torch.Tensor([7,       26,          46,             65,66,]).long(),\
                                      4:torch.Tensor([8,9,     27,28,       47,48,49,       67,68,69]).long(),\
                                      5:torch.Tensor([10,11,   29,          50,51,          70,71]).long(),\
                                      6:torch.Tensor([12,13,   30,31,32,    52,53,          72,73,74]).long(),\
                                      7:torch.Tensor([14,15,   33,34,       54,             75]).long(),\
                                      8:torch.Tensor([16,      37,          55,56,          76]).long(),\
                                       9:torch.Tensor([17,18,  35,36,       57,             77]).long(),\
                                      10:torch.Tensor([19,     38,39,        58,59,         78,79]).long(),\
                                     
                                     11:torch.Tensor([20]).long(),
                                    
                                   }
 
                
        elif 'TOWOD' in args.dataset:
            if 't1' in args.train_set:
                self.label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,\
                                2:1,7:1,9:1, 11:1,12:1, 16:1,\
                                4:2,\
                                8:3,10:3,15:3,17:3,\
                                14:4,\
                                19:5}
                               
                    
                    
                               
                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                               }
                
            elif 't2' in args.train_set:
                self.label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,    20:0,\
                                2:1,7:1,9:1, 11:1,12:1, 16:1,                   26:1,27:1,28:1,29:1,\
                                4:2,\
                                8:3,10:3,15:3,17:3,\
                                14:4,\
                                19:5,\
                                   21:6,22:6,23:6,24:6,25:6,
                                    30:7,31:7,32:7,33:7,34:7,
                                   35:8,36:8,37:8,38:8,39:8,
                                   }


                self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,                  26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                               }
            elif 't3' in args.train_set:
                 self.label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,    20:0,\
                                2:1,7:1,9:1, 11:1,12:1, 16:1,                   26:1,27:1,28:1,29:1,\
                                4:2,\
                                8:3,10:3,15:3,17:3,\
                                14:4,\
                                19:5,\
                                   21:6,22:6,23:6,24:6,25:6,
                                    30:7,31:7,32:7,33:7,34:7,
                                   35:8,36:8,37:8,38:8,39:8,
                                       40:9,41:9,42:9,43:9,44:9,45:9,46:9,47:9,48:9,49:9, \
                                    50:10,51:10,52:10,53:10,54:10,55:10,56:10,57:10,58:10,59:10,\
                                   }
                
                
                
                 self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,                26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4]).long(),\
                                  3:torch.Tensor([8,10,15,17]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                                     9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                     10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                               }
            elif 't4' in args.train_set:
                 self.label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,    20:0,\
                                2:1,7:1,9:1, 11:1,12:1, 16:1,                   26:1,27:1,28:1,29:1,\
                                4:2,                                                                          74:2,75:2,76:2,77:2,78:2,79:2,\
                                8:3,10:3,15:3,17:3,                                                           60:3,61:3, \
                                14:4,\
                                19:5,                                                                          62:5,63:5,64:5,65:5,66:5,\
                                   21:6,22:6,23:6,24:6,25:6,
                                    30:7,31:7,32:7,33:7,34:7,
                                   35:8,36:8,37:8,38:8,39:8,
                                       40:9,41:9,42:9,43:9,44:9,45:9,46:9,47:9,48:9,49:9, \
                                    50:10,51:10,52:10,53:10,54:10,55:10,56:10,57:10,58:10,59:10,\
                                     67:11,68:11,69:11,70:11,71:11,72:11,73:11,
                                   }
                
                
                
                 self.family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                                1:torch.Tensor([2,7,9,11,12,16,               26,27,28,29 ]).long(),  \
                                2:torch.Tensor([4,                                                                                        74,75,76,77,78,79]).long(),\
                                  3:torch.Tensor([8,10,15,17,                                                                             60,61     ]).long(),\
                                  4:torch.Tensor([14]).long(),\
                                  5:torch.Tensor([19,                                                                                     62,63,64,65,66]).long(),
                                     6:torch.Tensor([                               21,22,23,24,25]).long(),
                                     7:torch.Tensor([                               30,31,32,33,34]).long(),
                                      8:torch.Tensor([                               35,36,37,38,39]).long(),
                                     9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                     10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                                     11:torch.Tensor([                                                                                    67,68,69,70,71,72,73]).long(),
                               }
        elif 'OWDETR' in args.dataset:
                if 't1' in args.train_set:
                    self.label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                        2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                        18:2,\
                                   }
                    
                    self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                    1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                    2:torch.Tensor([18]).long(),\
                                   }
                elif 't2' in args.train_set:
                    
                    self.label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                        2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                        18:2,\
                                         19:3,20:3,21:3,22:3,23:3,\
                                        24:4,25:4,26:4,37:4,38:4,39:4,\
                                        27:5,28:5,29:5,30:5,31:5,\
                                        32:6,33:6,34:6,35:6,36:6,\
                                   }
                    
                    
                    
                    self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                    1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                    2:torch.Tensor([18]).long(),\

                                    3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                    4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                    5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                    6:torch.Tensor([                                 32,33,34,35,36]).long(),

                                   }
                elif 't3' in args.train_set:
                    self.label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                        2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                        18:2,\
                                         19:3,20:3,21:3,22:3,23:3,\
                                        24:4,25:4,26:4,37:4,38:4,39:4,\
                                        27:5,28:5,29:5,30:5,31:5,\
                                        32:6,33:6,34:6,35:6,36:6,\
                                        40:7,41:7,42:7,43:7,44:7,45:7,46:7,47:7,48:7,49:7,\
                                        50:8,51:8,52:8,53:8,54:8,55:8,56:8,57:8,58:8,59:8,\
                                   }
                    
                    self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                    1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                    2:torch.Tensor([18]).long(),\

                                    3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                    4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                    5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                    6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                    7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                    8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),

                                   }
                elif 't4' in args.train_set:
                    
                    self.label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                        2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                        18:2,\
                                         19:3,20:3,21:3,22:3,23:3,\
                                        24:4,25:4,26:4,37:4,38:4,39:4,\
                                        27:5,28:5,29:5,30:5,31:5,\
                                        32:6,33:6,34:6,35:6,36:6,\
                                        40:7,41:7,42:7,43:7,44:7,45:7,46:7,47:7,48:7,49:7,\
                                        50:8,51:8,52:8,53:8,54:8,55:8,56:8,57:8,58:8,59:8,\
                                        
                                        60:9,61:9,62:9,63:9,64:9,78:9,\
                                        65:10,66:10,67:10,68:10,69:10,70:10,71:10,\
                                        72:11,73:11,74:11,75:11,76:11,77:11,79:11
                                   }
                    
                    self.family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                    1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                    2:torch.Tensor([18]).long(),\

                                    3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                    4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                    5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                    6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                    7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                    8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),

                                    9:torch.Tensor([                                                                                             60,61,62,63,64,78]).long(),
                                    10:torch.Tensor([                                                                                            65,66,67,68,69,70,71]).long(),
                                    11:torch.Tensor([                                                                                            72,73,74,75,76,77,79]).long(),

                                   }
        else:
            print('Error not known dataset')
            exit()
            
        # self.family_grouping={
    
    def _get_src_single_permutation_idx(self, indices, index):
        ## Only need the src query index selection from this function for attention feature selection
        batch_idx = [torch.full_like(src, i) for i, src in enumerate(indices)][0]
        src_idx = indices[0]
        return batch_idx, src_idx
    
  
    
 
    def loss_hyperbolic(self, outputs, targets, indices, num_boxes, log=True,**kwargs):
            
         
            idx = self._get_src_permutation_idx(indices)
            old_idx=deepcopy(idx)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
           
                
          
                        

            x0=outputs['hyperbolic_emb'][idx]
        
            try:
                pos_pairs=self.hyperbolic_emb.return_positive_pairs(target_classes_o)
            except:
                print('this iteration fails {} and layers {}'.format(self.iteration,kwargs['layer_index']))
                pos_pairs=None
         
        
            stats_ep = []
            stats_ep2 = []
            if pos_pairs is not None:
               
                z=torch.cat((x0.unsqueeze(0),pos_pairs))
                
                boolean=0
                if self.args.family_regularizer and self.hyperbolic_emb.poincare_family_mean is not None:
                    boolean=1
                    #try:
                    #ForkedPdb().set_trace()
                    family_index=torch.Tensor([self.label_mapping[i.item()] for i in target_classes_o]).long()
                 
                    pos_family_mean=self.hyperbolic_emb.poincare_family_mean[family_index.long()]
                 
                    
                    z2=torch.cat((x0.unsqueeze(0),pos_family_mean.unsqueeze(0)))
                    

                loss = 0
                loss_family=0
                counter=0
                counter2=0
                for i in range(self.args.samples_per_category+1):
                    for j in range(self.args.samples_per_category+1):
                        if i != j:
                            counter+=1
                            l1,s1=self.loss_f(z[i], z[j])
                            loss+=l1
                       
                                
                            stats_ep.append({**s1, "loss": l1.item()})
               
                if self.args.family_regularizer and self.hyperbolic_emb.poincare_family_mean is not None:           
                
                    for ii in range(x0.unsqueeze(0).shape[0]):
                        #ForkedPdb().set_trace()
                        l2,s2=self.loss_family(x0.unsqueeze(0)[ii].detach(), pos_family_mean.unsqueeze(0)[ii])
                        loss_family+=l2
                        stats_ep2.append({**s2, "loss": l2.item()})
                        counter2+=1
                    


                if 'layer_index' in kwargs.keys() and kwargs['layer_index']==6:
                    hyperbolic_stats=np.array([stats_ep[j]['logits/acc'] for j in range(len(stats_ep))])
                    
                    
                    if self.iteration%self.args.logging_freq==0:
                        print('\n')
                        print('logits hyperbolic accuracy: {}'.format(round( np.mean(hyperbolic_stats) ,3)))
                        
                        if self.args.family_regularizer and self.hyperbolic_emb.poincare_family_mean is not None:
                            hyperbolic_family_stats=np.array([stats_ep2[j]['logits/acc'] for j in range(len(stats_ep2))])
                            print('Family hyperbolic accuracy: {}'.format(round( np.mean(hyperbolic_family_stats) ,3)))
                    
                    if self.iteration%self.args.update_freq==0 and self.args.relabel:
                        self.hyperbolic_emb.form_poincare_mean()
                      
                loss_hyperbolic = {'loss_hyperbolic': loss/counter} #torch.clamp(,max=10.0)}
                if loss_family>0:
                    loss_hyperbolic['loss_family_hyperbolic']= loss_family/counter2 #torch.clamp(loss_family/(10*counter2),max=10.0)
                else:
                    loss_hyperbolic['loss_family_hyperbolic']= torch.Tensor([0])[0].to(targets[0]['image_id'].device)
                    
            else:
                loss_hyperbolic = {'loss_hyperbolic': torch.Tensor([0])[0].to(targets[0]['image_id'].device),'loss_family_hyperbolic':torch.Tensor([0])[0].to(targets[0]['image_id'].device)}
                                
            return loss_hyperbolic
            
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True,**kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:,:, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits
        
        idx = self._get_src_permutation_idx(indices)
        old_idx=deepcopy(idx)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
     

        target_classes = torch.full(src_logits.shape[:2], self.num_classes-1, dtype=torch.int64, device=src_logits.device)
        # ForkedPdb().set_trace()
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        empty_weight=self.empty_weight
      
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        
        
        
        if self.args.relabel and self.args.all_background and self.epoch>=self.args.start_relabelling:
            target_classes_onehot[...,-1]=0 ## remove everything to relabel now
            if self.hyperbolic_emb.poincare_mean is not None:# and self.iteration>=20:
                        
                        relevant_matrix=self.hyperbolic_emb.poincare_mean[self.valid_classes]
                    
                        values=dist_matrix(outputs['hyperbolic_emb'].view(-1,outputs['hyperbolic_emb'].shape[-1]),relevant_matrix, c=self.args.hyperbolic_c).min(dim=-1)[0].view(-1,self.args.num_queries)
                        ref=values[old_idx].min().item()
                        maximum=values[old_idx].max().item()
                        values[old_idx]=1e6
                        
                        if self.args.use_max_uperbound:
                            unknown_idx=nonzero_tuple(values<=maximum)
                       
                        
                        if len(unknown_idx[0])>0:
                            target_classes_onehot[:,:,-1][unknown_idx]=1.0
                         
                            known_dist_stats2=dist_matrix(outputs['hyperbolic_emb'][old_idx], self.hyperbolic_emb.poincare_mean[self.valid_classes], c=self.args.hyperbolic_c).min(dim=-1)[0]
                           
                          
                            
            

        assert len(idx[0])== (target_classes.unsqueeze(-1)<80).sum()
       
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, num_classes=self.num_classes, empty_weight=empty_weight,args=self.args,use_focal=self.args.use_focal_cls) * src_logits.shape[1]
       
        if 'layer_index' in kwargs.keys() and kwargs['layer_index']==6 and self.args.use_hyperbolic:
                ## buffer memory
                self.hyperbolic_emb.add(outputs['hyperbolic_emb'][idx].detach() ,target_classes_o )
                
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes,**kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes,**kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes,**kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": seg_sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
   

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'hyperbolic': self.loss_hyperbolic
        }
        
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # ForkedPdb().set_trace()
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets,iteration,epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # try:
        indices = self.matcher(outputs_without_aux, targets)
     
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
       
      
        self.iteration=iteration
        self.epoch=epoch
        

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            
         
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes,layer_index=6))
       
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    kwargs['layer_index']=i
                    # kwargs['obj_head']=outputs['obj_head']
                    
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, layer_index=i)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, invalid_cls_logits, temperature=1, pred_per_im=100,args=None):
        super().__init__()
        self.temperature=temperature
        self.invalid_cls_logits=invalid_cls_logits
        self.pred_per_im=pred_per_im
        self.args=args
        
        
  
    

    @torch.no_grad()
    def forward(self, outputs, target_sizes,remove_background=False,pred_per_im=100):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """        
      

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_logits[:,:, self.invalid_cls_logits] = -10e10

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        

        prob=out_logits.sigmoid()
          
        
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),  pred_per_im, dim=1) #
        scores = topk_values
        
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        labels[:,-1]=torch.Tensor([81]*labels.shape[0]).to(labels.device).long()
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        # ForkedPdb().set_trace()
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
class ExemplarSelection(nn.Module):
    def __init__(self, args, num_classes, matcher, invalid_cls_logits, temperature=1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.invalid_cls_logits=invalid_cls_logits
        self.temperature=temperature
        print(f'running with exemplar_replay_selection')   
              
            
    def calc_energy_per_image(self, outputs, targets, indices):
        out_logits, pred_obj = outputs['pred_logits'], outputs['pred_obj']
        out_logits[:,:, self.invalid_cls_logits] = -10e10

        idx = self._get_src_permutation_idx(indices)
        torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        logit_dist = torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        prob = logit_dist*out_logits.sigmoid()

        image_sorted_scores={}
        for i in range(len(targets)):
            image_sorted_scores[int(targets[i]['image_id'].cpu().numpy())] = {'labels':targets[i]['labels'].cpu().numpy(),"scores": prob[i,indices[i][0],targets[i]['labels']].detach().cpu().numpy()}
        return [image_sorted_scores]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, samples, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj'}
        indices = self.matcher(outputs_without_aux, targets)       
        return self.calc_energy_per_image(outputs, targets, indices)


def build(args):
    num_classes = args.num_classes
    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS+args.CUR_INTRODUCED_CLS, num_classes-1))
    
    
    # ForkedPdb().set_trace()
    print("Invalid class range: " + str(invalid_cls_logits))
    
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,args=args
    )
    
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef, 'loss_obj_ll': args.obj_loss_coef}
  
    
  
    if args.use_hyperbolic:
        weight_dict['loss_hyperbolic']=args.hyperbolic_coeff
    if args.family_regularizer:
        weight_dict['loss_family_hyperbolic']=args.family_hyperbolic_coeff
        
        
        
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
  
    if args.use_hyperbolic:
        losses.append('hyperbolic')
  
    
        
        
    if args.masks:
        losses += ["masks"]

    
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha,args=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim,args=args)}
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, exemplar_selection
