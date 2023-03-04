# -*-coding:utf-8-*-
import torch.nn as nn
from ltr import model_constructor
import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network


class HCAT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes, settings):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        
        self.fusionblock = FusionBlock()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(16, hidden_dim)
        self.query_embed_dim = nn.Embedding(settings.vector_num, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.settings = settings
        

    def forward(self, search, template):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        # if not isinstance(search, NestedTensor):
        #     search = nested_tensor_from_tensor(search)
        # if not isinstance(template, NestedTensor):
        #     template = nested_tensor_from_tensor(template)
        epoch = self.settings.epoch
        #scale = self.warmup(epoch)
        search_rgb, search_d = search[:,:3,:,:],search[:,3:,:,:]
        template_rgb, template_d = template[:,:3,:,:],template[:,3:,:,:]
        
        # fusion
        r_t = self.fusionblock(template_rgb, template_d)
        r_s = self.fusionblock(search_rgb, search_d)

        
        search_rgbd = search_rgb + r_s * search_d
        template_rgbd = template_rgb + r_t * template_d
        
        # #add
        # search_rgbd = torch.add(search_rgb,search_d)
        # template_rgbd = torch.add(template_rgb ,template_d)
        
        # #max
        # search_rgbd = torch.max(search_rgb,search_d)
        # template_rgbd = torch.max(template_rgb ,template_d)
        
        # #mean
        # search_rgbd = torch.mean(search_rgb,search_d)
        # template_rgbd = torch.mean(template_rgb ,template_d)
        
        



        feature_search_rgbd, pos_search_rgbd = self.backbone(search_rgbd)
        feature_template_rgbd, pos_template_rgbd = self.backbone(template_rgbd)


        
        # src_search, mask_search= feature_search[-1].decompose()
        # assert mask_search is not None
        # src_template, mask_template = feature_template[-1].decompose()
        # assert mask_template is not None
        src_search_rgbd = feature_search_rgbd[-1]
        src_template_rgbd = feature_template_rgbd[-1]
        



        hs = self.featurefusion_network(self.input_proj(src_template_rgbd),
                                        self.input_proj(src_search_rgbd),
                                        pos_template_rgbd[-1], pos_search_rgbd[-1],
                                        self.query_embed_dim.weight)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        info = {"search": r_s, "template":r_t,"scale":0}
        return out, info
    def warmup(self, epoch):
        if epoch<=20:
            return 0.1
        elif epoch<=40:
            return 0.2
        elif epoch<=60:
            return 0.3
        elif epoch<=80:
            return 0.4
        elif epoch<=100:
            return 0.5
        elif epoch<=120:
            return 0.6
        elif epoch<=140:
            return 0.7
        else:
            return 0.9
    def track(self, search):
        # if not isinstance(search, NestedTensor):
        #     search = nested_tensor_from_tensor_2(search)
        # import time
        # tic = time.time()
        search_rgb, search_d = search[:,:3,:,:],search[:,3:,:,:]
        r_s = self.fusionblock(search_rgb, search_d)
        search_rgbd = search_rgb + r_s*search_d
        features_search_rgbd = self.backbone.track(search_rgbd)
        

        # print('backbone_time'+str(time.time()-tic))
        pos_search = self.pos_search
        feature_template = self.zf
        pos_template = self.pos_template
        # src_search, mask_search= features_search[-1].decompose()
        # assert mask_search is not None
        # src_template, mask_template = feature_template[-1].decompose()
        # assert mask_template is not None
        src_template = feature_template
        src_search = features_search_rgbd[-1]
        # tic = time.time()
        hs = self.featurefusion_network(self.input_proj(src_template),
                                        self.input_proj(src_search),
                                        pos_template[-1], pos_search[-1],
                                        self.query_embed_dim.weight)
        # print('fusion_time'+str(time.time()-tic))
        # tic = time.time()
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # print('head_time'+str(time.time()-tic))
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z, feature_size):
        z_rgb, z_d = z[:,:3,:,:],z[:,3:,:,:] 
        r_s = self.fusionblock(z_rgb, z_d)
        z_rgbd = z_rgb + r_s * z_d
        zf, pos_template = self.backbone(z_rgbd)
        b, c, _, _ = zf[-1].shape
        src_search = torch.ones([b, c, feature_size, feature_size]).to(zf[-1].device)
        pos_search = []
        pos_search.append(self.backbone[1](src_search).to(zf[-1].dtype))
        self.zf =zf[-1]
        self.pos_template = pos_template
        self.pos_search = pos_search

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
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
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
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

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

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

def conv33(in_planes, out_planes, stride=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=dilation, bias=False, dilation=dilation)
class Block(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, use_bn=True):
        super(Block, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = conv33(planes, planes, stride, dilation=dilation)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.stride = stride
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        try:
            out = self.conv1(x)
        except:
            print('error')

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.gap(out)

        return out
class FusionBlock(nn.Module):
    def __init__(self):
        super(FusionBlock, self).__init__()
        self.rgbblock = Block(3,64)
        self.depthblock = Block(3,64)
        self.se = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, rgb, depth):
        r_out = self.rgbblock(rgb)
        d_out = self.depthblock(depth)
        out = torch.cat((r_out, d_out), dim=1) #?
        r = self.se(out)
        return r

        
        

@model_constructor
def hcat(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = HCAT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes,
        settings = settings
    )
    device = torch.device(settings.device)
    model.to(device)
    return model

def hcat_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
