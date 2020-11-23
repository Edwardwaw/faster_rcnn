import torch
from torch import nn
from commons.boxs_utils import box_iou
from losses.commons import IOULoss
from utils.general_utils import Matcher,BalancedPositiveNegativeSampler




class AnchorAssign(object):
    '''
    为每个anchor生成对应的gt匹配索引
    '''
    def __init__(self,fg_iou_thresh=0.7,bg_iou_thresh=0.3):
        self.matcher=Matcher(high_threshold=fg_iou_thresh,
                             low_threshold=bg_iou_thresh,
                             allow_low_quality_matches=True)
        # allow_low_quality_matches: ensure at least one proposal is matched to gt-box

    def __call__(self,anchors,targets):
        '''

        :param anchors: shape=[N,4]
        :param targets: shape=[M,4]
        :return:
        matches_targets_ids: shape=[N,]     matches_targets_ids[i]=k(k>=0) positive_sample, k is index of matched gt_box
                                            matches_targets_ids[i]=-1  BELOW_LOW_THRESHOLD
                                            matches_targets_ids[i]=-2  BETWEEN_THRESHOLDS
        '''
        target_anchor_iou=box_iou(targets,anchors)  # shape=[M,N]
        matches_target_idx=self.matcher(target_anchor_iou)
        return matches_target_idx





class RPNLoss(object):
    def __init__(self,batch_sample_size=256,fg_iou_thresh=0.7,bg_iou_thresh=0.3,positive_fraction=0.5):
        super(RPNLoss, self).__init__()
        self.batch_sample_size=batch_sample_size
        self.fg_iou_thresh=fg_iou_thresh
        self.bg_iou_thresh=bg_iou_thresh
        self.positive_fraction=positive_fraction
        self.anchor_assign=AnchorAssign(fg_iou_thresh,bg_iou_thresh)
        self.sampler=BalancedPositiveNegativeSampler(batch_size_per_image=batch_sample_size,
                                                     positive_fraction=positive_fraction)
        self.bce=nn.BCEWithLogitsLoss()
        self.box_loss=IOULoss()

    def __call__(self,objectness,proposal,targets,anchors):
        '''

        :param objectness(torch.tensor):  shape=[bs,num_anchors,1]
        :param proposal(torch.tensor):    shape=[bs,num_anchors,4] 4==>x1,y1,x2,y2 in input sizes
        :param targets(torch.tensor):   shape=[gt_num,7]  7=>(bs_idx,weights,label_idx,x1,y1,x2,y2)
        :param anchors(torch.tensor):  shape=[num_anchors,4] 4==>x1,y1,x2,y2 in input sizes
        :return:
        '''

        bs=objectness.shape[0]
        cls_predicts=list()
        cls_targets=list()
        box_predicts=list()
        box_targets=list()

        for i in range(bs):
            batch_targets=targets[targets[:,0]==i,1:]
            # 为每个anchor生成对应的一维gt-box索引
            if len(batch_targets):
                matches=self.anchor_assign(anchors,batch_targets[:,-4:])
            else:
                matches=torch.full((proposal[i].shape[0],),fill_value=-1,dtype=torch.long,device=proposal.device)
            positive_negative_mask=self.sampler(matches)  # balance the ratio between positive samples and negative samples
            valid_mask=positive_negative_mask!=255  # =1 pos, =0 neg, =255 others

            cls_predicts.append(objectness[i][valid_mask].squeeze(-1))  # shape=[batch_size_per_image]
            cls_targets.append(positive_negative_mask[valid_mask].float())  # shape=[batch_size_per_image]
            if len(batch_targets):
                positive_mask=positive_negative_mask==1
                box_predicts.append(proposal[i][positive_mask])
                box_targets.append(batch_targets[matches[positive_mask].long(),-4:])
        cls_predicts=torch.cat(cls_predicts,dim=0)
        cls_targets=torch.cat(cls_targets,dim=0)
        box_predicts=torch.cat(box_predicts,dim=0)
        box_targets=torch.cat(box_targets,dim=0)
        cls_loss=self.bce(cls_predicts,cls_targets)
        box_loss=self.box_loss(box_predicts,box_targets).mean()
        return cls_loss,box_loss





class ROIHeadLoss(object):
    def __init__(self):
        super(ROIHeadLoss, self).__init__()
        self.ce=nn.CrossEntropyLoss()
        self.box_loss=IOULoss()

    def __call__(self, cls_predicts, cls_targets, box_predicts, box_targets):
        cls_predicts=torch.cat(cls_predicts)
        cls_targets=torch.cat(cls_targets)
        box_predicts=torch.cat(box_predicts)
        box_targets=torch.cat(box_targets)
        # pos and neg samples are used to compute classification loss
        cls_loss=self.ce(cls_predicts,cls_targets.long())
        # only pos samples partipicate regression loss computation
        iou_loss=self.box_loss(box_predicts[cls_targets>0],box_targets[cls_targets>0]).mean()
        return cls_loss,iou_loss










