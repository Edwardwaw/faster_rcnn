import torch
import math
import numpy as np
from torch import nn
from nets.common import FPN
from utils.general_utils import BoxCoder
from torchvision.ops.roi_align import roi_align
from torchvision.ops.boxes import batched_nms

from losses.faster_rcnn_losses import RPNLoss,ROIHeadLoss
from utils.general_utils import Matcher,BalancedPositiveNegativeSampler,init_level_wrapper
from commons.boxs_utils import box_iou





default_model_config={
    "num_cls":80,  # in roi_head, classification_output_size is 81
    "backbone":"resnet18",
    "fpn_out_channels":256,
    # anchor settings
    "anchor_sizes":[32.,64.,128.,256.,512.],
    "strides":[4,8,16,32,64],
    "anchor_ratios":[0.5,1.,2.],
    # rpn settings
    "min_size":1.0,
    "rpn_pre_nms_top_n_train":2000,
    "rpn_pre_nms_top_n_test":1000,
    "rpn_post_nms_top_n_train":2000,
    "rpn_post_nms_top_n_test":1000,
    "rpn_nms_thresh":0.7,
    "rpn_fg_iou_thresh":0.7,
    "rpn_bg_iou_thresh":0.3,
    "rpn_batch_size_per_image":256,
    "rpn_positive_fraction":0.5,
    # roi settings
    "box_fg_iou_thresh":0.5,
    "box_bg_iou_thresh":0.5,
    "box_batch_size_per_image":512,
    "box_positive_fraction":0.25,
    "roi_resolution":7,
    "box_score_thresh":0.05,
    "box_nms_thresh":0.5,
    "box_detections_per_img":100
}



def switch_backbones(bone_name, pretrained=True):
    from nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
    if bone_name == "resnet18":
        return resnet18(pretrained=pretrained)
    elif bone_name == "resnet34":
        return resnet34(pretrained=pretrained)
    elif bone_name == "resnet50":
        return resnet50(pretrained=pretrained)
    elif bone_name == "resnet101":
        return resnet101(pretrained=pretrained)
    elif bone_name == "resnet152":
        return resnet152(pretrained=pretrained)
    elif bone_name == "resnext50_32x4d":
        return resnext50_32x4d(pretrained=pretrained)
    elif bone_name == "resnext101_32x8d":
        return resnext101_32x8d(pretrained=pretrained)
    elif bone_name == "wide_resnet50_2":
        return wide_resnet50_2(pretrained=pretrained)
    elif bone_name == "wide_resnet101_2":
        return wide_resnet101_2(pretrained=pretrained)
    else:
        raise NotImplementedError(bone_name)
    
    



class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    def __init__(self,in_channels,num_anchors):
        super(RPNHead, self).__init__()
        self.conv=nn.Conv2d(in_channels,in_channels,3,1,1)
        self.cls_logits=nn.Conv2d(in_channels,num_anchors,1) # binary classification in rpn
        self.bbox_pred=nn.Conv2d(in_channels,num_anchors*4,1)

        for l in self.children():
            nn.init.normal_(l.weight,std=0.01)
            nn.init.constant_(l.bias,0.)

    def forward(self,x):
        t = self.conv(x).relu()
        cls = self.cls_logits(t)
        reg = self.bbox_pred(t)
        return cls, reg



class RPN(nn.Module):
    def __init__(self,cfg):
        super(RPN, self).__init__()
        self.cfg=cfg
        self.anchors=None
        self.rpn_head=RPNHead(self.cfg['fpn_out_channels'],len(self.cfg['anchor_ratios']))
        self.box_coder=BoxCoder(weights=[1.0,1.0,1.0,1.0])
        self.rpn_loss=RPNLoss(batch_sample_size=self.cfg['rpn_batch_size_per_image'],
                              fg_iou_thresh=self.cfg['rpn_fg_iou_thresh'],
                              bg_iou_thresh=self.cfg['rpn_bg_iou_thresh'],
                              positive_fraction=self.cfg['rpn_positive_fraction'])

    def build_anchors_delta(self, anchor_size=32.):
        """
        :param anchor_size:
        :return: [anchor_num, 4]
        """
        ratio = torch.tensor(self.cfg['anchor_ratios']).float()
        w = (anchor_size * ratio.sqrt()).view(-1) / 2
        h = (anchor_size / ratio.sqrt()).view(-1) / 2
        delta = torch.stack([-w, -h, w, h], dim=1)
        return delta

    def build_anchors(self, feature_maps):
        """
        :param feature_maps:
        :return: list(anchor) anchor:[all,4] (x1,y1,x2,y2)
        """
        assert len(self.cfg['anchor_sizes']) == len(feature_maps)
        assert len(self.cfg['anchor_sizes']) == len(self.cfg['strides'])
        anchors = list()
        for stride, size, feature_map in zip(self.cfg['strides'], self.cfg['anchor_sizes'], feature_maps):
            # 9*4
            anchor_delta = self.build_anchors_delta(size)
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            # h,w,4
            grid = torch.stack([xv, yv, xv, yv], 2).float()
            anchor = (grid[:, :, None, :] + 0.5) * stride + anchor_delta[None, None, :, :]
            anchor = anchor.view(-1, 4)
            anchors.append(anchor)
        anchors = torch.cat(anchors, dim=0)
        return anchors


    def filter_proposals(self,proposals,objectness,num_anchors_per_layer,shapes):
        '''

        :param proposals:  shape=[bs,num_anchors,4] 4==>x1,y1,x2,y2 in input sizes
        :param objectness:  shape=[bs,num_anchors,1]   binary classification
        :param num_anchors_per_layer(list,len=fpn_layers): list(n)  n=anchor_num of each featuremap
        :param shapes(list): len=bs
        :return:
        filtered_boxes(list, len=bs): list(boxes_one_image)  boxes_one_image.shape=[n,4]
        filtered_scores(list, len=bs): list(scores_one_image) scores_one_image.shape=[n,1]
        '''
        min_size=torch.tensor(self.cfg['min_size'],device=proposals.device)  #define min proposal size
        pre_nms_top_n=self.cfg['rpn_pre_nms_top_n_train'] if self.training else self.cfg['rpn_pre_nms_top_n_test']   # 2000
        post_nms_top_n=self.cfg['rpn_post_nms_top_n_train'] if self.training else self.cfg['rpn_post_nms_top_n_test']   # 1000

        start_idx=0
        filtered_idx=list()
        levels=list()

        for ldx,layer_num in enumerate(num_anchors_per_layer):
            levels.append(torch.full(size=(layer_num,),fill_value=ldx,dtype=torch.int64,device=proposals.device))  # shape=[layer_num,], value=the i-th layer indices
            layer_objectness=objectness[:,start_idx:start_idx+layer_num,:]
            layer_top_n=min(layer_objectness.size(1),pre_nms_top_n)
            _,top_k_idx=layer_objectness.topk(dim=1,k=layer_top_n)
            filtered_idx.append(top_k_idx+start_idx)
            start_idx+=layer_num

        levels=torch.cat(levels,dim=0).unsqueeze(0).repeat(proposals.size(0),1)  # shape=[bs,num_anchors]
        filtered_idx=torch.cat(filtered_idx,dim=1)   # shape=[bs,sum_of_pre_topn,1]
        objectness=objectness.gather(dim=1,index=filtered_idx).squeeze(-1)
        proposals=proposals.gather(dim=1,index=filtered_idx.repeat(1,1,4))
        levels=levels.gather(dim=1,index=filtered_idx[...,0])

        filtered_boxes=list()
        filtered_scores=list()

        # perform nms on each image, but do it on different fpn layer as lvl
        for box,scores,lvl,shape in zip(proposals,objectness,levels,shapes):
            # clip to img_size
            box[...,[0,2]]=box[...,[0,2]].clamp(min=0,max=shape[0])
            box[...,[1,3]]=box[...,[1,3]].clamp(min=0,max=shape[1])
            # remove small box
            dw=box[...,2]-box[...,0]
            dh=box[...,3]-box[...,1]
            keep=(dw>min_size)&(dh>min_size)
            box,scores,lvl=box[keep],scores[keep],lvl[keep]
            # perform nms on different layers by lvl
            keep=batched_nms(box,scores,lvl,self.cfg['rpn_nms_thresh'])
            keep=keep[:post_nms_top_n]
            box,scores=box[keep],scores[keep]
            # add it to list by bs
            filtered_boxes.append(box)
            filtered_scores.append(scores)
        # filtered_boxes = torch.stack(filtered_boxes, dim=0)
        # filtered_scores = torch.stack(filtered_scores, dim=0)
        return filtered_boxes, filtered_scores


    def forward(self,xs,shapes,targets):
        feature_size=[item.size(2)*item.size(3) for item in xs]
        anchor_length=np.array(feature_size).sum()*len(self.cfg['anchor_ratios'])  # num_anchors
        if self.anchors is None or self.anchors.shape[0]!=anchor_length:
            anchors=self.build_anchors(xs).to(xs[0].device)
            self.anchors=anchors

        cls_list=list()
        reg_list=list()
        num_anchors_per_layer=list()

        for x in xs:
            cls,reg=self.rpn_head(x)
            num_anchors_per_layer.append(int(reg.size(1)*reg.size(2)*reg.size(3)/4))
            cls_list.append(cls.permute(0,2,3,1).contiguous().view(cls.size(0),-1,1))
            reg_list.append(reg.permute(0,2,3,1).contiguous().view(reg.size(0),-1,4))
        objectness=torch.cat(cls_list,dim=1)  # shape=[bs,num_anchors,1]
        reg_targets=torch.cat(reg_list,dim=1) # shape=[bs,num_anchors,4]
        proposals=self.box_coder.decoder(reg_targets,self.anchors)  # shape=[bs,num_anchors,4] 4==>x1,y1,x2,y2 in input sizes
        # filtered_boxes(list,len=bs): list(filter_box)  filter_box.shape=[N,4]  N=post_nms_top_n
        # 得到筛选过后的proposals(已解码到输入尺度),为后续阶段的box_refine提供输入
        filtered_boxes, _ = self.filter_proposals(proposals.detach(),
                                                  objectness.detach(),
                                                  num_anchors_per_layer,
                                                  shapes=shapes)
        # 在训练阶段产生rpn_loss，以便测试时rpn可以产生准确的proposal
        loss=dict()
        if self.training:
            cls_loss,box_loss=self.rpn_loss(objectness,proposals,targets,self.anchors)
            loss['rpn_cls_loss']=cls_loss
            loss['rpn_box_loss']=box_loss
        return filtered_boxes,loss








class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """
    def __init__(self,in_channels,representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6=nn.Sequential(nn.Linear(in_channels,representation_size),
                               nn.ReLU(inplace=True))
        self.fc7=nn.Sequential(nn.Linear(representation_size,representation_size),
                               nn.ReLU(inplace=True))

    def forward(self,x):
        x=x.flatten(start_dim=1)
        x=self.fc6(x)
        x=self.fc7(x)
        return x





class FastRCNNPredictor(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score=nn.Linear(in_channels,num_classes)
        # self.bbox_pred=nn.Linear(in_channels,num_classes*4)
        self.bbox_pred=nn.Linear(in_channels,4)

    def forward(self,x):
        if x.dim()==4:
            assert list(x.shape[-2:])==[1,1]
        x=x.flatten(start_dim=1)
        scores=self.cls_score(x)
        bbox_deltas=self.bbox_pred(x)
        return scores,bbox_deltas





class MultiROIPool(nn.Module):
    def __init__(self,strides,filter_level=None,output_size=7,sampling_ratio=2):
        super(MultiROIPool, self).__init__()
        self.output_size=output_size
        self.sampling_ratio=sampling_ratio
        if filter_level is None:
            filter_level=[0,1,2,3]  # note: 仅从 2 3 4 5th featuremap作roi pooling
        self.filter_level=filter_level
        self.strides=strides  #[4,8,16,32,64]
        self.scales = (1 / np.array(strides, dtype=np.float32))[filter_level] #1/[4,8,16,32]
        self.feature_level=np.log2(1/self.scales)  #[2,3,4,5] featuremap level when using roi align
        # map each proposal into different featuremap level by its size
        self.mapper=init_level_wrapper(k_min=self.feature_level[0],
                                       k_max=self.feature_level[-1],
                                       canonical_level=4)

    @staticmethod
    def convert_to_roi_format(proposal):
        concat_boxes=torch.cat(proposal,dim=0)
        device,dtype=concat_boxes.device,concat_boxes.dtype
        ids=torch.cat([
            torch.full_like(b[:,:1],i,dtype=dtype,layout=torch.strided,device=device)
            for i,b in enumerate(proposal)
        ]
        ,dim=0)
        rois=torch.cat([ids,concat_boxes],dim=1)  # shape=[num_proposal_of_bs_img,5] 5==>batch_id,x1,y1,x2,y2
        return rois

    def forward(self,xs,proposal):
        '''

        :param xs(list,len=5):  [p2,p3,p4,p5,p6], that is fpn's output
        :param proposal(list,len=bs):  list(proposal)  proposal.shape=[np+nn,4]
        :return:
        '''
        # list(featuremap), len=len(filter_levl)=4  [p2,p3,p4,p5]
        filter_features=list()
        for i in self.filter_level:
            filter_features.append(xs[i])

        levels=self.mapper(proposal)  # shape=[sum_of_batch_proposals] 元素取值范围[0,1,2,3]
        rois=self.convert_to_roi_format(proposal)  # shape=[sum_proposal_of_bs_img,5] 5==>batch_id,x1,y1,x2,y2

        result=torch.zeros(
            (len(rois),xs[0].shape[1],self.output_size,self.output_size),
            dtype=xs[0].dtype,
            device=rois.device
        )

        for i in range(len(self.filter_level)):  #[0,1,2,3]
            per_level_feature=filter_features[i]
            scale=self.scales[i]
            level_mask=levels==i
            rois_per_level=rois[level_mask]
            result_idx_in_level=roi_align(
                per_level_feature,rois_per_level,
                output_size=self.output_size,
                spatial_scale=scale,sampling_ratio=self.sampling_ratio
            )
            result[level_mask,...]=result_idx_in_level
        return result  # shape=[len(rois),c,output_size,output_size]











class ROIHead(nn.Module):
    def __init__(self,cfg,num_cls):
        super(ROIHead, self).__init__()
        self.cfg=cfg
        self.box_head=TwoMLPHead(self.cfg["fpn_out_channels"]*self.cfg["roi_resolution"]**2,1024)
        self.box_predictor=FastRCNNPredictor(1024,num_cls)

        self.sampler=BalancedPositiveNegativeSampler(self.cfg["box_batch_size_per_image"],
                                                     self.cfg["box_positive_fraction"])
        self.matcher=Matcher(self.cfg["box_fg_iou_thresh"],self.cfg["box_bg_iou_thresh"])
        self.pooling=MultiROIPool(self.cfg["strides"])
        self.box_coder=BoxCoder()
        self.head_loss=ROIHeadLoss()

    def select_train_sample(self,proposal,targets):
        '''

        :param proposal(list,len=bs): list(filter_box)  filter_box.shape=[N,4]  N=post_nms_num
        :param targets: (bs,7)  7==>(bs_idx,weights,label_idx,x1,y1,x2,y2)
        :return:
        ret_proposal (list, len=bs): list(proposal)  shape=[n_p+n_n,4]
        ret_labels (list, len=bs):  list(labels)  shape=[n_p+n_n,1]  =0 neg, >0 pos
        ret_targets (list, len=bs):  list(targets)  shape=[n_p+n_n,4]  =[0,0,0,0] neg, else pos
        '''
        bs=len(proposal)
        ret_proposal=list()
        ret_labels=list()
        ret_targets=list()
        # ret_mask=list()

        for i in range(bs):
            batch_targets=targets[targets[:,0]==i,1:]
            if len(batch_targets):
                # question: why add gt_box into proposals?
                batch_proposal=torch.cat([proposal[i],batch_targets[:,-4:]],dim=0)
                targets_proposal_iou=box_iou(batch_targets[:,-4:],batch_proposal)
                match_idx=self.matcher(targets_proposal_iou)
            else:
                batch_proposal=proposal[i]
                match_idx=torch.full((batch_proposal.shape[0],),
                                     fill_value=-1,
                                     dtype=torch.long,
                                     device=batch_proposal.device)
            positive_negative_mask=self.sampler(match_idx)  # =1 pos, =0 neg, =255 other
            valid_mask=positive_negative_mask!=255  # pos neg =True, other =False
            ret_proposal.append(batch_proposal[valid_mask])  # shape=[n_p+n_n,4]
            compress_mask=positive_negative_mask[valid_mask].bool()  # shape=[n_p+n_n,] =True pos,  =False neg
            # ret_mask.append(compress_mask)
            labels_idx=torch.zeros_like(compress_mask,dtype=torch.float,requires_grad=False)
            labels_idx[compress_mask]=batch_targets[match_idx[valid_mask][compress_mask].long(),1]+1   # why add 1? the cls_output_size = (num_cls+1)
            ret_labels.append(labels_idx)  # shape=[n_p+n_g,1] =0 neg, >0 pos
            targets_box=torch.zeros_like(batch_proposal[valid_mask])
            targets_box[compress_mask,:]=batch_targets[match_idx[valid_mask][compress_mask].long(),-4:]
            ret_targets.append(targets_box)
        # ret_proposal = torch.stack(ret_proposal, dim=0)
        # ret_labels = torch.stack(ret_labels, dim=0)
        # ret_targets = torch.stack(ret_targets, dim=0)
        return ret_proposal, ret_labels, ret_targets



    def post_process(self,box_predicts,cls_predicts,shapes):
        '''

        :param box_predicts(len=bs):  list(box_predict) , box_predict=[n_p+n_n,4]  4==>x1,y1,x2,y2
        :param cls_predicts(len=bs):  list(cls_predict) , cls_predict=[n_p+n_n,num_cls]
        :param shapes(len=bs):  list(shape)
        :return:
        '''

        ret_dets=list()
        for box,cls,shape in zip(box_predicts,cls_predicts,shapes):
            score=cls.softmax(dim=-1)
            max_val,max_idx=score.max(dim=-1)
            thresh_mask=max_val>self.cfg['box_score_thresh']
            positive_mask = max_idx > 0  # question: why >0? because 0th means background score
            valid_mask = thresh_mask & positive_mask
            if valid_mask.sum()==0:
                ret_dets.append(None)
                continue
            nms_box=box[valid_mask]
            nms_scores=max_val[valid_mask]
            nms_label_idx = max_idx[valid_mask] - 1
            idx=batched_nms(nms_box,nms_scores,nms_label_idx,self.cfg['box_nms_thresh'])
            valid_idx=idx[:self.cfg['box_detections_per_img']]
            detects=torch.cat([
                nms_box[valid_idx],
                nms_scores[valid_idx].unsqueeze(-1),
                nms_label_idx[valid_idx].unsqueeze(-1)
            ],dim=-1)
            detects[..., [0, 2]] = detects[..., [0, 2]].clamp(min=0, max=shape[0])
            detects[..., [1, 3]] = detects[..., [1, 3]].clamp(min=0, max=shape[1])
            ret_dets.append(detects)
        return ret_dets



    def forward(self,xs,proposal,shapes,targets=None):
        '''

        对应的流程: proposals==>筛选正负样本并找到其对应的label_targets和box_targets
                            ==>roi pooling==>box_head==>predictor
                                                      ==> loss

        :param xs:
        :param proposal (list,len=bs): list(filter_box)  filter_box.shape=[N,4]  N=post_nms_top_n
        :param shapes:
        :param targets: (bs,7)  7==>(bs_idx,weights,label_idx,x1,y1,x2,y2)
        :return:
        '''
        label_targets,box_targets=None,None
        if self.training:
            assert targets is not None
            '''
            为proposal筛选对应的targets
            :return
            proposal (list, len=bs): list(proposal)  shape=[n_p+n_n,4]
            label_targets (list, len=bs):  list(labels)  shape=[n_p+n_n,1]  =0 neg, >0 pos
            box_targets (list, len=bs):  list(targets)  shape=[n_p+n_n,4]  =[0,0,0,0] neg, else pos
            '''
            proposal,label_targets,box_targets=self.select_train_sample(proposal,targets)

        box_features=self.pooling(xs,proposal)  # shape=[num_batch_proposal,c,outsize,outsize]
        box_features=self.box_head(box_features) # shape=[num_batch_proposal,1024]
        cls_predicts,box_delta_predicts=self.box_predictor(box_features)   #shape=[num_batch_proposal,num_cls], =[num_batch_proposal,4]

        proposal_lengths=[len(item) for item in proposal]  # proposal nums in different image
        # list(box_predict) len=bs, box_predict=[n_p+n_n,4]  4==>x1,y1,x2,y2
        box_predicts=[self.box_coder.decoder(delta_predict,proposal_item) for delta_predict,proposal_item
                       in zip(box_delta_predicts.split(proposal_lengths,dim=0),proposal)]

        loss=dict()

        if self.training:
            result=None
            cls_loss,box_loss=self.head_loss(cls_predicts.split(proposal_lengths, dim=0),
                                             label_targets,box_predicts,box_targets)
            loss['roi_cls_loss']=cls_loss
            loss['roi_box_loss']=box_loss
        else:
            # results (list,len=bs):  list(dets)  dets.shape=[num_box,6]  6==>x1,y1,x2,y2,score,label
            result=self.post_process(box_predicts,
                                     cls_predicts.split(proposal_lengths,dim=0),
                                     shapes)

        return result,loss








class FasterRCNN(nn.Module):
    def __init__(self,cfg=None):
        super(FasterRCNN, self).__init__()
        if cfg is None:
            cfg=default_model_config
        else:
            cfg={**default_model_config,**cfg}
        self.backbones=switch_backbones(cfg['backbone'])
        self.neck=FPN(self.backbones.inner_channels,cfg['fpn_out_channels'])
        self.rpn=RPN(cfg)
        self.roi_head=ROIHead(cfg,cfg['num_cls']+1)

    def forward(self,x,shapes=None,targets=None):
        if self.training:
            assert targets is not None
        if shapes is None:
            shapes= [(x.shape[2],x.shape[3]) for _ in range(x.size(0))]
        c2,c3,c4,c5=self.backbones(x)
        p2,p3,p4,p5,p6=self.neck([c2,c3,c4,c5])
        proposals,rpn_loss=self.rpn([p2,p3,p4,p5,p6],shapes,targets)
        detects,roi_loss=self.roi_head([p2,p3,p4,p5,p6],proposals,shapes,targets)
        return detects,{**rpn_loss,**roi_loss}



if __name__ == '__main__':
    input_tensor = torch.rand(size=(4, 3, 640, 640))
    rcnn = FasterRCNN()
    rcnn.eval()
    _,_=rcnn(input_tensor)























