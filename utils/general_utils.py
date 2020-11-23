import torch



class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

        return scale_reg






class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """
    BELOW_LOW_THRESHOLD=-1
    BETWEEN_THRESHOLDS=-2

    __annotations__={
        'BELOW_LOW_THRESHOLD':int,
        'BETWEEN_THRESHOLDS':int,
    }

    def __init__(self,high_threshold,low_threshold,allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        # self.BELOW_LOW_THRESHOLD = -1
        # self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel()==0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M(gt) x N(predict)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals,matches=match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches=matches.clone()
        else:
            all_matches=None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold=matched_vals<self.low_threshold
        between_thresholds=(matched_vals>=self.low_threshold)&(matched_vals<self.high_threshold)
        matches[below_low_threshold]=self.BELOW_LOW_THRESHOLD   # -1
        matches[between_thresholds]=self.BETWEEN_THRESHOLDS  # -2
        if self.allow_low_quality_matches:
            assert all_matches is not None
            matches=self.set_low_quality_matches_(matches,all_matches,match_quality_matrix)
        return matches

    def set_low_quality_matches_(self,matches,all_matches,match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt,_=match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality=torch.nonzero(
            match_quality_matrix==highest_quality_foreach_gt[:,None],as_tuple=False
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update=gt_pred_pairs_of_highest_quality[:,1]
        matches[pred_inds_to_update]=all_matches[pred_inds_to_update]
        return matches







class BalancedPositiveNegativeSampler(object):
    def __init__(self,batch_size_per_image,positive_fraction):
        self.batch_size_per_image=batch_size_per_image
        self.positive_fraction=positive_fraction

    def __call__(self,matched_idxs_per_image):
        '''
        从正负样本中随机选择个数平衡的num_ps num_neg个正负样本参与loss的计算
        :param matched_idxs_per_image:
        :return:
        idx_per_image_mask: shape=[n]    idx_per_image_mask[i]=1 positive sample,
                                         idx_per_image_mask[i]=0 negative sample,
                                         idx_per_image_mask[i]=255 others,
        '''
        positive=torch.nonzero(matched_idxs_per_image>=0,as_tuple=False).squeeze(1)
        negative=torch.nonzero(matched_idxs_per_image<0,as_tuple=False).squeeze(1)

        num_pos=int(self.batch_size_per_image*self.positive_fraction)
        # protect against not enough positive examples
        num_pos=min(positive.numel(),num_pos)
        num_neg=self.batch_size_per_image-num_pos
        # protect against not enough negative examples
        num_neg=min(num_neg,negative.numel())

        # randomly select positive and negative examples
        perm1=torch.randperm(positive.numel(),device=positive.device)[:num_pos]
        perm2=torch.randperm(negative.numel(),device=negative.device)[:num_neg]

        pos_idx_per_image=positive[perm1]
        neg_idx_per_image=negative[perm2]

        idx_per_image_mask=torch.full_like(matched_idxs_per_image,fill_value=255,dtype=torch.uint8)

        idx_per_image_mask[pos_idx_per_image]=1
        idx_per_image_mask[neg_idx_per_image]=0

        return idx_per_image_mask









def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])








class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Arguments:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """
    def __init__(self,k_min,k_max,canonical_scale=224,canonical_level=4,eps=1e-6):
        # type: (int, int, int, int, float) -> None
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self,boxlists):
        '''

        :param boxlists(list[BoxList]):
        :return:
        '''
        # compute level ids
        s=torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn(1) in FPN paper
        target_lvls=torch.floor(self.lvl0+torch.log2(s/self.s0)+torch.tensor(self.eps,dtype=s.dtype))
        target_lvls=torch.clamp(target_lvls,min=self.k_min,max=self.k_max)
        return (target_lvls.to(torch.int64)-self.k_min).to(torch.int64)




def init_level_wrapper(k_min,k_max,canonical_scale=224,canonical_level=4,eps=1e-6):
    return LevelMapper(k_min,k_max,canonical_scale,canonical_level,eps)







