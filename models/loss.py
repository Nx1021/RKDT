from utils.yaml import load_yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_area
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from .roi_pipeline import LandmarkDetectionResult
from .utils import denormalize_bbox, tensor_to_numpy, denormalize_points
from .results import GtResult, PredResult, MatchedRoi

from . import SYS
import platform
import matplotlib.pyplot as plt

from typing import Union

import time


def calculate_scores(pk, points, alpha = 0.15, beta = 0.4, eps = 1e-4):
    '''
    brief
    -----
    计算预测点的置信度目标

    parameters
    -----
    * pk: [pred_num, 2]
    * points: [gt_num, 2]

    return
    ----
    scores: [pred_num, gt_num+1]
    '''
    alpha   = alpha
    beta    = beta
    eps     = eps

    distances = torch.cdist(pk, points)  # 计算每个点与每个 pk 之间的欧几里得距离
    bg_distances = torch.clamp( -distances.min(dim=-1)[0] + alpha, eps, alpha)
    distances = torch.cat((distances, bg_distances.unsqueeze(-1)), dim= -1)

    scores = beta / distances
    scores = torch.clamp(scores, 0, 1/eps)

    scores = torch.softmax(scores, dim=-1)

    return scores

def find_best_rotation(A, B):
    """
    寻找使得每组点集 A 与 B 对应位置点的间距最小的旋转角向量
    输入参数：
    A: 点集 A，形状为 (k, N, 2) 的 Tensor，k 表示有 k 组点集，N 是每组点集的点的数量
    B: 点集 B，形状为 (k, N, 2) 的 Tensor
    返回值：
    rotation_angles: 旋转角向量，形状为 (k,) 的 Tensor
    """

    k, N, _ = A.shape

    # 计算每组点集 A 与点集 B 之间的质心
    centroid_A = torch.mean(A, dim=1)
    centroid_B = torch.mean(B, dim=1)

    # 将点集 A 和 B 中心化
    A_centered = A - centroid_A[:, None, :]
    B_centered = B - centroid_B[:, None, :]

    # 计算每组点集 A 与点集 B 之间的协方差矩阵
    cov_matrix = torch.matmul(A_centered.transpose(1, 2), B_centered)
    U, _, V = torch.svd(cov_matrix)

    # 计算旋转矩阵 R
    R = torch.matmul(V, U.transpose(1, 2))

    # 计算旋转角向量
    rotation_angles = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    return rotation_angles

class LossKW():
    LOSS    = "Loss"
    LDLOSS  = "Last Decoder Loss"
    CLS     = "Class Loss"
    PN      = "PN Loss"
    DIST    = "Distance Loss"
    ROT     = "Rotation Loss"

class LossResult():
    def __init__(self, loss_Tensor:Tensor, item_weights:Tensor, item_names:list[str], group_weights:Tensor = None) -> None:
        '''
        parameter
        -----
        * loss_Tensor: Tensor [C, num_groups, num_items]
        * item_weights: the weight of each column
        * group_weights: the weight of each group

        len(group_weights) must be equal to loss_Tensor.shape[0]
        len(item_weights) must be equal to loss_Tensor.shape[1]
        '''
        if len(loss_Tensor.shape) == 2:
            loss_Tensor = loss_Tensor.unsqueeze(0)
        if loss_Tensor.numel() == 0:
            self.loss_Tensor = loss_Tensor
            return
        if group_weights is None:
            group_weights = torch.ones(loss_Tensor.shape[0]).to(loss_Tensor.device)
        assert len(loss_Tensor.shape) == 3
        assert len(group_weights.shape) == 1 and group_weights.numel() == loss_Tensor.shape[1] 
        assert len(item_weights.shape) == 1 and item_weights.numel() == len(item_names) == loss_Tensor.shape[2] 
        self.loss_Tensor = loss_Tensor
        self.item_weights = item_weights.to(loss_Tensor.device)
        self.item_names = item_names
        self.group_weights = group_weights.to(loss_Tensor.device)

    def apply_weight(self):
        if self.valid:
            weight_matrix = torch.mm(
                self.group_weights.unsqueeze(-1),
                self.item_weights.unsqueeze(0))
            loss_Tensor = self.loss_Tensor * weight_matrix
            return loss_Tensor
        else:
            return self.loss_Tensor

    def loss(self) -> Tensor:
        if self.valid:
            loss_Tensor = self.apply_weight()
            loss = torch.mean(torch.sum(loss_Tensor, dim=-1))
            return loss
        else:
            return torch.Tensor([0.0]).to(self.loss_Tensor.device)

    @property
    def valid(self):
        return self.loss_Tensor.numel() != 0

    @property
    def C(self):
        return self.loss_Tensor.shape[0]

    @property
    def num_groups(self):
        return self.loss_Tensor.shape[1]

    @property
    def num_items(self):
        return self.loss_Tensor.shape[2]

    def to_dict(self) -> dict:
        dict_ = {}
        if self.valid:
            loss_Tensor:np.ndarray = self.apply_weight().detach().cpu().numpy()
            item_losses = np.mean(loss_Tensor, (0, 1))
            for name, v in zip(self.item_names, item_losses):
                dict_[name] = float(v)
            dict_[LossKW.LDLOSS] = float(np.sum(np.mean(loss_Tensor, 0)[-1]))
            dict_[LossKW.LOSS] = float(np.mean(np.sum(loss_Tensor, -1)))
        return dict_

    @staticmethod
    def concat(result_list: list["LossResult"]):
        if len(result_list) > 0:
            item_weights: Tensor    = result_list[0].item_weights
            item_names: list[str]   = result_list[0].item_names
            group_weights: Tensor   = result_list[0].group_weights

            concat = torch.concat([r.loss_Tensor for r in result_list])

            return LossResult(concat, item_weights, item_names, group_weights)
        else:
            return LossResult(torch.zeros(0,0,0), None, None, None)

class LandmarkLossRecorder():
    def __init__(self, name, top = True) -> None:
        self.name = name
        self.loss_sum:Tensor = torch.Tensor([0.0])
        self.loss_record:dict[float] = {}
        self.loss_record[LossKW.LOSS] = 0.0
        self.detect_num:int = 0
        if top:
            self.buffer = LandmarkLossRecorder("__buffer", top=False)

    def record(self, result:LossResult):
        for key, value in result.to_dict().items():
            self.buffer.loss_record.setdefault(key, 0.0)
            self.buffer.loss_record[key] += value * result.C
        self.buffer.detect_num += result.C

    def clear(self):
        self.loss_sum = torch.Tensor([0.0])
        self.loss_record.clear()
        self.loss_record[LossKW.LOSS] = 0.0
        self.detect_num = 0

    def loss(self):
        return self.__get_mean(self.loss_record[LossKW.LOSS])

    def __get_mean(self, value:float):
        if self.detect_num == 0:
            return 0.0
        else:
            return value / self.detect_num

    def to_dict(self):
        dict_:dict[str, float] = {}
        for key, sum_value in self.loss_record.items():
            dict_[self.name + ' ' + key] = self.__get_mean(sum_value)

        return dict_

    def merge(self):
        buffer = self.buffer
        self.detect_num += buffer.detect_num
        for key, value in buffer.loss_record.items():
            self.loss_record.setdefault(key, 0.0)
            self.loss_record[key] += value

class LandmarkLoss(nn.Module):
    def __init__(self, cfg_file) -> None:
        super().__init__()
        self.cfg = load_yaml(cfg_file)
        self.landmark_num:int = self.cfg["landmark_num"]
    
    @property
    def calc_intermediate(self):
        return self.cfg["calc_intermediate"]
    
    @property
    def class_loss_w(self):       
        return self.cfg["class_loss_w"]
    
    @property
    def dist_loss_w(self):
        return self.cfg["dist_loss_w"]
    
    @property
    def PN_loss_w(self):               
        return self.cfg["PN_loss_w"]
    
    @property
    def rotation_loss_w(self):         
        return self.cfg["rotation_loss_w"]

    @property
    def alpha(self):                   
        return self.cfg["score_alpha"]
    
    @property
    def beta(self):                    
        return self.cfg["score_beta"]
    
    @property
    def eps(self):                     
        return self.cfg["score_eps"]

    def match_roi(self, gt_bbox_n, rlts: list[LandmarkDetectionResult]):
        pred_bbox_n = torch.stack([x.bbox_n for x in rlts]) # [num_roi_active, 4]
        ### 用giou将预测的框和真值的框匹配
        cost_matrix         = generalized_box_iou(pred_bbox_n, gt_bbox_n)
        row_ind, col_ind    = linear_sum_assignment(cost_matrix.cpu(), maximize=True)
        filtered_row_ind, filtered_col_ind = np.array([], np.int64), np.array([], np.int64)
        for ri ,ci in zip(row_ind, col_ind):
            score = cost_matrix[ri, ci]
            if score > 0.5:
                filtered_row_ind = np.append(filtered_row_ind, ri)
                filtered_col_ind = np.append(filtered_col_ind, ci)
        return filtered_row_ind, filtered_col_ind

    @staticmethod
    def get_target(gt_landmarks:Tensor, matched_pred_bbox:Tensor):
        # if len(gt_landmarks.shape) == 2 and len(matched_pred_bbox.shape) == 1:
        #     gt_landmarks = gt_landmarks.unsqueeze(0)
        #     matched_pred_bbox = matched_pred_bbox.unsqueeze(0)
        # assert gt_landmarks.shape[0] == matched_pred_bbox.shape[0]
        landmark_num        = gt_landmarks.shape[1]
        matched_lt          = matched_pred_bbox[:, :2] # [match_num, (x, y)]
        matched_wh          = matched_pred_bbox[:, 2:] - matched_pred_bbox[:, :2] # [match_num, (w, h)]
        matched_lt          = matched_lt.unsqueeze(1).repeat(1, landmark_num, 1) # [match_num, ldmk_num, (w, h)]
        matched_wh          = matched_wh.unsqueeze(1).repeat(1, landmark_num, 1) # [match_num, ldmk_num, (w, h)]
        target_landmarks_n  = (gt_landmarks - matched_lt) / matched_wh # [match_num, ldmk_num, (w, h)]
        # target_landmarks_n  = target_landmarks_n.unsqueeze(0) # [output_num, match_num, ldmk_num, (w, h)]

        return target_landmarks_n

    def get_pred(self, row_ind, rlts:list[LandmarkDetectionResult]):
        def parse(*names):
            rlt = []
            for name in names:
                pred_           = torch.stack([x.__getattribute__(name) for x in rlts], dim=1)  # [output_num, num_roi_active, ...]
                # pred_           = torch.transpose(pred_, 0, 1)                                  # [num_roi_active, output_num, ...]]
                matched_pred    = pred_[:, row_ind, ...]        # [output_num, match_num, ...]
                rlt.append(matched_pred)
            return rlt
        (matched_pred_landmarks_n,           # [output_num, match_num, num_queries, (w, h)]
        matched_pred_landmarks_probs         # [output_num, match_num, num_queries, ldmk_num + 1]
        ) = parse("landmarks_n", "landmarks_probs")

        output_num = matched_pred_landmarks_n.shape[0]
        return (matched_pred_landmarks_n, matched_pred_landmarks_probs), output_num


    def match_landmarks(self, out_probs:Tensor, out_coord:Tensor, tgt_coord:Tensor):
        '''
        pred_landmarks_n    : [match_num, num_queries, (w, h)]
        perd_landmarks_probs: [match_num, num_queries, ldmk_num + 1]
        target_landmarks_n  : [match_num, ldmk_numm, (w, h)]
        '''
        match_num, num_queries = out_probs.shape[:2]
        ldmk_num = tgt_coord.shape[-1]

        # cost1: 距离
        cost_dist = torch.cdist(out_coord, tgt_coord, p=2) # [c]
        # cost2: 类别
        cost_class = -out_probs[..., :self.landmark_num] # [batch_size, num_queries, ldmk_num] 0-23是关键点类别，24是背景类

        # Final cost matrix
        # C = cost_dist + cost_class
        C = cost_dist#cost_class + cost_dist
        C = C.detach().cpu().numpy()
        # 使用匈牙利算法匹配预测的关键点
        indices = [Tensor(np.array(linear_sum_assignment(c))).transpose(0,1) for c in C]
        indices = torch.stack(indices).to(out_probs.device) # [batch_size, ldmk_num, 2]
        indices = indices.to(torch.int64)

        return indices, cost_dist, cost_class

    def distribute_gt(self,
                      gt_results:GtResult):
                    #   gt_labels_list:list[Tensor],
                    #   gt_landmarks_list:list[Tensor],
                    #   gt_bboxes_n_list:list[Tensor])->dict[int, dict[str, Tensor]]:
        batch_num = len(gt_results)
        gt_distribution:dict[int, GtResult] = {}
        for bn in range(batch_num):
            items:GtResult = gt_results[bn]
            for distributed_item in items.squeeze():
                id_ = int(distributed_item.gt_class_ids)
                gt_distribution.setdefault(id_, GtResult())
                gt_distribution[id_] += distributed_item
        return gt_distribution

    def match_gt_with_pred(self,
                           pred_distribution:dict[int, PredResult],
                           gt_distribution:dict[int, GtResult]):
        matched_dict:dict[int, MatchedRoi] = {}
        for key in pred_distribution.keys():
            matched_dict.setdefault(key, MatchedRoi())
            pred: PredResult    = pred_distribution[key] #[pred_num_rois_in_batch, ...]
            gt: GtResult        = gt_distribution[key]   #[gt_num_rois_in_batch, ...]
            max_batch_num = int(pred.pred_batch_idx.max())
            # match the objects in the same batch
            for bn in range(max_batch_num+1):
                pred_idx    = torch.concat(torch.where(pred.pred_batch_idx == bn))
                gt_idx      = torch.concat(torch.where(gt.gt_batch_idx == bn))
                if pred_idx.numel() == 0 or gt_idx.numel() == 0:
                    continue
                pred_bbox_n = pred.pred_bboxes_n[pred_idx]
                gt_bbox_n = gt.gt_bboxes_n[gt_idx]
                # 根据giou匹配
                cost_matrix         = generalized_box_iou(pred_bbox_n, gt_bbox_n)
                row_ind, col_ind    = linear_sum_assignment(cost_matrix.cpu(), maximize=True)
                for ri, ci in zip(row_ind, col_ind):
                    score = cost_matrix[ri, ci] # 过滤，如果bbox的giou小于阈值，也认为匹配失败
                    if score > 0.5:
                        vaild_pred = pred[pred_idx[ri]]
                        vaild_gt = gt[gt_idx[ci]]
                        new_matched = MatchedRoi.create(vaild_pred, vaild_gt)
                        matched_dict[key] += new_matched

        return matched_dict

    def probs_loss(self, pred, target, with_weight = False):
        P_threshold = 0.4
        # 进一步处理target，大于P_threshold被记为正例，中间的丢弃
        target[target >= P_threshold] = 1.0
        target[target < P_threshold] = 0.0 #[o, m, N, C]
        valid_mask = torch.max(target, dim=-1)[0] == 1# [o, m, N] 必须属于一个类，否则不计算

        ###
        # if SYS == "Windows":
        #     plt.subplot(1,2,1)
        #     plt.imshow(tensor_to_numpy(pred[0,0]))
        #     plt.subplot(1,2,2)
        #     plt.imshow(tensor_to_numpy(target[0,0]))
        #     plt.show()
        ###

        if with_weight:
            weight = torch.sum(target, dim=(0,1,2)) #[C]
            weight[weight == 0] = 1.0
            weight = weight[-1] / weight #正负样本均衡
            weight = torch.clip(weight, 0, 10)
            ce = F.binary_cross_entropy(pred, target, weight, reduction='none') # [o, m, N]
        else:
            ce = F.binary_cross_entropy(pred, target, reduction='none') # [o, m, N]
        ce = torch.mean(ce, -1)
        probs_loss = torch.sum((ce * valid_mask), dim=-1) / torch.sum(valid_mask, dim=-1) # [o, m]
        # 处理 nan
        probs_loss[torch.isnan(probs_loss)] = 0.0
        return probs_loss

    def calculate_cluster_scores(self, pred_coord, target_coord):
        '''
        brief
        -----
        计算预测点的置信度目标

        parameters
        -----
        * pred_coord: #[match_num, output_num, num_queries, 2]
        * target_coord: #[match_num, gt_num, 2]
        
        return
        ----
        scores: #match_num, [output_num, pred_num, gt_num+1]
        '''
        scores_list = []
        for i in range(pred_coord.shape[1]):
            pc = pred_coord[:, i] #[match_num, num_queries, 2]
            scores = calculate_scores(pc, target_coord, alpha = self.alpha, beta = self.beta, eps = self.eps)
            scores_list.append(scores)
        return torch.stack(scores_list, dim=1)

    def calc_loss(self, matched_rois:MatchedRoi) -> LossResult:
        '''
        indices     : [output_num, match_num, ldmk_num, 2]
        pred_landmarks_probs    : [output_num, match_num, num_queries, (x, y)]
        pred_landmarks_n        : [output_num, match_num, num_queries, ldmk_num + 1]
        target_landmarks_n      : [match_num, ldmk_num, (x, y)]
        bbox                    : [match_num, (x1, y1, x2, y2)]
        pred_cdf                : [match_num]
        '''
        pred_landmarks_probs    = matched_rois.pred_landmarks_probs
        pred_landmarks_n        = matched_rois.pred_landmarks_coord
        bbox                    = denormalize_bbox(matched_rois.pred_bboxes_n, matched_rois.input_size)
        target_landmarks_n      = self.get_target(matched_rois.gt_landmarks, bbox)
        ### 匹配
        hungary = []
        for i in range(pred_landmarks_probs.shape[1]):
            hungary.append(
                self.match_landmarks(pred_landmarks_probs[:, i], pred_landmarks_n[:, i], target_landmarks_n)
                )
        ldmk_num = target_landmarks_n.shape[-2]
        
        ### 分别计算
        ### 形位损失
        out_coord_list = []
        tgt_coord_list = []
        for output_i, (indices, cost_dixt, _) in enumerate(hungary):
            pred_class_id_lmdk = indices[..., 0]   # [match_num, ldmk_num]
            target_class_id_lmdk = indices[..., 1] # [match_num, ldmk_num]由于target_landmarks的数量和顺序都是固定的，因此对应的索引就是类别

            batch_ind = torch.arange(0, pred_class_id_lmdk.shape[0], dtype=torch.int64)
            batch_ind = batch_ind.unsqueeze(-1).repeat(1, pred_class_id_lmdk.shape[1]).view(-1)
            pred_ind    = pred_class_id_lmdk.reshape(-1)    # [ldmk_num * match_num]
            target_ind  = target_class_id_lmdk.reshape(-1)  # [ldmk_num * match_num]

            out_coord_list.append(
                pred_landmarks_n[batch_ind, output_i, pred_ind].reshape(-1, ldmk_num, 2))
            tgt_coord_list.append(
                target_landmarks_n[batch_ind, target_ind].reshape(-1, ldmk_num, 2))
        out_coord = torch.stack(out_coord_list, dim = 1) #[match_num, output_num, num_queries, (x, y)]
        tgt_coord = torch.stack(tgt_coord_list, dim = 1) #[match_num, output_num, num_queries, (x, y)]
        # 按比例还原
        w = bbox[:, 2:3] - bbox[:, 0:1] #[match_num, 1]
        h = bbox[:, 3:4] - bbox[:, 1:2] #[match_num, 1]
        ratio = (w / h) #[match_num, 1]
        alpha = torch.sqrt(1/ratio) #[match_num, 1]
        restore = torch.concat([alpha * ratio, alpha], dim=-1).to(bbox.device) #[match_num, 2]
        restore = restore.unsqueeze(1).unsqueeze(1) #[match_num, 1, 1, 2]
        out_coord = out_coord * restore
        tgt_coord = tgt_coord * restore

        # 距离损失
        dist_loss  = torch.mean(torch.norm(out_coord - tgt_coord, dim = -1), dim=(-1)) #[match_num, output_num]
        ### 分类损失
        # 类别损失
        target_probs: Tensor = self.calculate_cluster_scores(pred_landmarks_n, target_landmarks_n).detach() #[match_num, output_num, num_queries, ldmk_num + 1]
        class_loss: Tensor = self.probs_loss(pred_landmarks_probs, target_probs, with_weight=True) #[match_num, output_num]
 
        # 正负样本损失，只考虑正样本判断的损失
        # target_obj = torch.stack([torch.sum(target_probs[...,:-1], dim = -1), target_probs[..., -1]], dim=-1) # [match_num, num_queries, 2]
        # pred_obj = torch.stack([torch.sum(pred_landmarks_probs[...,:-1], dim = -1), pred_landmarks_probs[..., -1]], dim=-1)
        # obj_loss = self.probs_loss(pred_obj, target_obj, with_weight=True) #[output_num, match_num]

        # 中间输出权重
        output_num = class_loss.shape[1]
        intermediate_loss_weights = torch.linspace(0.5, 1, output_num) if output_num > 1 else Tensor([1.0])
        intermediate_loss_weights = torch.square(intermediate_loss_weights).to(class_loss.device)

        # 生成损失结果
        loss_Tensor = torch.stack([dist_loss, class_loss]) #[2, match_num, output_num]
        loss_Tensor = torch.transpose(torch.transpose(loss_Tensor, 0, 2), 0, 1) # -> [output_num, match_num, 2] -> [match_num, output_num, 2]
        item_weights = torch.Tensor([self.dist_loss_w, self.class_loss_w])
        result = LossResult(loss_Tensor, item_weights, [LossKW.DIST, LossKW.CLS], intermediate_loss_weights)
        return result

    def forward(self,
                gt_results:GtResult,
                pred_distribution:dict[int, dict[str, Tensor]],
                loss_recoder:LandmarkLossRecorder = None):
        device = gt_results.device
        # distribute ground truth
        gt_distribution = self.distribute_gt(gt_results)
        # match ground truth & prediction
        matched = self.match_gt_with_pred(pred_distribution, gt_distribution)
        # calc loss
        loss_result_list:list[LossResult] = []
        for id_, matched_rois in matched.items():
            if len(matched_rois.gt_class_ids) == 0:
                continue
            loss_result: LossResult = self.calc_loss(matched_rois)
            loss_result_list.append(loss_result)
        
        total_loss_result: LossResult = LossResult.concat(loss_result_list)
        loss = total_loss_result.loss().to(device)
        if loss_recoder:
            loss_recoder.record(total_loss_result)
            loss_recoder.merge()
        return loss

if __name__ == "__main__":
    A = Tensor([[0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]])
    B = Tensor([[0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 1, 1]])
    r = generalized_box_iou(A, B)
    print()