from utils.yaml import yaml_load
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_area
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from models.roi_pipeline import LandmarkDetectionResult
import platform
import matplotlib.pyplot as plt

from typing import Union

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

class LossItem():
    def __init__(self, name, value, weight) -> None:
        self.name = name
        self.weight = weight
        self.value:Tensor = value
    
    @property
    def loss(self):
        return self.value * self.weight

class LossItemGroup():
    class_loss_w        = 1.0
    dist_loss_w         = 1.0
    obj_loss_w          = 1.0
    rotation_loss_w     = 1.0
    def __init__(self, class_loss, dist_loss, obj_loss, rotation_loss, weight = 1.0) -> None:
        self.list:list[LossItem] = [LossItem("Class Loss",      class_loss   , LossItemGroup.class_loss_w),
                                    LossItem("Dist Loss",       dist_loss    , LossItemGroup.dist_loss_w),
                                    LossItem("Obj Loss",        obj_loss     , LossItemGroup.obj_loss_w),
                                    LossItem("Rotation Loss",   rotation_loss, LossItemGroup.rotation_loss_w)]
        self.weight = weight
    
    def set_weight(self, weight):
        self.weight = weight

    @staticmethod
    def mean(groups:list["LossItemGroup"]) -> "LossItemGroup":
        weights_sum = torch.sum(torch.stack([g.weight for g in groups]))
        items_loss = torch.stack(
            [torch.stack([lossitem.value*g.weight for lossitem in g.list]
                                         ) for g in groups]).to(weights_sum.device)
        mean = torch.sum(items_loss, dim=0) / weights_sum
        return LossItemGroup(mean[0], mean[1], mean[2], mean[3])

    def sum(self) -> Tensor:
        return torch.sum(
            torch.stack([x.loss for x in self.list]), 
            dim=0) * self.weight

    def to_dict(self) -> dict:
        dict_ = {}
        for item in self.list:
            dict_[item.name] = torch.sum(item.loss).item()
        dict_["Last Decoder Loss"] = torch.sum(self.sum()).item()
        return dict_

class LandmarkLossRecorder():
    def __init__(self, name, top = True) -> None:
        self.name = name
        self.loss_sum:Tensor = torch.Tensor([0.0])
        self.others_sum:dict[float] = {}
        self.detect_num:int = 0
        if top:
            self.buffer = LandmarkLossRecorder("__buffer", top=False)

    def _add_loss(self, loss:Tensor):
        try:
            loss = torch.sum(loss)
        except TypeError:
            return
        if self.loss_sum.device != loss.device:
            self.loss_sum = self.loss_sum.to(loss.device)
        self.loss_sum += loss

    def record(self, loss:Tensor, other:LossItemGroup):
        self.buffer._add_loss(loss)
        for key, value in other.to_dict().items():
            self.buffer.others_sum.setdefault(key, 0.0)
            self.buffer.others_sum[key] += value
        self.buffer.detect_num += loss.shape[0]

    def clear(self):
        self.loss_sum = torch.Tensor([0.0])
        self.others_sum.clear()
        self.detect_num = 0

    def mean(self) -> Tensor:
        return self.__get_mean(self.loss_sum)

    def __get_mean(self, value:Union[Tensor, float]):
        if self.detect_num == 0:
            if isinstance(value, Tensor):
                return Tensor([0.0]).to(value.device)
            else:
                return 0.0
        else:
            return value / self.detect_num

    def to_dict(self):
        dict_:dict[str, float] = {}
        for key, sum_value in self.others_sum.items():
            dict_[self.name + ' ' + key] = self.__get_mean(sum_value)
        dict_[self.name + ' ' + "Loss"] = self.mean().item()

        return dict_

    def merge(self):
        buffer = self.buffer

        self._add_loss(buffer.loss_sum.detach())
        self.detect_num += buffer.detect_num
        for key, value in buffer.others_sum.items():
            self.others_sum.setdefault(key, 0.0)
            self.others_sum[key] += value

class LandmarkLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = yaml_load(cfg)
        self.landmark_num:int = self.cfg["landmark_num"]
        self.calc_intermediate:bool = self.cfg["calc_intermediate"]
        LossItemGroup.class_loss_w      = self.cfg["class_loss_w"]         
        LossItemGroup.dist_loss_w       = self.cfg["dist_loss_w"]
        LossItemGroup.obj_loss_w        = self.cfg["obj_loss_w"]
        LossItemGroup.rotation_loss_w   = self.cfg["rotation_loss_w"]

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

    def get_target(self, gt_landmarks:Tensor, row_ind, col_ind, output_num:int, rlts:list[LandmarkDetectionResult]):
        pred_bbox           = torch.stack([x.bbox for x in rlts])   # [num_roi_active, 4]
        target_landmarks    = gt_landmarks[col_ind]   # [match_num, ldmk_num, 2]
        matched_pred_bbox   = pred_bbox[row_ind]      # [match_num, 4]
        matched_lt          = matched_pred_bbox[:, :2] # [match_num, (x, y)]
        matched_wh          = matched_pred_bbox[:, 2:] - matched_pred_bbox[:, :2] # [match_num, (w, h)]
        matched_lt          = matched_lt.unsqueeze(1).repeat(1, self.landmark_num, 1) # [match_num, ldmk_num, (w, h)]
        matched_wh          = matched_wh.unsqueeze(1).repeat(1, self.landmark_num, 1) # [match_num, ldmk_num, (w, h)]
        target_landmarks_n  = (target_landmarks - matched_lt) / matched_wh # [match_num, ldmk_num, (w, h)]
        target_landmarks_n  = target_landmarks_n.unsqueeze(0).repeat(output_num, 1, 1, 1) # [output_num, match_num, ldmk_num, (w, h)]

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

    def calc_loss(self, indices, out_probs, out_coord, tgt_coord, bbox) -> LossItemGroup:
        '''
        indices     : [match_num, ldmk_num, 2]
        out_probs   : [match_num, num_queries, (w, h)]
        out_coord   : [match_num, num_queries, ldmk_num + 1]
        tgt_coord   : [match_num, ldmk_num, (w, h)]
        bbox        : [match_num, (x1, y1, x2, y2)]
        '''
        match_num = out_probs.shape[0]
        ldmk_num = tgt_coord.shape[-2]
        # out_probs = out_probs.transpose(2, 1) # [match_num, ldmk_num + 1, num_queries]
        pred_class_id_lmdk = indices[..., 0]   # [match_num, ldmk_num]
        target_class_id_lmdk = indices[..., 1] # [match_num, ldmk_num]由于target_landmarks的数量和顺序都是固定的，因此对应的索引就是类别

        batch_ind = torch.arange(0, pred_class_id_lmdk.shape[0], dtype=torch.int64)
        batch_ind = batch_ind.unsqueeze(-1).repeat(1, pred_class_id_lmdk.shape[1]).view(-1)
        pred_ind    = pred_class_id_lmdk.reshape(-1)    # [ldmk_num * match_num]        
        target_ind  = target_class_id_lmdk.reshape(-1)  # [ldmk_num * match_num]
        
        ### 计算损失 ###
        ### 分类损失
        # 类别损失
        target_probs = calculate_scores(out_coord, tgt_coord).detach() #[match_num, num_queries, ldmk_num + 1]
        class_loss = F.binary_cross_entropy(out_probs, target_probs, reduction='none')
        class_loss = torch.mean(class_loss, (-1, -2)) #[match_num]
        # 正负样本损失，只考虑正样本判断的损失
        target_obj = torch.stack([torch.sum(target_probs[...,:-1], dim = -1), target_probs[..., -1]], dim=-1) # [match_num, num_queries, 2] 
        # target_obj = torch.argmax(target_obj, dim=-1) # [match_num, num_queries]
        obj_weight = Tensor([1.0, 1.0]).to(target_obj.device).detach() # 正负样本均衡 [match_num]
        pred_obj = torch.stack([torch.sum(out_probs[...,:-1], dim = -1), out_probs[..., -1]], dim=-1)
        obj_loss = F.binary_cross_entropy(pred_obj, target_obj, weight = obj_weight, reduction="none") # [match_num, num_queries] TODO: 正负样本均衡
        obj_loss = torch.mean(obj_loss, (-1, -2)) #[match_num]
        ### 形位损失
        out_coord_matched = out_coord[batch_ind, pred_ind].reshape(-1, ldmk_num, 2)
        tgt_coord_matched = tgt_coord[batch_ind, target_ind].reshape(-1, ldmk_num, 2)       
        w = bbox[:, 2:3] - bbox[:, 0:1] #[match_num, 1]
        h = bbox[:, 3:4] - bbox[:, 1:2] #[match_num, 1]
        ratio = (w / h) #[match_num, 1]
        alpha = torch.sqrt(1/(torch.square(ratio) + 1)) #[match_num, 1]
        out_coord_matched = out_coord_matched * torch.concat([alpha * ratio, alpha], dim=-1).unsqueeze(1).to(bbox.device)
        tgt_coord_matched = tgt_coord_matched * torch.concat([alpha * ratio, alpha], dim=-1).unsqueeze(1).to(bbox.device)
        # 距离损失
        dist_loss  = torch.mean(torch.norm(out_coord_matched - tgt_coord_matched, dim = -1), dim=-1)
        # 总体旋转损失，由于在初始训练时可能很大，因此它将被约束在一定范围内
        # rotation_loss = torch.abs(find_best_rotation(out_coord_matched, tgt_coord_matched))
        # rotation_loss = torch.clip(rotation_loss, 0, torch.pi/12)

        # obj_loss = torch.Tensor([0.0]).to(class_loss.device)
        rotation_loss = torch.zeros(class_loss.shape).to(class_loss.device)

        # group = LossItemGroup(class_loss, dist_loss, obj_loss, rotation_loss)
        group = class_loss + dist_loss + obj_loss
        return group

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
        C = cost_dist + cost_class
        C = C.detach().cpu().numpy()
        # 使用匈牙利算法匹配预测的关键点
        indices = [Tensor(np.array(linear_sum_assignment(c))).transpose(0,1) for c in C]
        indices = torch.stack(indices).to(out_probs.device) # [batch_size, ldmk_num, 2]
        indices = indices.to(torch.int64)

        return indices

    def forward(self, 
                gt_landmarks_list:list[Tensor], 
                gt_bboxes_n_list:list[Tensor], 
                pred_results_list:list[list[LandmarkDetectionResult]],
                loss_recoder:LandmarkLossRecorder):
        '''
        parameters
        -----
        '''
        BN = len(pred_results_list)
        group_matched_num = 0
        loss = []
        for bn in range(BN):
            rlts = pred_results_list[bn]
            if len(rlts) == 0:
                continue
            ### 将关键点真值（pixel）在预测的bboxes内归一化
            # with torch.no_grad():
            row_ind, col_ind = self.match_roi(gt_bboxes_n_list[bn],  #[num_roi_gt, 4]
                                            rlts) 
            group_matched_num += len(row_ind) 
            if len(row_ind) == 0:
                continue # 未能找到匹配的roi

            (matched_pred_landmarks_n, matched_pred_landmarks_probs), output_num = self.get_pred(row_ind, rlts)
            target_landmarks_n = self.get_target(gt_landmarks_list[bn], 
                                                    row_ind, col_ind, output_num, 
                                                    rlts).detach().to(torch.float32)
            ### 计算损失
            intermediate_loss:list[LossItemGroup] = []
            intermediate_loss_weights = torch.linspace(0.5, 1, output_num) if output_num > 1 else Tensor([1.0])
            intermediate_loss_weights = torch.square(intermediate_loss_weights).to(matched_pred_landmarks_probs.device)
            selected_output_idx = list(range(output_num))
            if not self.calc_intermediate:
                selected_output_idx = [selected_output_idx[-1]]
            for i in selected_output_idx:
                out_probs = matched_pred_landmarks_probs[i]
                out_coord = matched_pred_landmarks_n[i]
                tgt_coord = target_landmarks_n[i]
                weight = intermediate_loss_weights[i]
                indices = self.match_landmarks(out_probs, out_coord, tgt_coord)
                # loss_group:LossItemGroup = self.calc_loss(indices, out_probs, out_coord, tgt_coord, gt_bboxes_n_list[bn][col_ind])
                # loss_group.set_weight(weight)
                # intermediate_loss.append(loss_group)
                loss_group:Tensor = self.calc_loss(indices, out_probs, out_coord, tgt_coord, gt_bboxes_n_list[bn][col_ind])
                loss.append(torch.mean(loss_group))
        #     # 计算总损失、末位decoder损失、末尾各部分损失
        #     loss:Tensor = LossItemGroup.mean(intermediate_loss).sum()
        #     loss_recoder.record(loss, intermediate_loss[-1])
        # loss = torch.mean(loss_recoder.buffer.mean())
        loss_recoder.buffer.detect_num = group_matched_num
        loss = torch.mean(torch.stack(loss))
        loss_recoder.buffer.loss_sum = loss * group_matched_num
        loss_recoder.merge()
        return loss

if __name__ == "__main__":
    A = Tensor([[0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]])
    B = Tensor([[0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 1, 1]])
    r = generalized_box_iou(A, B)
    print()