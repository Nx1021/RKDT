from ultralytics.yolo.utils import ops, yaml_load
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_area
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from models.roi_handling import LandmarkDetectionResult
import platform
import matplotlib.pyplot as plt

def calculate_scores(pk, points):
    alpha = 0.15
    beta = 0.4
    eps = 1e-4

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
    A: 点集 A，形状为 (k, N, 2) 的 torch.Tensor，k 表示有 k 组点集，N 是每组点集的点的数量
    B: 点集 B，形状为 (k, N, 2) 的 torch.Tensor
    返回值：
    rotation_angles: 旋转角向量，形状为 (k,) 的 torch.Tensor
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


class LandmarkLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = yaml_load(cfg)
        self.landmark_num:int = self.cfg["landmark_num"]
        self.calc_intermediate_loss:bool = self.cfg["calc_intermediate_loss"]

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

    def calc_loss(self, indices, out_probs, out_coord, tgt_coord):
        '''
        indices     : [match_num, ldmk_num, 2]
        out_probs   : [match_num, num_queries, (w, h)]
        out_coord   : [match_num, num_queries, ldmk_num + 1]
        tgt_coord   : [match_num, ldmk_num, (w, h)]
        '''
        match_num = out_probs.shape[0]
        ldmk_num = tgt_coord.shape[-2]
        ### 计算损失
        # out_probs = out_probs.transpose(2, 1) # [batch_size, ldmk_num + 1, num_queries]
        pred_class_id_lmdk = indices[..., 0]   # [batch_size, ldmk_num]
        target_class_id_lmdk = indices[..., 1] # [batch_size, ldmk_num]由于target_landmarks的数量和顺序都是固定的，因此对应的索引就是类别

        batch_ind = torch.arange(0, pred_class_id_lmdk.shape[0], dtype=torch.int64)
        batch_ind = batch_ind.unsqueeze(-1).repeat(1, pred_class_id_lmdk.shape[1]).view(-1)
        pred_ind    = pred_class_id_lmdk.reshape(-1)    # [ldmk_num * batch_size]        
        target_ind  = target_class_id_lmdk.reshape(-1)  # [ldmk_num * batch_size]

        target_class_id = torch.full(out_probs.shape[:2], self.landmark_num,
                            dtype=torch.int64, device=out_probs.device) # [batch_size, num_queries]
        target_class_id[batch_ind, pred_ind] = target_ind
        
        ### 计算损失 ###
        # 类别损失
        target_probs = calculate_scores(out_coord, tgt_coord).detach() #[batch_size, num_queries, ldmk_num + 1]
        class_loss = F.binary_cross_entropy(out_probs, target_probs) * match_num
        # 正负样本损失，只考虑正样本判断的损失
        target_obj = torch.stack([torch.sum(target_probs[...,:-1], dim = -1), target_probs[..., -1]], dim=-1) # [batch_size, num_queries, 2] 
        target_obj = torch.argmax(target_obj, dim=-1) # [batch_size, num_queries]
        obj_weight = torch.Tensor([1.0, 1.0]).to(target_obj.device).detach() # 正负样本均衡 [batch_size]
        pred_obj = torch.stack([torch.sum(out_probs[...,:-1], dim = -1), out_probs[..., -1]], dim=-1)
        obj_loss = F.cross_entropy(pred_obj.transpose(-1, 1), target_obj, weight = obj_weight) * match_num # [batch_size, num_queries] TODO: 正负样本均衡
        # 距离损失
        out_coord_matched = out_coord[batch_ind, pred_ind].reshape(-1, ldmk_num, 2)
        tgt_coord_matched = tgt_coord[batch_ind, target_ind].reshape(-1, ldmk_num, 2)
        dist_loss  = torch.mean(torch.norm(out_coord_matched - tgt_coord_matched, dim = -1)) * match_num        
        # # 总体旋转损失，由于在初始训练时可能很大，因此它将被约束在一定范围内
        # rotation_angles = torch.abs(find_best_rotation(out_coord_matched, tgt_coord_matched))
        # rotation_angles = torch.clip(rotation_angles, 0, torch.pi/12)


        return class_loss + dist_loss * 5 + obj_loss # + rotation_angles * 5

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
        indices = [torch.Tensor(np.array(linear_sum_assignment(c))).transpose(0,1) for c in C]
        indices = torch.stack(indices).to(out_probs.device) # [batch_size, ldmk_num, 2]
        indices = indices.to(torch.int64)

        return indices


    #     ###
    #     if platform.system() == "Windows":
    #         print(indices[0].detach().cpu().numpy())
    #         out_coord_numpy = out_coord.detach().cpu().numpy()
    #         tgt_coord_numpy = tgt_coord.cpu().numpy()
    #         plt.subplot(2,2,1)
    #         plt.scatter(out_coord_numpy[0,:,0], out_coord_numpy[0,:,1], c = "b")
    #         for i in range(out_coord_numpy.shape[1]):
    #             plt.text(out_coord_numpy[0,i,0], out_coord_numpy[0,i,1], str(i))
    #         plt.scatter(tgt_coord_numpy[0,:,0], tgt_coord_numpy[0,:,1], c = "r")
    #         for i in range(tgt_coord_numpy.shape[1]):
    #             plt.text(tgt_coord_numpy[0,i,0], tgt_coord_numpy[0,i,1], str(i))
    #         plt.axis('equal')
    #         plt.axes([0,0,1,1])
            

    #         plt.subplot(2,2,2)
    #         plt.imshow(target_probs.detach().cpu()[0])

    #         plt.subplot(2,2,3)
    #         plt.imshow(torch.eye(25)[target_class_id].detach().cpu().numpy()[0])

    #         plt.subplot(2,2,4)
    #         plt.imshow(out_probs.detach().cpu().numpy()[0])
    #         plt.show()
    #     ###


    #     return class_loss + dist_loss * 5 + obj_loss


    def forward(self, 
                gt_landmarks_list:list[Tensor], 
                gt_bboxes_n_list:list[Tensor], 
                pred_results_list:list[list[LandmarkDetectionResult]]):
        '''
        parameters
        -----
        '''
        BN = len(pred_results_list)
        all_batch_loss = torch.Tensor([0.0]).to(gt_landmarks_list[0].device)
        all_batch_last_loss = torch.Tensor([0.0]).to(gt_landmarks_list[0].device)
        group_matched_num = 0
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
            intermediate_loss = []
            weights_sum = []
            intermediate_loss_weights = torch.linspace(0.5, 1, output_num) if output_num > 1 else torch.Tensor([1.0])
            intermediate_loss_weights = torch.square(intermediate_loss_weights).to(matched_pred_landmarks_probs.device)
            selected_output_idx = list(range(output_num))
            if not self.calc_intermediate_loss:
                selected_output_idx = [selected_output_idx[-1]]
            for i in selected_output_idx:
                out_probs = matched_pred_landmarks_probs[i]
                out_coord = matched_pred_landmarks_n[i]
                tgt_coord = target_landmarks_n[i]
                weight = intermediate_loss_weights[i]
                indices = self.match_landmarks(out_probs, out_coord, tgt_coord)
                loss = self.calc_loss(indices, out_probs, out_coord, tgt_coord)
                intermediate_loss.append(loss*weight)
                weights_sum.append(weight)
            loss = torch.stack(intermediate_loss) 
            all_batch_loss += torch.sum(loss) / torch.sum(torch.stack(weights_sum))
            all_batch_last_loss += torch.sum(intermediate_loss[-1]) # 最后一层的损失
        if group_matched_num > 0:
            all_batch_loss = all_batch_loss / group_matched_num
            all_batch_last_loss = all_batch_last_loss / group_matched_num
        return all_batch_loss, all_batch_last_loss, group_matched_num

if __name__ == "__main__":
    A = torch.Tensor([[0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]])
    B = torch.Tensor([[0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 1, 1]])
    r = generalized_box_iou(A, B)
    print()