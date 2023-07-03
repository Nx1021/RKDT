import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from models.loss import calculate_scores

from launcher.utils import compare_train_log
from models.utils import normalize_bbox
from models.loss import LandmarkLoss
import time

if __name__ == "__main__":
    compare_train_log(["20230629032403Linux", "20230629040700Linux"])
    # out_probs = torch.rand(6, 1, 3, 4).to("cuda")
    # target_probs = torch.rand(6, 1, 3, 4 rget_probs, reduction='none')[0])
    # print(F.binary_cross_entropy(out_prob[0], target_probs[0], reduction='none'))

    # bbox = torch.Tensor([[0.0, 0.0, 50.0, 50.0]]).repeat(1, 1)#.to("cuda")
    # ldml = (torch.rand(1, 24, 2)*50)#.to("cuda")
    # start = time.time()
    # for i in range(100000):
    #     LandmarkLoss.get_target(ldml, bbox)
    # print(time.time() - start)

