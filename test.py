# import __init__
import sys
import __init__
from launcher.Trainer import Trainer
# from posture_6d.utils import JsonIO
# from posture_6d.posture import Posture
# Posture()
# from posture_6d.dataset_format import VocFormat, LinemodFormat, _LinemodFormat_sub1
# from tqdm import tqdm
# import time

# file = "scene_trans_vector.json"
# dict_ = JsonIO.load_json(r"E:\shared\code\OLDT\datasets\morrison\{}".format(file))
# test_dict = dict_
# # test_dict = JsonIO.load_json("scene_bbox_3d.json")
# for k, v in test_dict.items():
#     for kk, vv in v.items():
#         v[kk] = vv.reshape(-1,3)
#         v[kk][1] *= 1000


# start = time.time()
# JsonIO.dump_json(file, test_dict, True)
# print(time.time() - start)
# # start = time.time()
# # JsonIO.dump_json("test_json.json", test_dict, False)
# # print(time.time() - start)

# # lf = _LinemodFormat_sub1(r"E:\shared\code\ObjectDatasetTools\LINEMOD\000019\aug_output")
# # vf = VocFormat(r"E:\shared\code\OLDT\datasets\morrison_")
# # vf.clear(True)
# # with vf.start_to_write():
# #     for data_i, viewmeta in tqdm(enumerate(lf.read_from_disk())):
# #         for v in viewmeta.extr_vecs.values():
# #             v[1] *= 1000
# #         viewmeta.modify_class_id([(20, 0), 
# #                                   (21, 1),
# #                                   (22, 2),
# #                                   (23, 3),
# #                                   (24, 4),
# #                                   (25, 5),
# #                                   (26, 6),
# #                                   (27, 7),
# #                                   (28, 8)])
# #         vf.write_to_disk(viewmeta, data_i)


# from .grasp_coord import Gripper