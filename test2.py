from __init__ import DATASETS_DIR, CFG_DIR, SERVER_DATASET_DIR
from posture_6d.data.dataset_example import VocFormat_6dPosture, UnifiedFilesHandle, ViewMeta
from posture_6d.data.dataset import SpliterGroup, Spliter
from posture_6d.core.utils import deserialize_object, serialize_object
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

morrison = VocFormat_6dPosture(os.path.join(DATASETS_DIR, "morrison_real_voc"), lazy=True)

names = {
  0: "bar_clamp",
  1: "gearbox",
  2: "nozzle",
  3: "part1",
  4: "part3",
  5: "pawn",
  6: "turbine_housing",
  7: "vase"
}

for i in [2, 2300, 9098, 4959]:
    if i == 2:
        show_ldmk = True
        plt.figure(figsize=(12, 9))
    else:
        show_ldmk = False
        plt.figure(figsize=(8, 6))
    v1 = morrison[i]
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    v1.plot(show_landmarks=show_ldmk, show_depth = False, exclude_class_ids=[8], obj_names=names)
    # plt.xlim(0, 640)
    # plt.ylim(0, 480)
    plt.savefig(f'output_plot_{i}.png', format='png', dpi=300)
    plt.savefig(f'output_plot_{i}.svg', format='svg', dpi=300)

