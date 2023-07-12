import OLDT_setup
import sys
from posture_6d.utils import JsonIO
from posture_6d.posture import Posture

Posture()

json_data = JsonIO.load_json(r"E:\shared\code\OLDT\datasets\morrison\models\models_info.json")
for k1, v1 in json_data.items():
    for k2, v2 in v1.items():
        v1[k2] = v2 * 1000
json_data = JsonIO.dump_json('path.json', json_data)

    
