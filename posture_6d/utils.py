import numpy as np
from _ctypes import PyObj_FromPtr
import json
import re

def get_bbox_connections(bbox_3d_proj:np.ndarray):
    '''
    bbox_3d_proj: [..., B, (x,y)]
    return
    -----
    lines: [..., ((x1,x2), (y1,y2)), 12]
    '''
    b = bbox_3d_proj
    lines = [
    ([b[...,0,0], b[...,1,0]], [b[...,0,1], b[...,1,1]]),
    ([b[...,0,0], b[...,6,0]], [b[...,0,1], b[...,6,1]]),
    ([b[...,6,0], b[...,7,0]], [b[...,6,1], b[...,7,1]]),
    ([b[...,1,0], b[...,7,0]], [b[...,1,1], b[...,7,1]]),

    ([b[...,2,0], b[...,3,0]], [b[...,2,1], b[...,3,1]]),
    ([b[...,2,0], b[...,4,0]], [b[...,2,1], b[...,4,1]]),
    ([b[...,4,0], b[...,5,0]], [b[...,4,1], b[...,5,1]]),
    ([b[...,3,0], b[...,5,0]], [b[...,3,1], b[...,5,1]]),

    ([b[...,0,0], b[...,2,0]], [b[...,0,1], b[...,2,1]]),
    ([b[...,1,0], b[...,3,0]], [b[...,1,1], b[...,3,1]]),
    ([b[...,7,0], b[...,5,0]], [b[...,7,1], b[...,5,1]]),
    ([b[...,6,0], b[...,4,0]], [b[...,6,1], b[...,4,1]]),
    ]
    lines = np.stack(lines)
    return lines #[12, ..., ((x1,x2), (y1,y2))]

class JsonIO():
    class _NoIndent(object):
        """ Value wrapper. """
        def __init__(self, value):
            self.value = value

    class _MyEncoder(json.JSONEncoder):
        FORMAT_SPEC = '@@{}@@'
        regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

        def __init__(self, **kwargs):
            # Save copy of any keyword argument values needed for use here.
            self.__sort_keys = kwargs.get('sort_keys', None)
            super(JsonIO._MyEncoder, self).__init__(**kwargs)

        def default(self, obj):
            return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, JsonIO._NoIndent)
                    else super(JsonIO._MyEncoder, self).default(obj))

        def encode(self, obj):
            format_spec = self.FORMAT_SPEC  # Local var to expedite access.
            json_repr = super(JsonIO._MyEncoder, self).encode(obj)  # Default JSON.

            # Replace any marked-up object ids in the JSON repr with the
            # value returned from the json.dumps() of the corresponding
            # wrapped Python object.
            for match in self.regex.finditer(json_repr):
                # see https://stackoverflow.com/a/15012814/355230
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                json_repr = json_repr.replace(
                                '"{}"'.format(format_spec.format(id)), json_obj_repr)

            return json_repr

    @staticmethod
    def __convert_dict_from_json(dictionary):
        new_dict = {}
        for key, value in dictionary.items():
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    pass
            
            if isinstance(value, list):
                array = np.array(value)
                if np.issubdtype(array.dtype, np.number):
                    new_value = array
                else:
                    new_value = [JsonIO.__convert_dict_from_json(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, dict):
                new_value = JsonIO.__convert_dict_from_json(value)
            else:
                new_value = value
            new_dict[key] = new_value
        return new_dict

    @staticmethod
    def __convert_dict_to_json(dictionary):
        new_dict = {}
        for key, value in dictionary.items():
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    pass
            
            if isinstance(value, np.ndarray):
                new_value = value.tolist()
                new_value = JsonIO._NoIndent(new_value)
            elif isinstance(value, list):
                new_value = [JsonIO.__convert_dict_to_json(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, dict):
                new_value = JsonIO.__convert_dict_to_json(value)
                if list(new_value.values()) == list(value.values()):
                    new_value = JsonIO._NoIndent(new_value)
            else:
                new_value = value
            new_dict[key] = new_value
        return new_dict

    @staticmethod
    def load_json(path, format = True):
        with open(path, 'r') as jf:
            dict_ = json.load(jf)
        if format:
            dict_ = JsonIO.__convert_dict_from_json(dict_)
        return dict_

    @staticmethod
    def dump_json(path, dict_):
        dict_ = JsonIO.__convert_dict_to_json(dict_)
        with open(path, 'w') as fw:
            # 整理格式，list部分不换行
            json_data = json.dumps(dict_, cls=JsonIO._MyEncoder, ensure_ascii=False, sort_keys=True, indent=2)
            fw.write(json_data)