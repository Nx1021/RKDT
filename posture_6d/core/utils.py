import os
import numpy as np
from _ctypes import PyObj_FromPtr
import json
from json import JSONDecodeError
import re
from typing import Any
import time
import io
import re
import warnings

from typing import Generic, TypeVar, Union, Callable, Iterable, Type
import types
from collections import OrderedDict

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

def modify_class_id(dict_list:list[dict[int, Any]], modify_class_id_pairs:list[tuple[int]]):
    orig_keys = [x[0] for x in modify_class_id_pairs]
    new_keys  = [x[1] for x in modify_class_id_pairs]
    assert len(orig_keys) == len(set(orig_keys))
    assert len(new_keys)  == len(set(new_keys))
    assert all([len(x) == 2 for x in modify_class_id_pairs])
    for orig_dict in dict_list:
        new_dict = {}
        for pair in modify_class_id_pairs:
            if pair[0] in orig_dict:
                new_dict[pair[1]] = orig_dict[pair[0]]
        orig_dict.clear()
        orig_dict.update(new_dict)    

def get_meta_dict(obj):
    orig_dict_list = []
    for name, orig_dict in vars(obj).items():
        if isinstance(orig_dict, dict) and all([isinstance(x, int) for x in orig_dict.keys()]):
            orig_dict_list.append(orig_dict)
    return orig_dict_list

def extract_doc(doc:str, title:str):
    idx = doc.find(title)
    sub_doc = doc[idx:]
    idx = re.search(r'\n\s*?\n', sub_doc).start()
    sub_doc = sub_doc[:idx]
    return sub_doc


def _ignore_warning(func, category = Warning):
    def warpper(*args, **kwargs):
        warnings.filterwarnings("ignore", category=category) # do not show warning of image size
        rlt = func(*args, **kwargs)
        warnings.filterwarnings("default", category=category) # recover warning of image size
        return rlt
    return warpper


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

    class Stream():
        def __init__(self, path, open = False, buffer_length = 100000) -> None:
            self.path = path
            self.buffer = ""
            self.buffer_length = buffer_length
            self._closed = True
            if open:
                self.open()

        @property
        def closed(self):
            return self._closed
        
        @closed.setter
        def closed(self, value):
            value = bool(value)
            if value == True:
                self.close()
            else:
                self.open()
            self._closed = value

        def open(self):
            # 
            if not self.closed:
                return
            print("open JsonIO stream of {}".format(self.path))
            if os.path.exists(self.path):
                try:
                    with open(self.path, 'rb+') as f:
                        f.seek(-3, 2)
                        f.truncate()
                    with open(self.path, 'a') as f:
                        f.write(",")
                except OSError:
                    pass
            else:
                with open(self.path, 'w') as f:
                    f.write("{")   
            self._closed = False         

        def close(self):
            if self.closed:
                return
            print("close JsonIO stream of {}".format(self.path))
            self.save_buffer()
            with open(self.path, 'rb+') as f:
                f.seek(-1, 2)
                f.truncate()
            with open(self.path, 'a') as f:
                f.write('\n}')
            self._closed = True

        def save_buffer(self):
            with open(self.path, 'a') as f:
                f.write(self.buffer)
            self.buffer = ""            

        def write(self, to_dump_dict):
            string = JsonIO._dumps(to_dump_dict)
            self.buffer += string
            if len(self.buffer) > self.buffer_length:
                self.save_buffer()

        def __del__(self):
            self.close()

    @staticmethod
    def create_stream(path):
        stream = JsonIO.Stream(path)
        return stream

    @staticmethod
    def __convert_formatdict_from_json(dictionary, cvt_list_to_array = True):
        def cvt_key(key):
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    pass
            return key

        def cvt_value(value):
            if isinstance(value, list):
                try:
                    if cvt_list_to_array:
                        array = np.array(value)
                        if np.issubdtype(array.dtype, np.number):
                            new_value = array
                        else:
                            raise ValueError
                    else:
                        new_value = value
                except ValueError:
                    new_value = []
                    for item in value:
                        if isinstance(item, dict):
                            item = JsonIO.__convert_formatdict_from_json(item, cvt_list_to_array)
                        else:
                            item = cvt_value(item)
                        new_value.append(item)
            elif isinstance(value, dict):
                new_value = JsonIO.__convert_formatdict_from_json(value, cvt_list_to_array)
            else:
                new_value = value
            return new_value

        new_dict = {}
        for key, value in dictionary.items():
            new_key = cvt_key(key)
            new_value = cvt_value(value)

            new_dict[new_key] = new_value
        return new_dict

    @staticmethod
    def __convert_dict_to_jsonformat(dictionary, regard_list_as_array = False):
        def cvt_key(key):
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    pass
            if isinstance(key, np.intc):
                key = int(key)
            return key

        def cvt_value(value):
            if isinstance(value, np.intc):
                new_value = int(value)
            if isinstance(value, np.ndarray):
                new_value = np.around(value, decimals=4).tolist()
                new_value = JsonIO._NoIndent(new_value)
            elif isinstance(value, list):
                if regard_list_as_array:
                    new_value = value
                    new_value = JsonIO._NoIndent(new_value)
                else:
                    new_value = []
                    for item in value:
                        if isinstance(item, dict):
                            item = JsonIO.__convert_dict_to_jsonformat(item, regard_list_as_array)
                        else:
                            item = cvt_value(item)
                        new_value.append(item)
            elif isinstance(value, dict):
                new_value = JsonIO.__convert_dict_to_jsonformat(value, regard_list_as_array)
                if not any([isinstance(x, JsonIO._NoIndent) for x in new_value.values()]) and\
                list(new_value.values()) == list(value.values()):
                    new_value = JsonIO._NoIndent(new_value)
            else:
                new_value = value
            return new_value

        new_dict = {}
        for key, value in dictionary.items():
            new_key = cvt_key(key)
            new_value = cvt_value(value)
            new_dict[new_key] = new_value
        return new_dict

    @staticmethod
    def load_json(path, format = True, cvt_list_to_array = True):
        with open(path, 'r') as jf:
            dict_ = json.load(jf)
        if format:
            dict_ = JsonIO.__convert_formatdict_from_json(dict_, cvt_list_to_array)
        return dict_

    @staticmethod
    def _dumps(to_dump_dict, regard_list_as_array = False):
        to_dump_dict = JsonIO.__convert_dict_to_jsonformat(to_dump_dict, regard_list_as_array)
        string = ""            
        for k, v in to_dump_dict.items():
            json_data = json.dumps({k: v}, cls=JsonIO._MyEncoder, ensure_ascii=False, sort_keys=True, indent=2)
            string += json_data[1:-2] + ','
        return string

    @staticmethod
    def dump_json(path, to_dump_dict, regard_list_as_array = False):
        string = JsonIO._dumps(to_dump_dict, regard_list_as_array)
        string = '{' + string[:-1] + '\n}'
        with open(path, 'w') as fw:
            fw.write(string)

KT = TypeVar("KT")
VT = TypeVar("VT")
def search_in_dict(_dict:dict[KT, VT], 
                   key:Union[int, str], 
                   process_func:Callable = None):
    key = int_str_cocvt(_dict.keys(), key, return_index=False, process_func=process_func)
    return _dict[key]

def int_str_cocvt(ref_iterable:Iterable[str], 
                   query:Union[int, str], 
                   return_index=False, 
                   process_func:Callable = None):
    def default_func(string):
        return string 

    ref_iterable = tuple(ref_iterable)

    if process_func is None:
        process_func:Callable = default_func

    if isinstance(query, int):
        key = ref_iterable[query]
        return query if return_index else key
    elif isinstance(query, str):
        if query in ref_iterable:
            return ref_iterable.index(query) if return_index else query
        else:
            matching_keys = [query == process_func(k) for k in ref_iterable]
            if sum(matching_keys) == 1:
                idx = matching_keys.index(True)
                return idx if return_index else ref_iterable[idx]
            else:
                raise ValueError("Key not found or ambiguous")
    else:
        raise TypeError("Key must be an int or str")


ITEM = TypeVar('ITEM')
ROWKEYT = TypeVar('ROWKEYT', int, str)
COLKETT = TypeVar('COLKETT', int, str)
class Table(Generic[ROWKEYT, COLKETT, ITEM]):
    '''
    # 使用示例
    table = Table()
    table.add_column("Name")
    table.add_column("Age")
    table.add_row("Row1")
    table.add_row("Row2")

    # 增加数据
    table["Row1", "Name"] = "Alice"
    table["Row1", "Age"] = 25
    table["Row2", "Name"] = "Bob"
    table["Row2", "Age"] = 30

    print(table.get_row("Row1"))  # 输出: {'Name': ['Alice'], 'Age': [25]}
    print(table.get_column("Age"))  # 输出: {'Row1': [25], 'Row2': [30]}

    # 删除数据
    del table["Row1", "Age"]
    print(table.get_row("Row1"))  # 输出: {'Name': ['Alice'], 'Age': []}

    '''
    def __init__(self, row_names:list[ROWKEYT] = None, col_names:list[COLKETT] = None, default_value_type:type[ITEM] = None,
                 *, 
                 row_name_type = str, 
                 col_name_type = str):
        self.__data:dict[ROWKEYT, dict[COLKETT, ITEM]] = {}
        self.__row_names:list[ROWKEYT] = []
        self.__col_names:list[COLKETT] = []
        self.__default_value_type = default_value_type
        self.__row_name_type:Type[ROWKEYT] = row_name_type
        self.__col_name_type:Type[COLKETT] = col_name_type

        for row_name in row_names or []:
            self.add_row(row_name)
        for col_name in col_names or []:
            self.add_column(col_name)

    @staticmethod
    def __type_process(self, value:Union[str, type], orig_list:list, orig_flag):
        if isinstance(value, str):
            assert value in ['int', 'str']
            if value == 'int':
                value = int
            elif value == 'str':
                value = str
        assert isinstance(value, type)
        if orig_flag != value:
            assert all([isinstance(x, value) for x in orig_list]), "row_name_type cannot be changed"
        return value

    @staticmethod
    def __name_filter(name, _type, _names):
        if _type == int:
            name = int(name)
        elif _type == str:
            name = int_str_cocvt(_names, name)
        return name

    @staticmethod
    def __key_assert(name, target:list[ROWKEYT], is_required, ignore = False):
        if ((name in target) ^ is_required):
            if ignore:
                return False
            else:
                raise ValueError(f"Key '{name}' {'does not exist' if is_required else 'already exists'}")
        return True

    @property
    def data(self):
        return types.MappingProxyType(self.__data)
    
    @property
    def row_names(self):
        return tuple(self.__row_names)
    
    @property
    def col_names(self):
        return tuple(self.__col_names)
    
    @property
    def default_value_type(self):
        return self.__default_value_type
    
    @default_value_type.setter
    def default_value_type(self, value):
        assert callable(value), "default_value_type must be callable"
        self.__default_value_type = value


    @property
    def row_name_type(self):
        return self.__row_name_type

    @property
    def col_name_type(self):
        return self.__col_name_type    

    @row_name_type.setter
    def row_name_type(self, value:Union[str, type]):
        self.__row_name_type = self.__type_process(value, self.__row_names, self.__row_name_type)
    
    @col_name_type.setter
    def col_name_type(self, value:Union[str, type]):
        self.__col_name_type = self.__type_process(value, self.__col_names, self.__col_name_type)

    @property
    def empty(self):
        return len(self.__data) == 0

    def gen_default_value(self)->ITEM:
        if self.__default_value_type is not None:
            return self.__default_value_type()
        return None


    def _row_name_filter(self, row_name:ROWKEYT) -> ROWKEYT:
        return self.__name_filter(row_name, self.__row_name_type, self.__row_names)
    
    def _col_name_filter(self, col_name:COLKETT) -> COLKETT:
        return self.__name_filter(col_name, self.__col_name_type, self.__col_names)

    def add_row(self, row_name:ROWKEYT, exist_ok=False):
        assert isinstance(row_name, self.row_name_type), "row_name must be an instance of row_name_type"
        if self.__key_assert(row_name, self.__row_names, False, ignore=exist_ok):
            self.__row_names.append(row_name)
            self.__data[row_name] = {col_name: self.gen_default_value() for col_name in self.__col_names}

    def remove_row(self, row_name:Union[int,str], not_exist_ok=False):
        row_name = self._row_name_filter(row_name)
        if self.__key_assert(row_name, self.__row_names, True, ignore=not_exist_ok):
            del self.__data[row_name]
            self.__row_names.remove(row_name)

    def add_column(self, col_name:COLKETT, exist_ok=False):
        assert isinstance(col_name, self.col_name_type)
        if self.__key_assert(col_name, self.__col_names, False, ignore=exist_ok):
            self.__col_names.append(col_name)
            for row_name in self.__row_names:
                self.__data[row_name][col_name] = self.gen_default_value()

    def remove_column(self, col_name:Union[int,str], not_exist_ok=False):
        col_name = self._col_name_filter(col_name)
        if self.__key_assert(col_name, self.__col_names, True, ignore=not_exist_ok):
            for row_name in self.__row_names:
                del self.__data[row_name][col_name]
            self.__col_names.remove(col_name)

    def resort_row(self, new_row_names:list[Union[int,str]]):
        new_row_names = [self._row_name_filter(row_name) for row_name in new_row_names]
        assert set(new_row_names) == set(self.__row_names), "new_row_names must contain all row names"
        if new_row_names == self.__row_names:
            return
        new_data = {}
        for row_name in new_row_names:
            new_data[row_name] = self.__data[row_name]
        self.__data = new_data
        self.__row_names = new_row_names

    def resort_column(self, new_col_names:list[Union[int,str]]):
        new_col_names = [self._col_name_filter(col_name) for col_name in new_col_names]
        assert set(new_col_names) == set(self.__col_names), "new_col_names must contain all col names"
        if new_col_names == self.__col_names:
            return
        for row_name in self.__row_names:
            new_row = {}
            for col_name in new_col_names:
                new_row[col_name] = self.__data[row_name][col_name]
            self.__data[row_name] = new_row
        self.__col_names = new_col_names

    def get_row(self, row_name:Union[int,str]):
        row_name = self._row_name_filter(row_name)
        if self.__key_assert(row_name, self.__row_names, True):
            return types.MappingProxyType(self.__data[row_name])

    def get_column(self, col_name:Union[int,str]):
        col_name = self._col_name_filter(col_name)
        if self.__key_assert(col_name, self.__col_names, True):
            return {row_name: self.__data[row_name][col_name] for row_name in self.__row_names}

    def tranverse(self, with_key=False):
        for row_name in self.__row_names:
            for col_name in self.__col_names:
                if with_key:
                    yield row_name, col_name, self.__data[row_name][col_name]
                else:
                    yield self.__data[row_name][col_name]

    def update(self, other:dict[ROWKEYT, dict[COLKETT, ITEM]]):
        for row_name, row in other.items():
            self.add_row(row_name, exist_ok=True)
            for col_name, value in row.items():
                self.add_column(col_name, exist_ok=True)
                self.__data[row_name][col_name] = value

    def merge(self, other:dict[str, dict[str, ITEM]], merge_func:Callable):
        assert callable(merge_func)
        for row_name, row in other.items():
            self.add_row(row_name, exist_ok=True)
            for col_name, value in row.items():
                self.add_column(col_name, exist_ok=True)
                self.__data[row_name][col_name] = merge_func(self.__data[row_name][col_name], value)

    def clean_invalid(self, judge_invalid_func:Callable):
        assert callable(judge_invalid_func)
        for row_name in self.__row_names:
            for col_name in self.__col_names:
                if judge_invalid_func(self.__data[row_name][col_name]):
                    self.__data[row_name][col_name] = self.gen_default_value()

    def __getitem__(self, keys:Union[int, str, tuple[Union[int, str]]]):
        if isinstance(keys, (int, str)):
            row_name = keys
            return self.get_row(row_name)
        elif isinstance(keys, tuple):
            assert len(keys) == 2, "Keys must be a tuple of length 2"
            row_name, col_name = keys
            col_name:COLKETT = self._col_name_filter(col_name)
            row = self.get_row(row_name)
            return row[col_name]
        else:
            raise ValueError
        
    def __setitem__(self, keys:tuple[Union[int, str]], value:ITEM = None):
        assert isinstance(keys, tuple), "Keys must be a tuple"
        if len(keys) == 2:
            row_name, col_name = keys
            row_name = self._row_name_filter(row_name)            
            col_name = self._col_name_filter(col_name)
            self.__data[row_name][col_name]= value
        else:
            raise ValueError("Keys must be a tuple of length 2")
        
    def __str__(self) -> str:
        if len(self.__data) == 0:
            return "Empty Table"

        max_col_widths = {col_name: len(col_name) for col_name in self.__col_names}
        str_data = {}

        _row_names = [str(x) for x in self.__row_names]
        _col_names = [str(x) for x in self.__col_names]

        for row_name in _row_names:
            str_data[row_name] = {}
            for col_name in _col_names:
                value = self[row_name, col_name]
                if isinstance(value, Iterable) and not isinstance(value, str):
                    str_value = type(value).__name__ + f"({len(value)})"
                else:
                    str_value = str(value)[:20]
                str_data[row_name][col_name] = str_value
                max_col_widths[col_name] = max(max_col_widths[col_name], len(str_value))

        # Create the table string
        max_col_widths['Row Name'] = max(len(x) for x in _row_names)
        table_str = f"{'':{max_col_widths['Row Name']}} | "
        for col_name in _col_names:
            table_str += f"{col_name.center(max_col_widths[col_name])} | "
        table_str += "\n"
        table_str += "-" * (sum(max_col_widths.values()) + len(max_col_widths) * 3) + "\n"

        for row_name in _row_names:
            table_str += f"{row_name.center(max_col_widths['Row Name'])} | "
            for col_name in _col_names:
                value:str = str_data[row_name][col_name]
                table_str += f"{value.center(max_col_widths[col_name])} | "
            table_str += "\n"

        return table_str

    @staticmethod
    def from_json(path):
        table_dict:dict[ROWKEYT, dict[COLKETT, ITEM]] = JsonIO.load_json(path)
        row_type = table_dict.keys().__iter__().__next__().__class__
        col_type = table_dict.values().__iter__().__next__().keys().__iter__().__next__().__class__
        table = Table(row_name_type=row_type, col_name_type=col_type)
        table.update(table_dict)
        return table
    
    def save(self, path):
        JsonIO.dump_json(path, self.__data)

