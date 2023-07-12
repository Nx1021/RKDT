from typing import Callable, Any, Tuple, Dict

def hint_function_b_params(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        # 提取函数B的参数类型提示
        b_params = func.__annotations__.get('x', None), func.__annotations__.get('y', None)
        # 根据函数B的参数类型提示生成提示字符串
        b_params_hint = f"({b_params[0]}: {b_params[1]})" if b_params else ""
        # 提示函数A的参数，并附加函数B的参数提示
        a_params_hint = func.__doc__ + b_params_hint
        # 设置函数A的参数提示
        wrapper.__doc__ = a_params_hint
        return func(*args, **kwargs)
    return wrapper

# 函数A
@hint_function_b_params
def function_a(*arg, **kw):
    """
    这是函数A的注释
    参数：
    *arg -- 可变位置参数
    **kw -- 可变关键字参数
    """
    pass

# 函数B
def function_b(x: int, y: str):
    """
    这是函数B的注释
    参数：
    x -- 整数参数
    y -- 字符串参数
    """
    pass

# 调用函数A，并获取参数提示
help(function_a)
