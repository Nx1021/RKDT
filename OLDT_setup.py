import sys
import os
import shutil
import MyLib
try:
    import MyLib
    sys.path.insert(0, MyLib.__path__[0])
    shutil.rmtree("./posture_6d")
    shutil.copytree(MyLib.posture_6d.__path__[0], "./posture_6d")
except:
    pass

# 获取当前脚本所在的目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录到当前脚本所在的目录
os.chdir(script_dir)
