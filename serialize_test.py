from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import io

# # 打开图像文件
# image = Image.open(r'E:\shared\code\OLDT\datasets\morrison2\depths\train\000001.png')

# # 创建一个内存缓冲区
# buffer = io.BytesIO()

# # 将图像保存为JPEG格式到缓冲区
# image.save(buffer, format='PNG')

# # 将缓冲区的内容转换为字节字符串
# image_bytes = buffer.getvalue()

# # 将字节字符串转换为可打印的字符串
# image_string = image_bytes.decode('latin-1')

# # 将字节字符串转换为numpy数组
# image_array = np.frombuffer(image_string.encode('latin-1'), dtype=np.uint8)

# # 将numpy数组解码为图像
# image = cv2.imdecode(image_array, cv2.IMREAD_ANYDEPTH)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # 打印图像的字符串表示
# cv2.imwrite("pickle_test_image.jpg", image)
# np.save("pickle_test_image.npy", image)
# with open("pickle_image_bytes.pkl", 'wb')  as f:
#     pickle.dump({"rgb": image_bytes}, f)
# with open("pickle_image_string.pkl", 'wb')  as f:
#     pickle.dump({"rgb": image_string}, f)

ldmk = (10 * np.random.random((24, 480))).astype(np.float32)
with open("pickle_ldmk_array.pkl", 'wb')  as f:
    pickle.dump({"ldmk": ldmk}, f)

with open("pickle_ldmk_list.pkl", 'wb')  as f:
    pickle.dump({"ldmk": ldmk.tobytes()}, f)
