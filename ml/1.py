import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 加载图像并转换为灰度
image = Image.open('2.png').convert('L')
image_array = np.array(image)

# 进行二维傅里叶变换
f_transform = np.fft.fft2(image_array)
f_shift = np.fft.fftshift(f_transform) # 将低频移动到中心
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

# 绘制原始图像和傅里叶变换结果
plt.figure(figsize=(10, 5))

# 原始图像
plt.subplot(1, 2, 1)
plt.title("原始图像")
plt.imshow(image_array, cmap='gray')
plt.axis('off')

# 傅里叶变换幅度谱
plt.subplot(1, 2, 2)
plt.title("傅里叶变换幅度谱")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()