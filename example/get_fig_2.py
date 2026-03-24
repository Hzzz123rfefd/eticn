import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread("datasets/camvid/train/0016E5_04590.png")

# 检查尺寸
h, w, c = img.shape
assert h > 100 and w > 100, "图像尺寸太小，请换大图"

# 选取 32×32 区域
start_y = 400
start_x = 100
block32 = img[start_y:start_y+320, start_x:start_x+320, :]

# 划分四个 16×16 块
block_tl = block32[0:160, 0:160, :]      # 左上
block_tr = block32[0:160, 160:320, :]     # 右上
block_bl = block32[160:320, 0:160, :]     # 左下
block_br = block32[160:320, 160:320, :]    # 右下（目标块）

# ==========================
# 帧内预测：使用上参考 + 左参考
# ==========================

top_ref = block_tr[-1, :, :]       # 上参考行
left_ref = block_bl[:, -1, :]      # 左参考列

pred_block = np.zeros((160,160,3), dtype=np.float32)

for i in range(160):
    for j in range(160):
        for c in range(3):
            pred_block[i, j, c] = (top_ref[j, c] + left_ref[i, c]) / 2

pred_block = pred_block.astype(np.uint8)

# 计算残差
residual = block_br.astype(np.int16) - pred_block.astype(np.int16)

# 为了显示，把残差归一化到 0-255
# residual_display = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
# residual_display = residual_display.astype(np.uint8)

# ==========================
# 显示所有块
# ==========================

cv2.imwrite("block_tl.png", block_tl)
cv2.imwrite("block_tr.png", block_tr)
cv2.imwrite("block_bl.png", block_bl)
cv2.imwrite("block_br.png", block_br)
cv2.imwrite("pred_block.png", pred_block)
cv2.imwrite("residual.png", residual)