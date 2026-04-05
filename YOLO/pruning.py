from ultralytics import YOLO
import torch
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

#加载稀疏模型
model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\runs\detect\train\weights\best.pt")
net = model.model.cpu()
net.eval()

#剪枝比例：40%
prune_ratio = 0.4

bn_modules = []
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn_modules.append(m)

# 计算全局阈值
all_gamma = []
for bn in bn_modules:
    all_gamma.extend(bn.weight.data.abs().cpu().numpy())

threshold = np.percentile(all_gamma, prune_ratio * 100)
print(f"✅ 剪枝比例: {prune_ratio*100}%")
print(f"✅ 阈值: {threshold:.6f}")


for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        gamma = m.weight.data.abs()
        mask = gamma > threshold  # 保留大于阈值的通道
        keep_indices = torch.nonzero(mask).squeeze()

        if keep_indices.dim() == 0:
            keep_indices = keep_indices.unsqueeze(0)

        # 剪枝 BN 层
        m.weight.data = m.weight.data[keep_indices]
        m.bias.data = m.bias.data[keep_indices]
        m.running_mean = m.running_mean[keep_indices]
        m.running_var = m.running_var[keep_indices]

torch.save({"model": net}, "yolov8n_pruned.pt")